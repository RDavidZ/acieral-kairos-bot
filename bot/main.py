"""
bot/main.py — Main 1H trading loop for Acieral Kairos Bot

Scheduling:
  HH:01 UTC  — on_candle_close  (1 min past each hour; ensures OANDA candle is settled)
  00:05 UTC  — on_daily_reset   (midnight equity sync + daily state reset)
  21:05 UTC  — on_daily_summary (end-of-US-session daily summary)

Retrain fires at 00:05 UTC every WALKFORWARD_RETRAIN_DAYS trading days
(entry models only; exit models are retrained quarterly on manual trigger).

Run:
  python -m bot.main
"""

import importlib
import logging
import os
import pickle
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

import config as cfg
from config import HARD_CONSTRAINTS, DISCOVERED_PARAMS
from execution.oanda_client import OandaClient
from execution.order_manager import OrderManager
from execution.price_stream import PriceStream
from notifications.telegram_bot import TelegramBot
from risk.manager import RiskManager
from risk.news_filter import should_block_entry, should_close_pre_news, get_size_multiplier
from strategy.feature_builder import build_live_features_v2
from data.fetcher_supplementary import fetch_supplementary

# ---------------------------------------------------------------------------
# Logging setup — file (rotating) + stream
# ---------------------------------------------------------------------------

_LOGS_DIR            = Path(__file__).parent.parent / "logs"
_MODELS_DIR          = Path(__file__).parent.parent / "ml" / "models"
_RETRAIN_STATE_FILE  = Path(__file__).parent.parent / "retrain_state.json"

_LOGS_DIR.mkdir(exist_ok=True)

_log_fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_date_fmt = "%Y-%m-%d %H:%M:%S"

_root = logging.getLogger()
_root.setLevel(logging.INFO)

_file_handler = RotatingFileHandler(
    _LOGS_DIR / "acieral_kairos.log",
    maxBytes=10 * 1024 * 1024,   # 10 MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_log_fmt, datefmt=_date_fmt))

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(_log_fmt, datefmt=_date_fmt))

_root.addHandler(_file_handler)
_root.addHandler(_stream_handler)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AcieralKairosBot
# ---------------------------------------------------------------------------

class AcieralKairosBot:
    """
    Main bot class — owns all subsystems and the APScheduler loop.

    Instantiate once, call start() to begin live trading.
    """

    def __init__(self) -> None:
        log.info("=== Acieral Kairos Bot initialising ===")

        # Reload config to pick up latest DISCOVERED_PARAMS
        importlib.reload(cfg)
        self.active_instruments = cfg.DISCOVERED_PARAMS["ACTIVE_INSTRUMENTS"]

        # --------------------------------------------------------- Load models
        self.entry_models: dict = {}
        self.exit_models:  dict = {}

        for instrument in self.active_instruments:
            entry_path = _MODELS_DIR / f"entry_{instrument}.pkl"
            exit_path  = _MODELS_DIR / f"exit_{instrument}.pkl"

            with open(entry_path, "rb") as f:
                self.entry_models[instrument] = pickle.load(f)
            with open(exit_path, "rb") as f:
                self.exit_models[instrument] = pickle.load(f)

            log.info("Models loaded | %s", instrument)

        # -------------------------------------------- OANDA client
        self.client = OandaClient()

        # -------------------------------------------- Risk manager
        account   = self.client.get_account()
        balance   = account["balance"]
        self.risk = RiskManager(initial_equity=balance)
        log.info("RiskManager ready | equity=£%.2f", balance)

        # ------------------------------------------ Order manager
        self.order_manager = OrderManager(
            self.risk, self.client,
            self.entry_models, self.exit_models,
        )

        # ------------------------------------------ Telegram
        self.telegram = TelegramBot()
        # Wire telegram into order_manager so it can send alerts
        self.order_manager.telegram = self.telegram
        # Wire kill switch callback — fires once when drawdown limit is breached
        self.risk.on_kill_switch = self._on_kill_switch_activated

        # ------------------------------------------ Scheduler
        self.scheduler = BackgroundScheduler(timezone="UTC")

        # ------------------------------------------ State
        self.retrain_interval           = HARD_CONSTRAINTS["WALKFORWARD_RETRAIN_DAYS"]
        self.trading_days_since_retrain = self._load_retrain_counter()
        log.info("Retrain counter restored: %d / %d", self.trading_days_since_retrain, self.retrain_interval)
        self._best_confidence:  dict[str, float] = {}   # per-instrument daily best
        self._daily_trade_results: dict[str, dict] = {}  # {inst: {status, pnl}}
        self._stream: PriceStream | None = None
        self._weekend_close_attempted: set[str] = set()

        # ------------------------------------------ Supplementary data (VIX/SPY)
        try:
            fetch_supplementary()
            log.info("Supplementary data initialised (VIX/SPY)")
        except Exception as exc:
            log.warning("fetch_supplementary on startup failed (non-blocking): %s", exc)

        # ------------------------------------------ Startup alert + DB
        self.telegram.send_startup(balance, self.active_instruments, "practice")

        try:
            from dashboard.db import register_bot, write_equity_snapshot  # type: ignore[import]
            register_bot(
                "acieral_kairos_v1", "Acieral Kairos Bot v1",
                self.active_instruments, balance,
                product="acieral", is_live=False,
            )
            write_equity_snapshot("acieral_kairos_v1", balance)
            log.info("Dashboard registration complete")
        except ImportError:
            pass   # dashboard module not yet deployed
        except Exception as exc:
            log.warning("Dashboard registration failed (non-blocking): %s", exc)

        # ------------------------------------------ Startup reconciliation
        try:
            self.order_manager.reconcile_open_trades()
        except Exception as exc:
            log.warning("Startup reconciliation failed (non-blocking): %s", exc)
        self.order_manager.reconcile_db_trades()

        log.info(
            "Bot ready | instruments=%d | equity=£%.2f",
            len(self.active_instruments), balance,
        )

        # ------------------------------------------ Price streaming
        self._stream = PriceStream(
            api_key=os.environ["OANDA_API_KEY"],
            account_id=os.environ["OANDA_ACCOUNT_ID"],
            instruments=self.active_instruments,
            on_tick=self._on_price_tick,
            environment=os.environ.get("OANDA_ENVIRONMENT", "practice"),
        )
        self._stream.start()
        log.info("Price streaming started for %d instruments", len(self.active_instruments))

    # ------------------------------------------------------------------
    # Retrain counter persistence
    # ------------------------------------------------------------------

    def _load_retrain_counter(self) -> int:
        import json
        if _RETRAIN_STATE_FILE.exists():
            try:
                data = json.loads(_RETRAIN_STATE_FILE.read_text())
                return int(data.get("trading_days_since_retrain", 0))
            except Exception:
                return 0
        return 0

    def _save_retrain_counter(self) -> None:
        import json
        try:
            _RETRAIN_STATE_FILE.write_text(
                json.dumps({"trading_days_since_retrain": self.trading_days_since_retrain})
            )
        except Exception as exc:
            log.warning("Failed to save retrain counter: %s", exc)

    # ------------------------------------------------------------------
    # Kill switch alert — called once when drawdown limit is breached
    # ------------------------------------------------------------------

    def _on_kill_switch_activated(self) -> None:
        """Fired by RiskManager on the False → True kill switch transition."""
        msg = (
            "🚨 <b>KILL SWITCH ACTIVATED</b>\n"
            f"Daily drawdown limit reached. Bot has stopped trading for today.\n"
            f"Equity: £{self.risk.equity:,.2f} | Floor: £{self.risk.drawdown_floor:,.2f}"
        )
        try:
            self.telegram.send(msg)
        except Exception as exc:
            log.error("Kill switch Telegram alert failed: %s", exc)

    # ------------------------------------------------------------------
    # Streaming tick handler — called from PriceStream daemon thread
    # ------------------------------------------------------------------

    def _on_price_tick(self, instrument: str, bid: float, ask: float, now: datetime) -> None:
        """Called from streaming thread on every price tick."""
        try:
            self.order_manager.manage_on_tick(instrument, bid, ask, now)
        except Exception as exc:
            log.error("[%s] manage_on_tick error: %s", instrument, exc)

    # ------------------------------------------------------------------
    # Weekend trade management — index EOD cleanup only
    # ------------------------------------------------------------------

    def _manage_open_trades_weekend(self, now: datetime) -> None:
        """
        Called on weekends instead of full candle processing.
        Index positions found open should have been force-closed at EOD Friday —
        close them now. Forex positions are protected by OANDA's hard SL; log only.
        No entry evaluation, no exit model, no candle fetch.
        """
        _INDEX = HARD_CONSTRAINTS["INDEX_INSTRUMENTS"]
        open_trades = self.risk.get_all_open_trades()

        for instrument, _trade in open_trades.items():
            if instrument in _INDEX:
                if instrument in self._weekend_close_attempted:
                    log.warning(
                        "[%s] Weekend: close already attempted this weekend — skipping retry",
                        instrument,
                    )
                    continue
                self._weekend_close_attempted.add(instrument)
                log.warning(
                    "[%s] Weekend: open index trade found — force-closing as EOD",
                    instrument,
                )
                try:
                    self.order_manager.force_close(instrument, "EOD", now)
                except Exception as exc:
                    log.error("[%s] Weekend force_close failed: %s", instrument, exc)
            else:
                log.warning(
                    "[%s] Weekend: open forex trade found — OANDA hard SL is active, no action taken",
                    instrument,
                )

    # ------------------------------------------------------------------
    # Candle close handler — called at HH:01 UTC
    # ------------------------------------------------------------------

    def on_candle_close(self, now: datetime | None = None) -> None:
        """
        Process the just-closed H1 candle for all active instruments.

        Steps per instrument:
          1. Skip if market closed
          2. Fetch 200 H1 candles
          3. Build live features (returns last row)
          4. Manage open trade (if any) — SL update, exit model, hard close
          5. Evaluate new entry (if no open trade)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Weekend — skip entries and exit model but manage any open index trades
        if now.weekday() >= 5:
            self._manage_open_trades_weekend(now)
            return

        # Refresh cached GBP/USD and GBP/EUR rates on the scheduler thread
        # so manage_on_tick never blocks on a live HTTP fetch during P&L computation.
        try:
            self.order_manager.refresh_gbpusd_rate()
        except Exception as exc:
            log.warning("GBP rate refresh failed (non-blocking): %s", exc)

        log.info("on_candle_close | %s", now.strftime("%Y-%m-%d %H:%M UTC"))

        for instrument in self.active_instruments:
            try:
                self._process_instrument(instrument, now)
            except Exception as exc:
                log.error("[%s] on_candle_close error: %s", instrument, exc, exc_info=True)
            try:
                from risk.news_filter import evaluate_news_status
                from dashboard.db import write_news_status
                ns = evaluate_news_status(instrument, now)
                write_news_status(
                    bot_id='acieral_kairos_v1',
                    instrument=ns["instrument"],
                    status=ns["status"],
                    reason=ns["reason"],
                    size_multiplier=ns["size_multiplier"],
                    next_event_time=ns["next_event_time"],
                    next_event_title=ns["next_event_title"],
                    next_event_currency=ns["next_event_currency"],
                )
            except Exception as e:
                log.warning(f"News status DB write failed for {instrument}: {e}")

    def _process_instrument(self, instrument: str, now: datetime) -> None:
        """Single-instrument candle processing (error-isolated per instrument)."""

        # 1. Market open check
        # Outside session hours: still manage open trades but skip entry evaluation
        market_open = self.client.is_market_open(instrument, now=now)
        open_trade = self.risk.get_open_trade(instrument)
        if not market_open and open_trade is None:
            # Market closed, no open trade — nothing to do
            return

        if not market_open and open_trade is not None:
            # Market closed but trade is open — manage it (SL/trail/EOD) but skip exit model
            # Fetch candles to get current price for SL and trail checks
            try:
                h1_df = self.client.get_latest_candles(instrument, granularity="H1", count=50)
            except Exception as exc:
                log.error("[%s] Candle fetch failed (outside session): %s", instrument, exc)
                return
            if len(h1_df) < 10:
                return
            try:
                current_candle = build_live_features_v2(instrument, h1_df, now=now)
            except Exception as exc:
                log.error("[%s] build_live_features_v2 failed (outside session): %s", instrument, exc)
                return

            # ATR patch for restored trades
            if open_trade.get("atr_at_entry", 0.0) == 0.0:
                try:
                    open_trade["atr_at_entry"] = float(current_candle["atr_14"])
                    log.info(
                        "[%s] Patched atr_at_entry=%.5f (outside session)",
                        instrument, open_trade["atr_at_entry"],
                    )
                except Exception as exc:
                    log.warning("[%s] Could not patch atr_at_entry: %s", instrument, exc)

            # Manage trade — SL check, trail update, EOD check
            # Pass a flag to manage_open_trade to skip exit model outside session
            close_now, news_reason = should_close_pre_news(instrument, now)
            if close_now:
                log.warning("[%s] Pre-news close (outside session) — %s", instrument, news_reason)
                eq_before   = self.risk.equity
                exit_reason = self.order_manager.force_close(instrument, "NEWS_CLOSE", now)
            else:
                eq_before   = self.risk.equity
                exit_reason = self.order_manager.manage_open_trade(
                    instrument, current_candle, now, skip_exit_model=True
                )
            if exit_reason:
                pnl = round(self.risk.equity - eq_before, 2)
                self._daily_trade_results[instrument] = {
                    "status": "win" if pnl >= 0 else "loss",
                    "pnl":    pnl,
                }
                log.info(
                    "[%s] Trade closed outside session | reason=%s | P&L=£%.2f",
                    instrument, exit_reason, pnl,
                )
            return  # Never evaluate entry outside session

        # Market is open — proceed with normal flow (existing code continues unchanged from here)

        # 2. Fetch latest 500 H1 candles (500 bars = ~21 days; gives swing
        #    detector enough history to match backtest structural analysis)
        try:
            h1_df = self.client.get_latest_candles(instrument, granularity="H1", count=500)
        except Exception as exc:
            log.error("[%s] Candle fetch failed: %s", instrument, exc)
            return

        if len(h1_df) < 50:
            log.warning("[%s] Only %d candles returned — skipping", instrument, len(h1_df))
            return

        # 3. Build live feature row
        try:
            current_candle = build_live_features_v2(instrument, h1_df, now=now)
        except Exception as exc:
            log.error("[%s] build_live_features_v2 failed: %s", instrument, exc, exc_info=True)
            return

        # 4. Manage open trade (before evaluating new entry)
        if open_trade is not None:
            # Patch ATR for restored trades — reconcile sets atr_at_entry=0.0
            # which breaks trailing SL and exit model. Use current candle ATR instead.
            if open_trade.get("atr_at_entry", 0.0) == 0.0:
                try:
                    open_trade["atr_at_entry"] = float(current_candle["atr_14"])
                    log.info(
                        "[%s] Patched atr_at_entry=%.5f for restored trade",
                        instrument, open_trade["atr_at_entry"],
                    )
                except Exception as exc:
                    log.warning("[%s] Could not patch atr_at_entry: %s", instrument, exc)

            # 4a. Pre-news close — High impact event ≤15 min away
            close_now, news_reason = should_close_pre_news(instrument, now)
            if close_now:
                log.warning("[%s] Pre-news close — %s", instrument, news_reason)
                eq_before   = self.risk.equity
                exit_reason = self.order_manager.force_close(
                    instrument, "NEWS_CLOSE", now
                )
            else:
                eq_before   = self.risk.equity
                exit_reason = self.order_manager.manage_open_trade(
                    instrument, current_candle, now
                )

            if exit_reason:
                pnl = round(self.risk.equity - eq_before, 2)
                self._daily_trade_results[instrument] = {
                    "status": "win" if pnl >= 0 else "loss",
                    "pnl":    pnl,
                }
                log.info(
                    "[%s] Trade closed | reason=%s | P&L=£%.2f",
                    instrument, exit_reason, pnl,
                )
            return  # don't evaluate entry on the same bar a trade was managed

        # 5. Evaluate new entry
        # 5a. News block — High impact event within ±30 min
        blocked, block_reason = should_block_entry(instrument, now)
        if blocked:
            log.info("[%s] News block — %s", instrument, block_reason)
            return

        size_mult = get_size_multiplier(instrument, now)
        opened, confidence = self.order_manager.attempt_entry(
            instrument, current_candle, current_candle, now,
            size_multiplier=size_mult,
        )
        self._best_confidence[instrument] = max(
            self._best_confidence.get(instrument, 0.0), confidence
        )
        if opened:
            self._daily_trade_results[instrument] = {
                "status": "open",
                "pnl":    0.0,
            }
            log.info("[%s] Entry taken | conf=%.3f", instrument, confidence)

    # ------------------------------------------------------------------
    # Daily reset — 00:05 UTC
    # ------------------------------------------------------------------

    def on_daily_reset(self, now: datetime | None = None) -> None:
        """
        Reset daily state, sync equity from OANDA, send skip alerts.
        Called at 00:05 UTC.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        log.info("on_daily_reset | %s", now.strftime("%Y-%m-%d %H:%M UTC"))

        # 0. Update supplementary data (VIX, SPY volume) once per day
        try:
            fetch_supplementary()
            log.info("Supplementary data updated (VIX/SPY)")
        except Exception as exc:
            log.warning("fetch_supplementary failed (non-blocking): %s", exc)

        # 1. Fetch equity from OANDA first so daily_reset sets day_start_equity
        #    from the real balance rather than a potentially stale in-memory value.
        new_equity: float | None = None
        try:
            account    = self.client.get_account()
            new_equity = account["balance"]
            log.info("Equity synced from OANDA | £%.2f", new_equity)
        except Exception as exc:
            log.warning("Equity sync failed — daily_reset will use in-memory equity: %s", exc)

        # 2. Daily risk reset — passes fetched balance (or None on fetch failure)
        self.risk.daily_reset(now, new_equity=new_equity)

        # 3. Reconcile open trades with OANDA
        try:
            self.order_manager.reconcile_open_trades()
        except Exception as exc:
            log.warning("Daily reconciliation failed (non-blocking): %s", exc)
        self.order_manager.reconcile_db_trades()

        # 4. Write equity snapshot to dashboard
        try:
            from dashboard.db import write_equity_snapshot  # type: ignore[import]
            write_equity_snapshot("acieral_kairos_v1", self.risk.equity)
        except ImportError:
            pass
        except Exception as exc:
            log.warning("Equity snapshot write failed: %s", exc)

        # 5. Increment retrain counter
        self.trading_days_since_retrain += 1
        self._save_retrain_counter()
        log.info(
            "Trading days since retrain: %d / %d",
            self.trading_days_since_retrain, self.retrain_interval,
        )

        # 6. Send skip alerts for instruments with no trade today
        for instrument in self.active_instruments:
            if instrument not in self._daily_trade_results:
                best_conf = self._best_confidence.get(instrument, 0.0)
                curve     = cfg.DISCOVERED_PARAMS["CONFIDENCE_CURVE"][instrument]
                # Use noon (12:00) as representative threshold for the skip alert
                threshold = float(curve.get(12, 0.50))
                self.telegram.send_skipped(instrument, best_conf, threshold)

        # 7. Reset daily tracking
        self._best_confidence       = {}
        self._daily_trade_results   = {}
        self._weekend_close_attempted.clear()

        # 8. Trigger retrain if due
        if self.trading_days_since_retrain >= self.retrain_interval:
            try:
                self.on_retrain()
            except Exception as exc:
                log.error("Retrain failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Daily summary — 21:05 UTC
    # ------------------------------------------------------------------

    def on_daily_summary(self, now: datetime | None = None) -> None:
        """
        Send the daily summary Telegram alert at 21:05 UTC.
        Compiles closed-trade results from _daily_trade_results.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Build results dict for telegram (only include closed trades)
        summary_results: dict = {}
        for instrument in self.active_instruments:
            result = self._daily_trade_results.get(instrument, {})
            st     = result.get("status", "skip")
            # 'open' means trade not yet closed — show as skip for summary
            if st == "open":
                st = "skip"
            summary_results[instrument] = {
                "status": st,
                "pnl":    result.get("pnl", 0.0),
            }

        status        = self.risk.get_status()
        equity        = status["equity"]
        day_pnl       = status["day_pnl"]
        days_to_retrain = max(
            0, self.retrain_interval - self.trading_days_since_retrain
        )

        date_str = now.strftime("%a %d %b")
        self.telegram.send_daily_summary(
            date_str, summary_results, equity, day_pnl, days_to_retrain
        )

        log.info(
            "Daily summary sent | P&L=£%.2f | equity=£%.2f",
            day_pnl, equity,
        )

    # ------------------------------------------------------------------
    # Retrain — entry models only
    # ------------------------------------------------------------------

    def on_retrain(self) -> None:
        """
        Retrain entry models for all active instruments.
        Called from on_daily_reset when trading_days_since_retrain >= interval.
        Entry models only — exit models retrained quarterly (manual trigger).
        """
        log.info("=== Retrain starting ===")

        from data.fetcher        import fetch_all
        from data.preprocessor   import preprocess_all
        from strategy.feature_builder import build_all
        from ml.labeler          import label_all
        from ml.trainer          import train_instrument

        # 1. Fetch latest data
        try:
            log.info("Fetching latest data…")
            fetch_all()
        except Exception as exc:
            log.error("Data fetch failed — retrain aborted: %s", exc)
            return

        # 2. Preprocess
        try:
            log.info("Preprocessing…")
            preprocess_all()
        except Exception as exc:
            log.error("Preprocess failed — retrain aborted: %s", exc)
            return

        # 3. Build features
        try:
            log.info("Building features…")
            build_all()
        except Exception as exc:
            log.error("Feature build failed — retrain aborted: %s", exc)
            return

        # 4–6. Label + retrain per instrument
        retrain_results: dict = {}

        for instrument in self.active_instruments:
            try:
                inst_params = cfg.DISCOVERED_PARAMS["INSTRUMENT_PARAMS"][instrument]
                N = int(inst_params["N"])
                T = float(inst_params["T"])

                log.info("[%s] Labelling N=%d T=%.4f…", instrument, N, T)
                label_all(instruments=[instrument], N_values=[N], T_values=[T])

                log.info("[%s] Training…", instrument)
                result = train_instrument(instrument, N, T)

                # Reload model into memory
                model_path = _MODELS_DIR / f"entry_{instrument}.pkl"
                with open(model_path, "rb") as f:
                    self.entry_models[instrument] = pickle.load(f)
                self.order_manager.entry_models[instrument] = self.entry_models[instrument]

                retrain_results[instrument] = {
                    "mean_auc": result["mean_val_auc"],
                    "n_folds":  result["n_folds"],
                }
                log.info(
                    "[%s] Retrain complete | AUC=%.4f | folds=%d",
                    instrument, result["mean_val_auc"], result["n_folds"],
                )

            except Exception as exc:
                log.error("[%s] Retrain failed: %s", instrument, exc, exc_info=True)
                retrain_results[instrument] = {"mean_auc": float("nan"), "n_folds": 0}

        self.trading_days_since_retrain = 0
        self._save_retrain_counter()

        self.telegram.send_retrain_complete(retrain_results)
        log.info("=== Retrain complete ===")

    # ------------------------------------------------------------------
    # News DB sync — every 12 hours
    # ------------------------------------------------------------------

    def _news_db_sync_job(self) -> None:
        try:
            from data.news_fetcher import write_events_to_db
            count = write_events_to_db('acieral_kairos_v1')
            log.info("News DB sync: %d events written", count)
        except Exception as e:
            log.warning(f"News DB sync failed: {e}")

    # ------------------------------------------------------------------
    # Heartbeat — every 30 minutes
    # ------------------------------------------------------------------

    def _heartbeat_job(self) -> None:
        try:
            from dashboard.db import write_heartbeat
            write_heartbeat('acieral_kairos_v1')
            log.debug("Heartbeat written")
        except Exception as e:
            log.warning(f'Heartbeat failed: {e}')

    # ------------------------------------------------------------------
    # Intraday equity snapshot — every 30 minutes
    # ------------------------------------------------------------------

    def _equity_snapshot_job(self) -> None:
        try:
            from dashboard.db import write_equity_snapshot
            write_equity_snapshot('acieral_kairos_v1', self.risk.equity, source='practice')
            log.debug("Equity snapshot written | £%.2f", self.risk.equity)
        except Exception as e:
            log.warning(f'Equity snapshot job failed: {e}')

    # ------------------------------------------------------------------
    # Feature cache update — HH:00 UTC
    # ------------------------------------------------------------------

    def _update_feature_cache_job(self) -> None:
        """
        Incrementally update the H1 feature cache for all active instruments.
        Fetches only the latest 10 H1 bars per instrument, appends any new bars
        to the raw H1 parquet, then rebuilds the feature cache only when new
        bars are present (at most 1 new bar per instrument per hour).

        Runs at HH:00 UTC — one minute before candle close at HH:01.
        Typical runtime: ~2.5s per instrument × 8 instruments = ~20s total,
        well within the 60-second window before the candle close job.
        """
        import config
        import pandas as pd

        active    = config.DISCOVERED_PARAMS.get(
            "ACTIVE_INSTRUMENTS",
            config.HARD_CONSTRAINTS["ALL_INSTRUMENTS"],
        )
        cache_dir = Path(__file__).parent.parent / "data" / "cache"

        for instrument in active:
            try:
                # Step 1: fetch only the last 10 H1 bars from OANDA
                new_bars = self.client.get_latest_candles(
                    instrument, granularity="H1", count=10
                )
                if new_bars is None or new_bars.empty:
                    continue

                new_bars["time"] = pd.to_datetime(new_bars["time"], utc=True)

                # Step 2: append truly new bars to raw H1 parquet
                raw_path = cache_dir / f"{instrument}_H1.parquet"
                if not raw_path.exists():
                    log.warning("[%s] Raw H1 cache missing — skipping incremental update", instrument)
                    continue

                raw = pd.read_parquet(raw_path)
                raw["time"] = pd.to_datetime(raw["time"], utc=True)
                last_cached = raw["time"].max()
                truly_new   = new_bars[new_bars["time"] > last_cached]

                if truly_new.empty:
                    log.debug("[%s] Feature cache: no new bars", instrument)
                    continue

                raw = pd.concat([raw, truly_new], ignore_index=True)
                raw = (raw.drop_duplicates(subset=["time"])
                          .sort_values("time")
                          .reset_index(drop=True))
                raw.to_parquet(raw_path, index=False)

                # Step 3: run preprocessor for this instrument to update processed parquet
                try:
                    from data.preprocessor import preprocess_instrument
                    preprocess_instrument(instrument)
                except Exception as exc:
                    log.error("[%s] Preprocessor failed — feature cache not updated for this bar: %s", instrument, exc)
                    continue

                # Step 4: rebuild feature cache from updated processed parquet
                feat_path = cache_dir / f"{instrument}_H1_features.parquet"
                last_feat_time = None
                if feat_path.exists():
                    existing = pd.read_parquet(feat_path)
                    existing["time"] = pd.to_datetime(existing["time"], utc=True)
                    last_feat_time = existing["time"].max()

                if last_feat_time is None or truly_new["time"].max() > last_feat_time:
                    from strategy.feature_builder import build_features
                    updated = build_features(instrument)
                    updated.to_parquet(feat_path, index=False)
                    log.info(
                        "[%s] Feature cache updated | +%d new bars | latest=%s",
                        instrument, len(truly_new),
                        updated["time"].max() if "time" in updated.columns else "?",
                    )
                else:
                    log.debug("[%s] Feature cache already current", instrument)

            except Exception as exc:
                log.error("[%s] Feature cache update failed: %s", instrument, exc)

    # ------------------------------------------------------------------
    # Start — schedule jobs and enter keep-alive loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Register cron jobs and start the scheduler.
        Blocks until KeyboardInterrupt.
        """
        # HH:00 UTC — update feature cache before candle close evaluation
        self.scheduler.add_job(
            self._update_feature_cache_job,
            CronTrigger(minute=0, timezone="UTC"),
            id="feature_cache_update",
            name="Feature cache update",
            misfire_grace_time=30,
        )

        # HH:01 UTC — candle close handler
        self.scheduler.add_job(
            self.on_candle_close,
            CronTrigger(minute=1, timezone="UTC"),
            id="candle_close",
            name="H1 candle close",
            misfire_grace_time=60,
        )

        # 00:05 UTC — daily reset
        self.scheduler.add_job(
            self.on_daily_reset,
            CronTrigger(hour=0, minute=5, timezone="UTC"),
            id="daily_reset",
            name="Daily reset",
            misfire_grace_time=300,
        )

        # 21:05 UTC — daily summary
        self.scheduler.add_job(
            self.on_daily_summary,
            CronTrigger(hour=21, minute=5, timezone="UTC"),
            id="daily_summary",
            name="Daily summary",
            misfire_grace_time=300,
        )

        # Every 30 min — intraday equity snapshot for dashboard curve
        self.scheduler.add_job(
            self._equity_snapshot_job,
            CronTrigger(minute='0,30'),
            id='equity_snapshot',
            name='Equity snapshot',
            misfire_grace_time=60,
        )

        # Every 30 min — bot heartbeat so dashboard can show liveness
        self.scheduler.add_job(
            self._heartbeat_job,
            CronTrigger(minute='0,30'),
            id='heartbeat',
            name='Heartbeat',
            misfire_grace_time=60,
        )

        # Every 12 hours — sync news event cache to dashboard DB
        self.scheduler.add_job(
            self._news_db_sync_job,
            CronTrigger(hour='*/12'),
            id='news_db_sync',
            name='News DB sync',
            misfire_grace_time=300,
        )

        self.scheduler.start()
        log.info("Acieral Kairos Bot running. Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(60)
                # Heartbeat log every hour at HH:30
                now = datetime.now(timezone.utc)
                if now.minute == 30:
                    status = self.risk.get_status()
                    log.info(
                        "Heartbeat | equity=£%.2f | open_trades=%d | "
                        "kill_switch=%s",
                        status["equity"],
                        status["open_trades"],
                        status["kill_switch"],
                    )
        except KeyboardInterrupt:
            log.info("Shutdown requested — stopping scheduler")
            self.scheduler.shutdown()
            if self._stream:
                self._stream.stop()
            log.info("Bot stopped cleanly")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bot = AcieralKairosBot()
    bot.start()
