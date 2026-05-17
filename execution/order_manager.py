"""
execution/order_manager.py — Full trade lifecycle management for Acieral Kairos Bot

Coordinates between entry model, exit model, risk manager, and OANDA client.
Each public method corresponds to one event in the 1H loop:
  - attempt_entry   : called on every bar for instruments without an open trade
  - manage_open_trade : called on every bar for instruments with an open trade

DB writes and Telegram alerts are best-effort: failures are logged but never
block a trade operation.

Dependency order: data/ → strategy/ → ml/ → backtest/ → risk/ → execution/
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from config import HARD_CONSTRAINTS, DISCOVERED_PARAMS
from ml.labeler import FEATURE_COLS
from ml.exit_trainer import EXIT_FEATURE_COLS, TC_FEATURES
from risk.manager import RiskManager
from execution.oanda_client import OandaClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_HC              = HARD_CONSTRAINTS
FOREX_PAIRS      = _HC["FOREX_PAIRS"]
INDEX_INSTRUMENTS = _HC["INDEX_INSTRUMENTS"]
SL_ATR_MULT      = float(_HC["BACKTEST_SL_ATR_MULT"])
QUOTE_TYPE       = _HC["INSTRUMENT_QUOTE_TYPE"]

# Index session close times (UTC) — hard EOD
_SESSION_CLOSE_UTC: dict[str, tuple[int, int]] = {
    "SPX500_USD": (21,  0),
    "NAS100_USD": (21,  0),
    "DE30_EUR":   (16, 30),
}

MAX_BARS_HARD = 24   # hard max hold duration for forex (1 trading day = 24 x 1H bars)


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Manages the full trade lifecycle for Acieral Kairos Bot.

    Instantiate once at bot startup.  Pass ``telegram`` if the notifications
    module is available; it defaults to None and is silently skipped.

    Parameters
    ----------
    risk_manager  : live RiskManager instance
    oanda_client  : live OandaClient instance
    entry_models  : {instrument: loaded XGBoost Booster/Sklearn API model}
    exit_models   : {instrument: loaded XGBoost Booster/Sklearn API model}
    telegram      : optional TelegramBot instance for alerts
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        oanda_client: OandaClient,
        entry_models: dict,
        exit_models: dict,
        telegram=None,
    ) -> None:
        self.risk          = risk_manager
        self.client        = oanda_client
        self.entry_models  = entry_models
        self.exit_models   = exit_models
        self.telegram      = telegram

        # Cached FX conversion rates — refreshed every candle close by the
        # scheduler thread so manage_on_tick never blocks on a live HTTP fetch.
        self._gbpusd_rate:         float            = 1.0
        self._gbpeur_rate:         float            = 1.0
        self._gbpusd_rate_updated: datetime | None  = None

        # Reload DISCOVERED_PARAMS from config each time (picks up retrain writes)
        import importlib
        importlib.reload(cfg)
        self.params = cfg.DISCOVERED_PARAMS

        log.info(
            "OrderManager ready | instruments=%s",
            sorted(self.entry_models.keys()),
        )

    # ------------------------------------------------------------------
    # FX rate cache — called from scheduler thread (on_candle_close)
    # ------------------------------------------------------------------

    def refresh_gbpusd_rate(self) -> None:
        """
        Refresh the cached GBP/USD and GBP/EUR rates from OANDA.
        Called on the scheduler thread (HH:01) so manage_on_tick never
        blocks on a live HTTP fetch during P&L computation.
        Falls back silently to the last known rate on failure.
        """
        try:
            bid, _ = self.client.get_current_price('GBP_USD')
            if bid and bid > 0:
                self._gbpusd_rate = bid
                self._gbpusd_rate_updated = datetime.now(timezone.utc)
        except Exception as e:
            log.warning(
                f'GBP/USD rate refresh failed — using cached {self._gbpusd_rate:.5f}: {e}'
            )
        try:
            eurgbp_bid, _ = self.client.get_current_price('EUR_GBP')
            if eurgbp_bid and eurgbp_bid > 0:
                self._gbpeur_rate = 1.0 / eurgbp_bid
        except Exception as e:
            log.warning(
                f'EUR_GBP rate refresh failed — using cached {self._gbpeur_rate:.5f}: {e}'
            )

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------

    def evaluate_entry(
        self,
        instrument: str,
        features: pd.Series,
        now: datetime,
    ) -> tuple[int, float]:
        """
        Run the entry model for one instrument at one candle close.

        Parameters
        ----------
        instrument : OANDA instrument name
        features   : single row from the H1 features DataFrame
                     (must contain all FEATURE_COLS)
        now        : current UTC datetime (tz-aware)

        Returns
        -------
        (predicted_class, confidence)
          predicted_class : 0=No trade, 1=Short, 2=Long
          confidence      : max(predict_proba) — model's best-class probability
        Returns (0, confidence) if class == 0 or confidence < hourly threshold.
        """
        model = self.entry_models[instrument]

        # Build feature vector — XGBoost handles NaN natively
        X = features[FEATURE_COLS].values.reshape(1, -1)

        probas           = model.predict_proba(X)[0]        # shape (3,)
        predicted_class  = int(np.argmax(probas))
        confidence       = float(probas[predicted_class])

        # Hourly confidence threshold (integer key in CONFIDENCE_CURVE)
        curve     = self.params["CONFIDENCE_CURVE"][instrument]
        threshold = float(curve.get(now.hour, 0.50))

        dir_label = {0: "NoTrade", 1: "SHORT", 2: "LONG"}.get(predicted_class, str(predicted_class))
        log.info("%s: model score — dir=%d conf=%.3f threshold=%.3f", instrument, predicted_class, confidence, threshold)

        if predicted_class == 0 or confidence < threshold:
            log.info("%s: no signal (%s conf=%.3f)", instrument, dir_label, confidence)
            return 0, confidence
        log.info("%s: SIGNAL — %s conf=%.3f", instrument, dir_label, confidence)
        return predicted_class, confidence

    # ------------------------------------------------------------------
    # Exit evaluation
    # ------------------------------------------------------------------

    def evaluate_exit(
        self,
        instrument: str,
        features: pd.Series,
        trade: dict,
        now: datetime,
    ) -> tuple[bool, float]:
        """
        Run the exit model for one instrument on one bar.

        Parameters
        ----------
        instrument : OANDA instrument name
        features   : current H1 candle features (must contain FEATURE_COLS)
        trade      : open trade dict from risk_manager.get_open_trade()
        now        : current UTC datetime (tz-aware)

        Returns
        -------
        (should_exit, p_exit)
        """
        model = self.exit_models[instrument]

        close     = float(features["close"])
        atr       = float(trade["atr_at_entry"])
        direction = int(trade["direction"])
        entry     = float(trade["entry_price"])
        sl        = float(trade["sl_price"])
        mfe_atr   = float(trade.get("mfe_atr", 0.0))

        # Compute trade-context features
        unrealised_atr    = (close - entry) * direction / atr if atr > 0 else 0.0
        pct_mfe_given_back = (
            (mfe_atr - unrealised_atr) / mfe_atr if mfe_atr > 0 else 0.0
        )
        dist_to_sl_atr = (
            (close - sl) * direction / atr if atr > 0 else 0.0
        )

        tc = {
            "tc_bars_held":          float(trade.get("bars_held", 0)),
            "tc_unrealised_atr":     unrealised_atr,
            "tc_mfe_atr":            mfe_atr,
            "tc_mae_atr":            float(trade.get("mae_atr", 0.0)),
            "tc_pct_mfe_given_back": pct_mfe_given_back,
            "tc_direction":          float(direction),
            "tc_dist_to_sl_atr":     dist_to_sl_atr,
        }

        # Build full exit feature vector
        market_vals = features[FEATURE_COLS].values
        tc_vals     = np.array([tc[f] for f in TC_FEATURES], dtype=float)
        X           = np.concatenate([market_vals, tc_vals]).reshape(1, -1)

        p_exit = float(model.predict_proba(X)[0][1])

        threshold = float(
            self.params["INSTRUMENT_PARAMS"][instrument]["exit_threshold"]
        )

        return p_exit >= threshold, p_exit

    # ------------------------------------------------------------------
    # Bar-by-bar open trade management
    # ------------------------------------------------------------------

    def manage_open_trade(
        self,
        instrument: str,
        candle: pd.Series,
        now: datetime,
        skip_exit_model: bool = False,
    ) -> str | None:
        """
        Called on every H1 candle close while a trade is open.

        ``candle`` must contain FEATURE_COLS plus high, low, close, atr_14.
        Returns the exit_reason string if the trade was closed, else None.

        Exit priority (checked in order):
          1. SL hit (candle high/low breaches stop)
          2. Hard EOD close (index session end)
          3. Forex Friday 21:00 UTC close
          4. Max bars (24 bars — 1 trading day)
          5. Exit model (only when trail is active)
        """
        trade = self.risk.get_open_trade(instrument)
        if trade is None:
            return None

        close     = float(candle["close"])
        high      = float(candle["high"])
        low       = float(candle["low"])
        atr       = float(candle.get("atr_14", trade["atr_at_entry"]))
        direction = int(trade["direction"])
        entry     = float(trade["entry_price"])

        # ------------------------------------------------------------------
        # Step 2 — Update MFE, MAE, bars_held
        # ------------------------------------------------------------------
        unrealised_atr = (close - entry) * direction / atr if atr > 0 else 0.0
        trade["mfe_atr"]   = max(trade.get("mfe_atr", 0.0), unrealised_atr)
        trade["mae_atr"]   = max(trade.get("mae_atr", 0.0), -unrealised_atr, 0.0)
        trade["bars_held"] = trade.get("bars_held", 0) + 1

        unrealised_pnl_gbp = self._compute_pnl_gbp(instrument, trade, close, now)

        # ------------------------------------------------------------------
        # Step 3 — Update trailing SL
        # ------------------------------------------------------------------
        trail_mult  = float(
            self.params["INSTRUMENT_PARAMS"][instrument]["trail_atr_mult"]
        )
        mfe_atr     = trade["mfe_atr"]
        trail_atr   = float(trade["atr_at_entry"])
        trail_active = mfe_atr >= trail_mult

        # Fire trail activation alert once per trade
        if trail_active and not trade.get("trail_alert_sent", False):
            trade["trail_alert_sent"] = True
            current_price = float(candle["close"])
            try:
                self.telegram.send_trail_activated(
                    instrument=instrument,
                    direction="LONG" if direction == 1 else "SHORT",
                    entry_price=entry,
                    current_price=current_price,
                    unrealised_pnl=unrealised_pnl_gbp,
                    new_sl=trade["sl_price"],
                )
            except Exception as exc:
                log.warning("[%s] Trail activation alert failed: %s", instrument, exc)

        if trail_active:
            if direction == 1:   # LONG
                new_sl = entry + (mfe_atr - trail_mult) * trail_atr
                if new_sl > trade["sl_price"]:
                    trade["sl_price"] = new_sl
                    self.risk.update_sl(instrument, new_sl)
                    try:
                        self.client.update_stop_loss(trade["trade_id"], instrument, new_sl)
                    except Exception as exc:
                        log.error("[%s] SL update failed: %s", instrument, exc)
            else:                # SHORT
                new_sl = entry - (mfe_atr - trail_mult) * trail_atr
                if new_sl < trade["sl_price"]:
                    trade["sl_price"] = new_sl
                    self.risk.update_sl(instrument, new_sl)
                    try:
                        self.client.update_stop_loss(trade["trade_id"], instrument, new_sl)
                    except Exception as exc:
                        log.error("[%s] SL update failed: %s", instrument, exc)

        trade["trail_activated"] = trail_active

        log.info(
            "[%s] Managing | bars=%d | unrealised=£%.2f | trail=%s | sl=%.5f",
            instrument,
            trade["bars_held"],
            unrealised_pnl_gbp,
            "ACTIVE" if trail_active else "inactive",
            trade["sl_price"],
        )

        # ------------------------------------------------------------------
        # Unrealised P&L milestone alerts — fire once per threshold per trade
        # ------------------------------------------------------------------
        _UNREALISED_MILESTONES = [50.0, 100.0]
        milestones_hit = trade.get("milestones_hit", set())
        for threshold in _UNREALISED_MILESTONES:
            if unrealised_pnl_gbp >= threshold and threshold not in milestones_hit:
                milestones_hit.add(threshold)
                trade["milestones_hit"] = milestones_hit
                log.info(
                    "[%s] Unrealised milestone reached: £%.0f",
                    instrument, threshold,
                )
                try:
                    self.telegram.send_unrealised_milestone(
                        instrument=instrument,
                        direction="LONG" if trade["direction"] == 1 else "SHORT",
                        unrealised_pnl=unrealised_pnl_gbp,
                        milestone=threshold,
                        entry_price=trade["entry_price"],
                        current_price=close,
                    )
                except Exception as exc:
                    log.warning("[%s] Milestone alert failed: %s", instrument, exc)

        # ------------------------------------------------------------------
        # Step 4 — SL hit check (use candle high/low)
        # ------------------------------------------------------------------
        exit_reason: str | None = None

        sl_hit = (
            (direction ==  1 and low  <= trade["sl_price"]) or
            (direction == -1 and high >= trade["sl_price"])
        )
        if sl_hit:
            exit_reason = "TRAIL_SL" if trail_active else "SL"

        # ------------------------------------------------------------------
        # Step 5 — Hard close checks (checked before exit model)
        # ------------------------------------------------------------------
        if exit_reason is None:
            # Index EOD
            if instrument in INDEX_INSTRUMENTS:
                close_hm = _SESSION_CLOSE_UTC[instrument]
                hm       = (now.hour, now.minute)
                if hm >= close_hm:
                    exit_reason = "EOD"

        if exit_reason is None:
            # Forex Friday close
            if instrument in FOREX_PAIRS:
                if now.weekday() == 4 and now.hour >= 21:
                    exit_reason = "FRIDAY_CLOSE"

        if exit_reason is None:
            # Max bars
            if trade["bars_held"] >= MAX_BARS_HARD:
                exit_reason = "MAX_BARS"

        # ------------------------------------------------------------------
        # Step 6 — Exit model (only when trail is active, no hard close)
        # ------------------------------------------------------------------
        if exit_reason is None and trail_active and not skip_exit_model:
            try:
                should_exit, p_exit = self.evaluate_exit(
                    instrument, candle, trade, now
                )
                log.debug(
                    "[%s] Exit model | p_exit=%.4f | trail_active=%s",
                    instrument, p_exit, trail_active,
                )
                if should_exit:
                    exit_reason = "EXIT_MODEL"
            except Exception as exc:
                log.error("[%s] Exit model error: %s", instrument, exc)

        # ------------------------------------------------------------------
        # Step 7 — Execute close if triggered
        # ------------------------------------------------------------------
        if exit_reason is not None:
            try:
                result      = self.client.close_trade(trade["trade_id"], instrument)
                close_price = result["close_price"]
            except Exception as exc:
                exc_str = str(exc)
                if "404" in exc_str or "TRADE_DOESNT_EXIST" in exc_str or "closeoutBid" in exc_str:
                    log.warning(
                        "[%s] Trade %s not found on OANDA — likely manually closed. "
                        "Fetching close details.",
                        instrument, trade["trade_id"],
                    )
                    if self.risk.get_open_trade(instrument) is None:
                        log.info(
                            "[%s] 404 recovery skipped — streaming thread already closed this trade",
                            instrument,
                        )
                        return None
                    close_data = self.client.get_closed_trade(trade["trade_id"])
                    if close_data is None:
                        close_data = self.client.get_trade_transaction(trade["trade_id"])
                        if close_data:
                            close_price = close_data.get("close_price") or trade["sl_price"]
                            log.info(
                                "[%s] Close recovered via transaction history | close=%.5f",
                                instrument, close_price,
                            )
                        else:
                            close_price = trade.get("sl_price") or trade["entry_price"]
                            log.warning(
                                "[%s] transaction history also failed — using SL price %.5f as close price approximation",
                                instrument, close_price,
                            )
                    else:
                        close_price = close_data.get("close_price") or trade["sl_price"]
                        log.info(
                            "[%s] Manual close recovered | close=%.5f",
                            instrument, close_price,
                        )
                    pnl_gbp = self._compute_pnl_gbp(instrument, trade, close_price, now)
                    log.info("[%s] P&L=£%.2f", instrument, pnl_gbp)
                    # Determine actual exit reason — SL hit vs genuine manual close
                    sl_price = float(trade.get("sl_price", 0.0))
                    _direction = int(trade.get("direction", 1))
                    actual_exit_reason = "MANUAL_CLOSE"
                    if sl_price > 0 and close_price > 0:
                        sl_tolerance = float(trade.get("atr_at_entry", 0.0)) * 0.1
                        if _direction == 1 and close_price <= sl_price + sl_tolerance:
                            actual_exit_reason = "SL"
                            if trade.get("trail_activated", False):
                                actual_exit_reason = "TRAIL_SL"
                        elif _direction == -1 and close_price >= sl_price - sl_tolerance:
                            actual_exit_reason = "SL"
                            if trade.get("trail_activated", False):
                                actual_exit_reason = "TRAIL_SL"
                    self.risk.register_close_trade(instrument, pnl_gbp, now)
                    try:
                        from dashboard.db import write_equity_snapshot
                        write_equity_snapshot('acieral_kairos_v1', self.risk.equity, source='practice')
                    except Exception as e:
                        log.warning(f'Equity snapshot failed: {e}')
                    self._write_trade_to_db(
                        instrument, trade, close_price, pnl_gbp,
                        "CLOSED", now, exit_reason=actual_exit_reason
                    )
                    self._send_close_alert(
                        instrument, trade, close_price, pnl_gbp, actual_exit_reason, now
                    )
                    return actual_exit_reason
                log.error(
                    "[%s] close_trade failed: %s — will retry next bar", instrument, exc
                )
                return None

            pnl_gbp = self._compute_pnl_gbp(instrument, trade, close_price, now)
            self.risk.register_close_trade(instrument, pnl_gbp, now)
            try:
                from dashboard.db import write_equity_snapshot
                write_equity_snapshot('acieral_kairos_v1', self.risk.equity, source='practice')
            except Exception as e:
                log.warning(f'Equity snapshot failed: {e}')
            self._write_trade_to_db(instrument, trade, close_price, pnl_gbp,
                                    "CLOSED", now, exit_reason=exit_reason)
            self._send_close_alert(instrument, trade, close_price, pnl_gbp,
                                   exit_reason, now)

            log.info(
                "[%s] Trade closed | reason=%s | close=%.5f | P&L=£%.2f | bars=%d",
                instrument, exit_reason, close_price, pnl_gbp, trade["bars_held"],
            )
            return exit_reason

        return None

    # ------------------------------------------------------------------
    # Real-time tick management (streaming thread)
    # ------------------------------------------------------------------

    def manage_on_tick(
        self,
        instrument: str,
        bid: float,
        ask: float,
        now: datetime,
    ) -> str | None:
        """
        Called on every price tick from the streaming thread.
        Only handles SL hit detection and trail SL updates.
        Exit model and EOD close remain on the H1 candle close loop.

        Returns exit_reason if trade was closed, else None.
        """
        trade = self.risk.get_open_trade(instrument)
        if trade is None:
            return None

        direction = int(trade["direction"])
        sl_price  = float(trade["sl_price"])
        entry     = float(trade["entry_price"])
        atr       = float(trade.get("atr_at_entry", 0.0))

        # Use bid for LONG (we sell at bid), ask for SHORT (we buy at ask)
        current_price = bid if direction == 1 else ask

        # --- SL hit check ---
        sl_hit = (
            (direction ==  1 and bid  <= sl_price) or
            (direction == -1 and ask >= sl_price)
        )
        if sl_hit:
            trail_active = trade.get("trail_activated", False)
            exit_reason  = "TRAIL_SL" if trail_active else "SL"
            log.info(
                "[%s] SL hit on tick | price=%.5f | sl=%.5f | reason=%s",
                instrument, current_price, sl_price, exit_reason,
            )
            try:
                result      = self.client.close_trade(trade["trade_id"], instrument)
                close_price = result["close_price"]
            except Exception as exc:
                exc_str = str(exc)
                if "404" in exc_str or "TRADE_DOESNT_EXIST" in exc_str:
                    # OANDA already closed it
                    if self.risk.get_open_trade(instrument) is None:
                        log.info(
                            "[%s] 404 recovery skipped — streaming thread already closed this trade",
                            instrument,
                        )
                        return None
                    close_data = self.client.get_closed_trade(trade["trade_id"])
                    if close_data is None:
                        close_data = self.client.get_trade_transaction(trade["trade_id"])
                        if close_data:
                            close_price = close_data.get("close_price") or trade["sl_price"]
                        else:
                            close_price = trade.get("sl_price") or entry
                            log.warning(
                                "[%s] transaction history also failed — using SL price %.5f as close price approximation",
                                instrument, close_price,
                            )
                    else:
                        close_price = close_data.get("close_price") or trade["sl_price"]
                else:
                    log.error("[%s] close_trade on tick failed: %s", instrument, exc)
                    return None

            pnl_gbp = self._compute_pnl_gbp(instrument, trade, close_price, now)
            self.risk.register_close_trade(instrument, pnl_gbp, now)
            try:
                from dashboard.db import write_equity_snapshot
                write_equity_snapshot("acieral_kairos_v1", self.risk.equity)
            except Exception as exc:
                log.warning("[%s] Equity snapshot after tick close failed: %s", instrument, exc)
            self._write_trade_to_db(instrument, trade, close_price, pnl_gbp,
                                    "CLOSED", now, exit_reason=exit_reason)
            self._send_close_alert(instrument, trade, close_price, pnl_gbp,
                                   exit_reason, now)
            log.info(
                "[%s] Closed on tick | reason=%s | price=%.5f | P&L=£%.2f",
                instrument, exit_reason, close_price, pnl_gbp,
            )
            return exit_reason

        # --- Trail SL update ---
        if atr > 0:
            trail_mult     = float(self.params["INSTRUMENT_PARAMS"][instrument]["trail_atr_mult"])
            unrealised_atr = (current_price - entry) * direction / atr
            new_mfe_atr    = max(trade.get("mfe_atr", 0.0), unrealised_atr)
            trail_active   = new_mfe_atr >= trail_mult

            # Track what needs to happen outside the lock
            do_oanda_sl_update = False
            new_sl             = None
            send_trail_alert   = False

            with self.risk._lock:
                # Mutate mfe_atr
                trade["mfe_atr"] = new_mfe_atr

                if trail_active:
                    trade["trail_activated"] = True

                    if direction == 1:  # LONG
                        new_sl = entry + (trade["mfe_atr"] - trail_mult) * atr
                        if new_sl > trade["sl_price"]:
                            last_sl_update = trade.get("last_sl_update_time")
                            if last_sl_update and (now - last_sl_update).total_seconds() < 60:
                                # Throttled — update in-memory only, skip OANDA API call
                                trade["sl_price"] = new_sl
                                self.risk.update_sl(instrument, new_sl)
                            else:
                                trade["last_sl_update_time"] = now
                                trade["sl_price"] = new_sl
                                self.risk.update_sl(instrument, new_sl)
                                do_oanda_sl_update = True
                    else:  # SHORT
                        new_sl = entry - (trade["mfe_atr"] - trail_mult) * atr
                        if new_sl < trade["sl_price"]:
                            last_sl_update = trade.get("last_sl_update_time")
                            if last_sl_update and (now - last_sl_update).total_seconds() < 60:
                                # Throttled — update in-memory only, skip OANDA API call
                                trade["sl_price"] = new_sl
                                self.risk.update_sl(instrument, new_sl)
                            else:
                                trade["last_sl_update_time"] = now
                                trade["sl_price"] = new_sl
                                self.risk.update_sl(instrument, new_sl)
                                do_oanda_sl_update = True

                    # Mark trail alert as sent inside lock so no double-fire
                    if not trade.get("trail_alert_sent", False):
                        trade["trail_alert_sent"] = True
                        send_trail_alert = True

            # OANDA API call — outside lock to avoid blocking stream parser
            if do_oanda_sl_update and new_sl is not None:
                try:
                    self.client.update_stop_loss(trade["trade_id"], instrument, new_sl)
                    log.debug(
                        "[%s] Trail SL updated on tick | new_sl=%.5f",
                        instrument, new_sl,
                    )
                except Exception as exc:
                    log.error(
                        "[%s] Trail SL update on tick failed: %s",
                        instrument, exc,
                    )

            # Telegram send — outside lock
            if send_trail_alert:
                unrealised_pnl = self._compute_pnl_gbp(instrument, trade, current_price, now)
                try:
                    self.telegram.send_trail_activated(
                        instrument=instrument,
                        direction="LONG" if direction == 1 else "SHORT",
                        entry_price=entry,
                        current_price=current_price,
                        unrealised_pnl=unrealised_pnl,
                        new_sl=trade["sl_price"],
                    )
                except Exception as exc:
                    log.warning(
                        "[%s] Trail activation alert failed: %s", instrument, exc
                    )

        # --- Real-time index session close check ---
        # Checked on every tick so the position closes within seconds of session
        # end rather than waiting up to 59 minutes for the next HH:01 candle.
        if instrument in INDEX_INSTRUMENTS:
            session_close_str = _HC["INDEX_SESSION_CLOSE_UTC"].get(instrument)
            if session_close_str:
                close_h, close_m = map(int, session_close_str.split(":"))
                if (now.hour, now.minute) >= (close_h, close_m):
                    if self.risk.get_open_trade(instrument):
                        log.info(
                            "[%s] Session close detected on tick — force closing (EOD)",
                            instrument,
                        )
                        return self.force_close(instrument, "EOD", now)

        # --- Real-time pre-news check ---
        # Uses should_close_pre_news (0–15 min window) — same window as the
        # candle-close path.  Applies P&L-aware logic: close on loss,
        # move SL to BE on profit without trail, do nothing if trail active.
        try:
            from risk.news_filter import should_close_pre_news
            close_news, _reason = should_close_pre_news(instrument, now)
            if close_news and self.risk.get_open_trade(instrument):
                self.handle_pre_news(instrument, current_price, now)
                return
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # P&L-aware pre-news management
    # ------------------------------------------------------------------

    def handle_pre_news(
        self,
        instrument: str,
        current_price: float,
        now: datetime,
    ) -> str | None:
        """
        Called when a High impact news event is 0–15 min away.

        Decision tree (based on floating P&L and trail state):
          Floating loss           → close immediately (NEWS_CLOSE)
          Profit + trail active   → do nothing; trail SL already above entry
          Profit + trail inactive → move SL to breakeven; trade stays open

        Returns the exit_reason string if the trade was closed, else None.
        """
        trade = self.risk.get_open_trade(instrument)
        if trade is None:
            return None

        unrealised_pnl = self._compute_pnl_gbp(instrument, trade, current_price, now)
        direction      = int(trade["direction"])   # 1=LONG, -1=SHORT
        entry_price    = float(trade["entry_price"])
        current_sl     = float(trade["sl_price"])

        if unrealised_pnl < 0:
            # Floating loss — close before news hits
            log.warning(
                "[%s] Pre-news: floating loss £%.2f — closing immediately",
                instrument, unrealised_pnl,
            )
            return self.force_close(instrument, "NEWS_CLOSE", now)

        if trade.get("trail_activated", False):
            # Trail already ratcheted SL above entry — no forced action needed
            log.info(
                "[%s] Pre-news: trail active — no action "
                "(sl=%.5f unrealised=£%.2f)",
                instrument, current_sl, unrealised_pnl,
            )
            return None

        # Floating profit, trail not yet active → move SL to breakeven
        # Only move if breakeven is genuinely an improvement over current SL
        if direction == 1 and entry_price <= current_sl:
            log.info(
                "[%s] Pre-news: SL already at/beyond breakeven — no action",
                instrument,
            )
            return None
        if direction == -1 and entry_price >= current_sl:
            log.info(
                "[%s] Pre-news: SL already at/beyond breakeven — no action",
                instrument,
            )
            return None

        new_sl = entry_price
        try:
            self.client.update_stop_loss(trade["trade_id"], instrument, new_sl)
            trade["sl_price"] = new_sl
            self.risk.update_sl(instrument, new_sl)
            log.info(
                "[%s] Pre-news: SL moved to breakeven %.5f (was %.5f) "
                "unrealised=£%.2f — trade stays open",
                instrument, new_sl, current_sl, unrealised_pnl,
            )
        except Exception as exc:
            log.warning(
                "[%s] Pre-news: breakeven SL update failed (%s) — falling back to force close",
                instrument, exc,
            )
            return self.force_close(instrument, "NEWS_CLOSE", now)

        return None

    # ------------------------------------------------------------------
    # Force close (news pre-close, manual intervention)
    # ------------------------------------------------------------------

    def force_close(
        self,
        instrument: str,
        exit_reason: str,
        now: datetime,
    ) -> str | None:
        """
        Immediately close an open trade regardless of model signals.
        Used for pre-news closes and other forced exits.

        Returns exit_reason if a trade was closed, else None.
        """
        trade = self.risk.get_open_trade(instrument)
        if trade is None:
            return None

        try:
            result      = self.client.close_trade(trade["trade_id"], instrument)
            close_price = result["close_price"]

            # Validate — close_trade should raise if price missing, but guard here too
            if not close_price:
                raise ValueError(
                    f"close_trade returned zero/None close_price for {instrument} — "
                    f"treating as failed close"
                )
        except Exception as exc:
            exc_str = str(exc)
            if "404" in exc_str or "TRADE_DOESNT_EXIST" in exc_str or "closeoutBid" in exc_str or "treating as failed close" in exc_str:
                log.warning(
                    "[%s] Trade %s not found on OANDA — likely manually closed. "
                    "Fetching close details.",
                    instrument, trade["trade_id"],
                )
                close_data = self.client.get_closed_trade(trade["trade_id"])
                if close_data is None:
                    close_data = self.client.get_trade_transaction(trade["trade_id"])
                    if close_data:
                        close_price = close_data.get("close_price") or trade["sl_price"]
                        log.info(
                            "[%s] Close recovered via transaction history | close=%.5f",
                            instrument, close_price,
                        )
                    else:
                        close_price = trade.get("sl_price") or trade["entry_price"]
                        log.warning(
                            "[%s] transaction history also failed — using SL price %.5f as close price approximation",
                            instrument, close_price,
                        )
                else:
                    close_price = close_data.get("close_price") or trade["sl_price"]
                    log.info(
                        "[%s] Manual close recovered | close=%.5f",
                        instrument, close_price,
                    )
                pnl_gbp = self._compute_pnl_gbp(instrument, trade, close_price, now)
                log.info("[%s] P&L=£%.2f", instrument, pnl_gbp)
                # Determine actual exit reason — SL hit vs genuine manual close
                sl_price = float(trade.get("sl_price", 0.0))
                _direction = int(trade.get("direction", 1))
                actual_exit_reason = "MANUAL_CLOSE"
                if sl_price > 0 and close_price > 0:
                    sl_tolerance = float(trade.get("atr_at_entry", 0.0)) * 0.1
                    if _direction == 1 and close_price <= sl_price + sl_tolerance:
                        actual_exit_reason = "SL"
                        if trade.get("trail_activated", False):
                            actual_exit_reason = "TRAIL_SL"
                    elif _direction == -1 and close_price >= sl_price - sl_tolerance:
                        actual_exit_reason = "SL"
                        if trade.get("trail_activated", False):
                            actual_exit_reason = "TRAIL_SL"
                self.risk.register_close_trade(instrument, pnl_gbp, now)
                try:
                    from dashboard.db import write_equity_snapshot
                    write_equity_snapshot('acieral_kairos_v1', self.risk.equity, source='practice')
                except Exception as e:
                    log.warning(f'Equity snapshot failed: {e}')
                self._write_trade_to_db(
                    instrument, trade, close_price, pnl_gbp,
                    "CLOSED", now, exit_reason=actual_exit_reason
                )
                self._send_close_alert(
                    instrument, trade, close_price, pnl_gbp, actual_exit_reason, now
                )
                return actual_exit_reason
            log.error("[%s] force_close failed: %s", instrument, exc)
            return None

        pnl_gbp = self._compute_pnl_gbp(instrument, trade, close_price, now)
        self.risk.register_close_trade(instrument, pnl_gbp, now)
        try:
            from dashboard.db import write_equity_snapshot
            write_equity_snapshot('acieral_kairos_v1', self.risk.equity, source='practice')
        except Exception as e:
            log.warning(f'Equity snapshot failed: {e}')
        self._write_trade_to_db(instrument, trade, close_price, pnl_gbp,
                                "CLOSED", now, exit_reason=exit_reason)
        self._send_close_alert(instrument, trade, close_price, pnl_gbp,
                               exit_reason, now)

        log.info(
            "[%s] Trade force-closed | reason=%s | close=%.5f | P&L=£%.2f",
            instrument, exit_reason, close_price, pnl_gbp,
        )
        return exit_reason

    # ------------------------------------------------------------------
    # Entry — attempt to open a new trade
    # ------------------------------------------------------------------

    def attempt_entry(
        self,
        instrument: str,
        candle: pd.Series,
        features: pd.Series,
        now: datetime,
        size_multiplier: float = 1.0,
    ) -> tuple[bool, float]:
        """
        Attempt to open a new trade on instrument.

        Parameters
        ----------
        instrument : OANDA instrument name
        candle     : current H1 candle (needs close, atr_14)
        features   : current H1 feature row (FEATURE_COLS)
        now        : current UTC datetime (tz-aware)

        Returns
        -------
        (opened, confidence)
          opened     : True if a trade was opened, False otherwise.
          confidence : entry model confidence for the best class (0–1).
                       Always returned so the caller can update _best_confidence
                       without a second evaluate_entry call.
        """
        # ------------------------------------------------------------------
        # Step 1 — Entry model evaluation
        # ------------------------------------------------------------------
        log.info("%s: evaluating candle close", instrument)
        predicted_class, confidence = self.evaluate_entry(instrument, features, now)
        if predicted_class == 0:
            return False, confidence

        # ------------------------------------------------------------------
        # Step 2 — Risk gate
        # ------------------------------------------------------------------
        ok, reason = self.risk.can_trade(instrument, confidence, now)
        if not ok:
            log.info(
                "[%s] SKIP: %s (conf=%.3f)", instrument, reason, confidence
            )
            return False, confidence

        # ------------------------------------------------------------------
        # Step 3 — Compute entry, SL
        # ------------------------------------------------------------------
        atr         = float(candle["atr_14"])
        direction   = 1 if predicted_class == 2 else -1   # 2=LONG, 1=SHORT
        entry_price = float(candle["close"])
        sl_distance = SL_ATR_MULT * atr
        sl_price    = entry_price - direction * sl_distance

        # ------------------------------------------------------------------
        # Step 4 — Fetch conversion rates for position sizing
        # ------------------------------------------------------------------
        try:
            gbpusd_bid, _ = self.client.get_current_price("GBP_USD")
            gbpusd_rate   = gbpusd_bid
        except Exception as exc:
            log.error("[%s] Cannot fetch GBP/USD rate: %s", instrument, exc)
            return False, confidence

        gbpeur_rate: float | None = None
        if instrument == "DE30_EUR":
            try:
                eurgbp_bid, _ = self.client.get_current_price("EUR_GBP")
                gbpeur_rate   = 1.0 / eurgbp_bid   # GBP→EUR rate
            except Exception as exc:
                log.error("[%s] Cannot fetch EUR/GBP rate: %s", instrument, exc)
                return False, confidence

        # ------------------------------------------------------------------
        # Step 5 — Position sizing
        # ------------------------------------------------------------------
        try:
            units_float = self.risk.compute_position_size(
                instrument, sl_distance, entry_price, gbpusd_rate, gbpeur_rate
            )
        except ValueError as exc:
            log.error("[%s] Position sizing failed: %s", instrument, exc)
            return False, confidence

        units_final = round(units_float * size_multiplier, 2)
        if size_multiplier != 1.0:
            log.info(
                "[%s] size_multiplier=%.2f (news filter) — units=%.2f",
                instrument, size_multiplier, units_final,
            )
        if units_final <= 0:
            log.warning("[%s] Computed units=0 after sizing — skipping", instrument)
            return False, confidence

        log.info(
            "[%s] Position size | units_raw=%.4f units_final=%.2f",
            instrument, units_float, units_final,
        )

        signed_units = units_final if direction == 1 else -units_final

        # ------------------------------------------------------------------
        # Step 6 — Place market order
        # ------------------------------------------------------------------
        try:
            result = self.client.market_order(instrument, signed_units, sl_price)
        except Exception as exc:
            log.error("[%s] market_order failed: %s", instrument, exc)
            return False, confidence

        # ------------------------------------------------------------------
        # Step 7 — Build trade record
        # ------------------------------------------------------------------
        risk_gbp_used = min(
            self.risk.equity * self.risk._get_risk_pct(self.risk.equity),
            _HC["MAX_RISK_PER_TRADE_GBP"][instrument],
        )
        trade: dict = {
            "trade_id":        result["trade_id"],
            "instrument":      instrument,
            "direction":       direction,
            "entry_price":     result["fill_price"],
            "entry_time":      result["fill_time"],
            "sl_price":        sl_price,
            "units":           units_final,
            "atr_at_entry":    atr,
            "risk_gbp":        risk_gbp_used,
            "confidence":      confidence,
            "mfe_atr":         0.0,
            "mae_atr":         0.0,
            "bars_held":       0,
            "trail_activated":  False,
            "trail_alert_sent": False,
            "milestones_hit":   set(),
        }

        # ------------------------------------------------------------------
        # Step 8 — Register, write DB, alert
        # ------------------------------------------------------------------
        self.risk.register_open_trade(instrument, trade, now)
        self._write_trade_to_db(instrument, trade, None, None, "OPEN", now)
        self._send_open_alert(instrument, trade, now)

        log.info(
            "[%s] Trade opened | dir=%s | entry=%.5f | sl=%.5f | units=%.2f | conf=%.3f",
            instrument,
            "LONG" if direction == 1 else "SHORT",
            trade["entry_price"],
            sl_price,
            units_final,
            confidence,
        )

        return True, confidence

    # ------------------------------------------------------------------
    # OANDA reconciliation
    # ------------------------------------------------------------------

    def reconcile_open_trades(self) -> None:
        """
        Reconcile the risk manager's in-memory open trade state against
        OANDA's live positions.

        Called on bot startup and on daily reset to prevent stale trade_ids
        from blocking new entries after a restart or an unexpected close.

        Actions taken:
          - Removes from risk manager any trades OANDA has already closed.
          - Restores to risk manager any trades open on OANDA that are
            missing locally (e.g. after a bot restart mid-trade).

        Note: restored trades use atr_at_entry=0.0 because that value is not
        available from the OANDA API.  This means the trailing SL and exit
        model will not activate for the restored leg; the hard OANDA SL still
        protects the position.
        """
        all_instruments = set(_HC["ALL_INSTRUMENTS"])

        try:
            oanda_trades = self.client.get_open_trades()
        except Exception as exc:
            log.error("Reconciliation: failed to fetch OANDA open trades: %s", exc)
            return

        # Map instrument → OANDA trade dict (only instruments we manage)
        oanda_by_instrument = {
            t["instrument"]: t
            for t in oanda_trades
            if t["instrument"] in all_instruments
        }

        # 1. Remove stale trades — present in risk manager but closed on OANDA
        stale = [
            inst for inst in list(self.risk._open_trades)
            if inst not in oanda_by_instrument
        ]
        for instrument in stale:
            if instrument in FOREX_PAIRS:
                self.risk._open_forex = max(0, self.risk._open_forex - 1)
            else:
                self.risk._open_indices = max(0, self.risk._open_indices - 1)
            del self.risk._open_trades[instrument]
            log.info(
                "[%s] Reconciliation: removed stale trade (no longer open on OANDA)",
                instrument,
            )

        # 2. Restore missing trades — open on OANDA but absent from risk manager
        now = datetime.now(timezone.utc)
        for instrument, ot in oanda_by_instrument.items():
            if instrument in self.risk._open_trades:
                continue   # already tracked — no action needed

            direction = 1 if ot["units"] > 0 else -1
            trade: dict = {
                "trade_id":        ot["trade_id"],
                "instrument":      instrument,
                "direction":       direction,
                "entry_price":     ot["open_price"],
                "entry_time":      ot["open_time"],
                "sl_price":        ot["sl_price"],
                "units":           abs(ot["units"]),
                "atr_at_entry":    0.0,   # not available from OANDA API
                "confidence":      0.0,
                "mfe_atr":         0.0,
                "mae_atr":         0.0,
                "bars_held":       0,
                "trail_activated":  False,
                "trail_alert_sent": False,
                "milestones_hit":   set(),
            }
            self.risk.register_open_trade(instrument, trade, now)
            try:
                from dashboard.db import write_trade
                write_trade({
                    'bot_id':       'acieral_kairos_v1',
                    'pair':         instrument,
                    'direction':    'LONG' if direction == 1 else 'SHORT',
                    'entry_time':   ot["open_time"],
                    'entry_price':  trade["entry_price"],
                    'sl_price':     trade["sl_price"],
                    'units':        trade["units"],
                    'atr_at_entry': trade["atr_at_entry"],
                    'status':       'open',
                    'trade_type':   'practice',
                })
            except Exception as e:
                log.warning(f'DB reconcile write failed: {e}')
            log.info(
                "[%s] Reconciliation: restored trade from OANDA | "
                "trade_id=%s | dir=%s | entry=%.5f | sl=%.5f",
                instrument, ot["trade_id"],
                "LONG" if direction == 1 else "SHORT",
                ot["open_price"], ot["sl_price"],
            )

        log.info(
            "OANDA reconciliation complete: "
            "%d open on OANDA, "
            "%d in risk manager",
            len(oanda_trades), len(self.risk._open_trades),
        )

    def reconcile_db_trades(self) -> None:
        """
        Close any Supabase 'open' trade records that are no longer open on OANDA.
        Called on startup and daily reset so the dashboard never shows phantom positions.
        Best-effort — failures are logged but never block trading.
        """
        try:
            oanda_trades = self.client.get_open_trades()
            oanda_instruments = [t["instrument"] for t in oanda_trades]
            from dashboard.db import reconcile_db
            count = reconcile_db("acieral_kairos_v1", oanda_instruments)
            if count > 0:
                log.warning("DB reconciliation: %d stale record(s) closed", count)
            else:
                log.info("DB reconciliation: clean")
        except Exception as e:
            log.warning("DB reconciliation failed: %s", e)

    # ------------------------------------------------------------------
    # P&L computation
    # ------------------------------------------------------------------

    def _compute_pnl_gbp(
        self,
        instrument: str,
        trade: dict,
        close_price: float,
        now: datetime,
    ) -> float:
        """
        Compute realised P&L in GBP using the CLAUDE.md formulas.

        Fetches fresh conversion rates at close time.
        """
        direction   = int(trade["direction"])
        entry_price = float(trade["entry_price"])
        units       = float(trade["units"])
        quote_type  = QUOTE_TYPE[instrument]

        pnl_points = (close_price - entry_price) * direction

        # Use cached GBP/USD and GBP/EUR rates (refreshed every candle close).
        # Warn if the cache is stale (>5 minutes); fall back to 1.0 if uninitialised.
        gbpusd_rate: float = self._gbpusd_rate
        gbpeur_rate: float = self._gbpeur_rate

        if self._gbpusd_rate_updated is not None:
            age = (datetime.now(timezone.utc) - self._gbpusd_rate_updated).total_seconds()
            if age > 300:
                log.warning(
                    "[%s] GBP/USD rate cache is %.0f seconds old — P&L may be imprecise",
                    instrument, age,
                )

        if quote_type == "usd_quote":
            # EUR_USD, GBP_USD, AUD_USD
            # pnl_gbp = (pnl_price × units) / gbpusd_rate
            pnl_gbp = (pnl_points * units) / gbpusd_rate

        elif quote_type == "usd_base":
            # USD_JPY, USD_CHF
            # pnl_gbp = (pnl_price / exit_price × units) / gbpusd_rate
            pnl_gbp = (pnl_points / close_price * units) / gbpusd_rate

        elif quote_type == "usd_index":
            # SPX500_USD, NAS100_USD
            # pnl_gbp = (pnl_points × units) / gbpusd_rate
            pnl_gbp = (pnl_points * units) / gbpusd_rate

        elif quote_type == "eur_index":
            # DE30_EUR
            # pnl_gbp = (pnl_points × units) / gbpeur_rate
            pnl_gbp = (pnl_points * units) / gbpeur_rate

        else:
            log.error("[%s] Unknown quote_type '%s' — P&L set to 0", instrument, quote_type)
            pnl_gbp = 0.0

        return round(pnl_gbp, 2)

    # ------------------------------------------------------------------
    # DB persistence (best-effort)
    # ------------------------------------------------------------------

    def _write_trade_to_db(
        self,
        instrument: str,
        trade: dict,
        close_price,
        pnl_gbp,
        status: str,
        now: datetime,
        exit_reason: str | None = None,
    ) -> None:
        """
        Write trade state to Supabase via dashboard.db.write_trade().
        Failure is logged but never blocks a trade operation.
        """
        try:
            from dashboard.db import write_trade  # type: ignore[import]
            entry_time = trade.get("entry_time")
            write_trade({
                "bot_id":       cfg.HARD_CONSTRAINTS.get("BOT_ID", "acieral_kairos_v1"),
                "pair":         instrument,
                "direction":    "LONG" if trade.get("direction", 1) == 1 else "SHORT",
                "entry_time":   entry_time.isoformat() if hasattr(entry_time, "isoformat") else entry_time,
                "exit_time":    now.isoformat() if status.upper() in ("CLOSED", "closed") else None,
                "entry_price":  trade.get("entry_price"),
                "exit_price":   close_price,
                "sl_price":     trade.get("sl_price"),
                "units":        trade.get("units"),
                "atr_at_entry": trade.get("atr_at_entry"),
                "confidence":   trade.get("confidence"),
                "pnl_gbp":      round(pnl_gbp, 2) if pnl_gbp is not None else None,
                "exit_reason":  exit_reason,
                "bars_held":    trade.get("bars_held"),
                "mfe_price":    (
                    trade.get("entry_price") + trade.get("direction", 1) * trade.get("mfe_atr", 0.0) * trade.get("atr_at_entry")
                    if trade.get("atr_at_entry") else None
                ),
                "mae_price":    (
                    trade.get("entry_price") - trade.get("direction", 1) * trade.get("mae_atr", 0.0) * trade.get("atr_at_entry")
                    if trade.get("atr_at_entry") else None
                ),
                "status":       "closed" if status.upper() in ("CLOSED", "closed") else "open",
                "trade_type":   "practice",
            })
        except ImportError:
            pass   # dashboard module not yet available — silently skip
        except Exception as exc:
            log.error("[%s] DB write failed (non-blocking): %s", instrument, exc)

    # ------------------------------------------------------------------
    # Telegram alerts (best-effort)
    # ------------------------------------------------------------------

    def _send_open_alert(
        self, instrument: str, trade: dict, now: datetime
    ) -> None:
        if self.telegram is None:
            return
        try:
            self.telegram.send_trade_opened(
                instrument  = instrument,
                direction   = "LONG" if trade["direction"] == 1 else "SHORT",
                confidence  = float(trade.get("confidence", 0.0)),
                entry_price = float(trade["entry_price"]),
                sl_price    = float(trade["sl_price"]),
                units       = int(trade["units"]),
                risk_gbp    = float(trade.get("risk_gbp", 50.0)),
                now         = now,
            )
        except Exception as exc:
            log.error("[%s] Telegram open alert failed: %s", instrument, exc)

    def _send_close_alert(
        self,
        instrument: str,
        trade: dict,
        close_price: float,
        pnl_gbp: float,
        exit_reason: str,
        now: datetime,
    ) -> None:
        if self.telegram is None:
            return
        try:
            self.telegram.send_trade_closed(
                instrument  = instrument,
                direction   = "LONG" if trade["direction"] == 1 else "SHORT",
                entry_price = float(trade["entry_price"]),
                close_price = close_price,
                pnl_gbp     = pnl_gbp,
                exit_reason = exit_reason,
                bars_held   = int(trade.get("bars_held", 0)),
                mfe_atr     = float(trade.get("mfe_atr", 0.0)),
                mae_atr     = float(trade.get("mae_atr", 0.0)),
                atr         = float(trade["atr_at_entry"]),
                now         = now,
            )
        except Exception as exc:
            log.error("[%s] Telegram close alert failed: %s", instrument, exc)
