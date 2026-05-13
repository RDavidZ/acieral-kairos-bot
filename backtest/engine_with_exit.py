"""
backtest/engine_with_exit.py — Full backtest with trailing SL + exit model

Extends engine.py by replacing the fixed 3:1 TP with:
  1. A trailing stop-loss that ratchets in the direction of profit once
     unrealised gain >= trail_atr_mult × ATR.
  2. An XGBoost exit model that fires when P(exit) >= exit_threshold,
     but only while the trail is active (trade has positive MFE).

Shared helpers (_compute_units, _compute_pnl_gbp, _merge_rate,
_compute_metrics) are imported directly from engine.py.

Exit reason taxonomy
--------------------
  SL          — initial ATR stop hit (trail not yet active)
  TRAIL_SL    — trailing stop hit (trail was active, price pulled back)
  EXIT_MODEL  — exit model probability crossed threshold
  EOD         — end of session (indices)
  FRIDAY_CLOSE— Friday 21:00 UTC (forex)
  MAX_BARS    — hard 24-bar cap
  DATA_END    — forced close at data boundary
"""

import csv
import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import HARD_CONSTRAINTS, DISCOVERED_PARAMS
from ml.labeler import FEATURE_COLS
from ml.exit_trainer import EXIT_FEATURE_COLS, TC_FEATURES
from backtest.engine import (
    _compute_units,
    _compute_pnl_gbp,
    _merge_rate,
    _compute_metrics,
    CACHE_DIR,
    MODELS_DIR,
    FALLBACK_GBPUSD,
    SL_ATR_MULT,
    HOLDOUT_START,
    DAILY_DD_KILL_PCT,
    MIN_CONFIDENCE,
    STARTING_EQUITY,
    MAX_BARS_HELD,
    SESSION_LAST_OPEN_HOUR,
    FOREX_PAIRS,
    INDEX_INSTRUMENTS,
    ALL_INSTRUMENTS,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RESULTS_EXIT_DIR = Path(__file__).parent / "results_exit"
RESULTS_EXIT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core backtest with exit model
# ---------------------------------------------------------------------------

def run_backtest_with_exit(
    instrument: str,
    entry_model_path: str,
    exit_model_path: str,
    trail_atr_mult: float = 1.0,
    exit_threshold: float = 0.50,
    split: str = "holdout",
    _save: bool = True,
    end_date: str | None = None,
    out_dir: Path | None = None,
) -> dict:
    """
    Run the full backtest combining entry model, trailing SL, and exit model.

    Parameters
    ----------
    instrument       : OANDA instrument name, e.g. 'EUR_USD'
    entry_model_path : Path to trained entry model .pkl
    exit_model_path  : Path to trained exit model .pkl
    trail_atr_mult   : Trail SL activates when MFE >= N × ATR (default 1.0)
    exit_threshold   : Exit model probability threshold (default 0.50)
    split            : 'holdout' or 'train'
    _save            : Write output files (set False during sweeps)
    end_date         : Optional ISO date string to cap the backtest window (exclusive)
    out_dir          : Directory for output files (defaults to RESULTS_EXIT_DIR)

    Returns
    -------
    dict of summary metrics, plus pct_exit_model and pct_trail_sl.
    """
    _out_dir = Path(out_dir) if out_dir is not None else RESULTS_EXIT_DIR
    _out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and filter features
    # ------------------------------------------------------------------
    feat_path = CACHE_DIR / f"{instrument}_H1_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Features file not found: {feat_path}")

    df = pd.read_parquet(feat_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    holdout_ts = pd.Timestamp(HOLDOUT_START, tz="UTC")
    if split == "holdout":
        df = df[df["time"] >= holdout_ts].reset_index(drop=True)
    else:
        df = df[df["time"] <  holdout_ts].reset_index(drop=True)

    if end_date is not None:
        end_ts = pd.Timestamp(end_date, tz="UTC")
        df = df[df["time"] < end_ts].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No data for {instrument} with split='{split}'.")

    log.info(
        "[%s] Backtest (with exit) split='%s': %d bars (%s -> %s)",
        instrument, split, len(df),
        df["time"].iloc[0].date(), df["time"].iloc[-1].date(),
    )

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    with open(entry_model_path, "rb") as fh:
        entry_model = pickle.load(fh)
    with open(exit_model_path, "rb") as fh:
        exit_model = pickle.load(fh)

    # ------------------------------------------------------------------
    # 3. Merge rate data
    # ------------------------------------------------------------------
    df = _merge_rate(df, "GBP_USD", "_gbpusd", fallback=FALLBACK_GBPUSD)

    is_de30 = (instrument == "DE30_EUR")
    if is_de30:
        df = _merge_rate(df, "EUR_USD", "_eurusd", fallback=0.92)
        df["_gbpeur"] = (
            df["_gbpusd"] / df["_eurusd"].replace(0.0, np.nan)
        ).ffill().fillna(1.17)

    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Batch predict entry signals
    # ------------------------------------------------------------------
    X_entry      = np.nan_to_num(df[FEATURE_COLS].to_numpy(dtype=np.float32), nan=0.0)
    probas_entry = entry_model.predict_proba(X_entry)
    pred_classes = np.argmax(probas_entry, axis=1).astype(np.int8)
    confidences  = np.max(probas_entry, axis=1).astype(np.float32)

    # Precompute market feature rows for exit model (used bar-by-bar)
    X_market = np.nan_to_num(df[FEATURE_COLS].to_numpy(dtype=np.float32), nan=0.0)

    # ------------------------------------------------------------------
    # 5. Extract numpy arrays for fast loop access
    # ------------------------------------------------------------------
    n        = len(df)
    closes   = df["close"].to_numpy(dtype=np.float64)
    highs    = df["high"].to_numpy(dtype=np.float64)
    lows     = df["low"].to_numpy(dtype=np.float64)
    atrs     = df["atr_14"].to_numpy(dtype=np.float64)
    gbpusd   = df["_gbpusd"].to_numpy(dtype=np.float64)
    gbpeur   = df["_gbpeur"].to_numpy(dtype=np.float64) if is_de30 else None

    dates_arr = df["time"].dt.date.to_numpy()
    hours_arr = df["time"].dt.hour.to_numpy(dtype=np.int32)
    dow_arr   = df["time"].dt.dayofweek.to_numpy(dtype=np.int32)
    ts_vals   = df["time"].to_numpy()

    # Per-bar confidence threshold from CONFIDENCE_CURVE (per-hour, per-instrument).
    # Falls back to MIN_CONFIDENCE if instrument not in DISCOVERED_PARAMS.
    _conf_curve = DISCOVERED_PARAMS.get("CONFIDENCE_CURVE", {}).get(instrument, {})
    entry_thresholds = np.array(
        [float(_conf_curve.get(int(h), MIN_CONFIDENCE)) for h in hours_arr],
        dtype=np.float32,
    )

    is_index       = instrument in INDEX_INSTRUMENTS
    last_sess_hour = SESSION_LAST_OPEN_HOUR.get(instrument)

    # ------------------------------------------------------------------
    # 6. Simulation state
    # ------------------------------------------------------------------
    equity           = STARTING_EQUITY
    day_start_equity = STARTING_EQUITY
    current_date     = None
    traded_today     = False
    kill_switch      = False
    open_trade: dict | None = None

    trades:         list[dict] = []
    equity_by_date: dict       = {}

    # ------------------------------------------------------------------
    # 7. Main simulation loop
    # ------------------------------------------------------------------
    for i in range(n):
        bar_date  = dates_arr[i]
        bar_hour  = int(hours_arr[i])
        bar_dow   = int(dow_arr[i])
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]

        # ---- Day boundary -------------------------------------------
        if bar_date != current_date:
            if current_date is not None:
                equity_by_date[current_date] = equity
            current_date     = bar_date
            day_start_equity = equity
            traded_today     = False
            kill_switch      = False

        # ---- Exit check for open trade ------------------------------
        if open_trade is not None:
            bars_held    = i - open_trade["entry_bar"]
            direction    = open_trade["direction"]   # 1=SHORT, 2=LONG
            ep           = open_trade["entry_price"]
            atr_e        = open_trade["atr_at_entry"]
            sl_price     = open_trade["sl_price"]
            tc_dir       = 1 if direction == 2 else -1

            # ---- Unrealised P&L in ATR units -------------------------
            if direction == 2:  # LONG
                unrealised_atr = (bar_close - ep) / atr_e
            else:               # SHORT
                unrealised_atr = (ep - bar_close) / atr_e

            # ---- Update MFE / MAE in ATR units -----------------------
            open_trade["mfe_atr"] = max(open_trade["mfe_atr"], unrealised_atr)
            open_trade["mae_atr"] = max(open_trade["mae_atr"],
                                        max(0.0, -unrealised_atr))
            mfe_atr = open_trade["mfe_atr"]
            mae_atr = open_trade["mae_atr"]

            # Also track MFE/MAE as prices (for CSV output, same as engine.py)
            if direction == 2:
                open_trade["mfe_price"] = max(open_trade["mfe_price"], bar_high)
                open_trade["mae_price"] = min(open_trade["mae_price"], bar_low)
            else:
                open_trade["mfe_price"] = min(open_trade["mfe_price"], bar_low)
                open_trade["mae_price"] = max(open_trade["mae_price"], bar_high)

            # ---- Trailing SL management ------------------------------
            trail_active = mfe_atr >= trail_atr_mult
            if trail_active:
                open_trade["trail_activated"] = True
                locked_atr    = mfe_atr - trail_atr_mult
                if direction == 2:  # LONG: SL ratchets up
                    trail_sl = ep + locked_atr * atr_e
                    sl_price  = max(sl_price, trail_sl)
                else:               # SHORT: SL ratchets down
                    trail_sl = ep - locked_atr * atr_e
                    sl_price  = min(sl_price, trail_sl)
                open_trade["sl_price"] = sl_price

            exit_price: float | None = None
            exit_reason: str  | None = None

            # ---- SL check (initial or trailing) — checked FIRST ------
            if direction == 2 and bar_low <= sl_price:
                exit_price  = sl_price
                exit_reason = "TRAIL_SL" if open_trade["trail_activated"] else "SL"
            elif direction == 1 and bar_high >= sl_price:
                exit_price  = sl_price
                exit_reason = "TRAIL_SL" if open_trade["trail_activated"] else "SL"

            # ---- EOD / session close ---------------------------------
            if exit_reason is None:
                if is_index and last_sess_hour is not None and bar_hour >= last_sess_hour:
                    exit_price, exit_reason = bar_close, "EOD"
                elif not is_index and bar_dow == 4 and bar_hour >= 20:
                    exit_price, exit_reason = bar_close, "FRIDAY_CLOSE"

            # ---- Hard max-bars cap -----------------------------------
            if exit_reason is None and bars_held >= MAX_BARS_HELD:
                exit_price, exit_reason = bar_close, "MAX_BARS"

            # ---- Exit model (only when trail is active, no hard close) ---
            p_exit = 0.0
            if exit_reason is None and trail_active:
                # Build exit feature vector: market features + tc features
                tc_pct_gb = ((mfe_atr - unrealised_atr) / mfe_atr
                             if mfe_atr > 0 else 0.0)
                tc_pct_gb = float(np.clip(tc_pct_gb, 0.0, 1.0))

                if direction == 2:
                    tc_dist_sl = (bar_close - sl_price) / atr_e
                else:
                    tc_dist_sl = (sl_price - bar_close) / atr_e

                tc_vec = np.array([
                    float(bars_held),
                    float(unrealised_atr),
                    float(mfe_atr),
                    float(mae_atr),
                    tc_pct_gb,
                    float(tc_dir),
                    float(tc_dist_sl),
                ], dtype=np.float32)

                feat_vec = np.concatenate(
                    [X_market[i], tc_vec]
                ).reshape(1, -1)

                p_exit = float(exit_model.predict_proba(feat_vec)[0][1])

                if p_exit >= exit_threshold:
                    exit_price  = bar_close
                    exit_reason = "EXIT_MODEL"
                    open_trade["exit_model_fired"] = True

            open_trade["p_exit_at_close"] = p_exit

            # ---- Close trade ----------------------------------------
            if exit_price is not None:
                gbu = float(gbpusd[i])
                gbe = float(gbpeur[i]) if is_de30 else None
                if not math.isfinite(gbu) or gbu <= 0:
                    gbu = FALLBACK_GBPUSD

                pnl = _compute_pnl_gbp(
                    instrument, direction, ep, exit_price,
                    open_trade["units"], gbu, gbpeur_exit=gbe,
                )
                equity += pnl

                trades.append({
                    "entry_time":        open_trade["entry_time"],
                    "exit_time":         str(pd.Timestamp(ts_vals[i])),
                    "direction":         "LONG" if direction == 2 else "SHORT",
                    "entry_price":       round(ep, 5),
                    "exit_price":        round(exit_price, 5),
                    "sl_price":          round(open_trade["sl_price"], 5),
                    "units":             round(open_trade["units"], 2),
                    "pnl_gbp":           round(pnl, 4),
                    "exit_reason":       exit_reason,
                    "bars_held":         bars_held,
                    "mfe_price":         round(open_trade["mfe_price"], 5),
                    "mae_price":         round(open_trade["mae_price"], 5),
                    "mfe_atr":           round(mfe_atr, 4),
                    "mae_atr":           round(mae_atr, 4),
                    "confidence":        round(float(open_trade["confidence"]), 4),
                    "atr_at_entry":      round(atr_e, 6),
                    "trail_activated":   open_trade["trail_activated"],
                    "exit_model_fired":  open_trade["exit_model_fired"],
                    "p_exit_at_close":   round(open_trade["p_exit_at_close"], 4),
                })
                open_trade = None

                # Kill switch
                if (day_start_equity - equity) / max(day_start_equity, 1.0) \
                        >= DAILY_DD_KILL_PCT:
                    kill_switch = True

        # ---- Entry check -------------------------------------------
        if open_trade is None and not traded_today and not kill_switch:
            pc   = int(pred_classes[i])
            conf = float(confidences[i])

            if pc in {1, 2} and conf >= float(entry_thresholds[i]):
                atr = float(atrs[i])
                if not math.isfinite(atr) or atr <= 0:
                    continue

                ep_price = closes[i]
                sl_dist  = SL_ATR_MULT * atr

                if pc == 2:  # LONG
                    sl_price = ep_price - sl_dist
                else:        # SHORT
                    sl_price = ep_price + sl_dist

                gbu = float(gbpusd[i])
                gbe = float(gbpeur[i]) if is_de30 else None
                if not math.isfinite(gbu) or gbu <= 0:
                    gbu = FALLBACK_GBPUSD

                units = _compute_units(
                    instrument, ep_price, sl_dist, equity,
                    gbu, gbpeur_entry=gbe,
                )

                open_trade = {
                    "entry_bar":         i,
                    "entry_time":        str(pd.Timestamp(ts_vals[i])),
                    "direction":         pc,
                    "entry_price":       ep_price,
                    "sl_price":          sl_price,
                    "sl_distance":       sl_dist,
                    "units":             units,
                    "confidence":        conf,
                    "atr_at_entry":      atr,
                    "mfe_price":         ep_price,
                    "mae_price":         ep_price,
                    "mfe_atr":           0.0,
                    "mae_atr":           0.0,
                    "trail_activated":   False,
                    "exit_model_fired":  False,
                    "p_exit_at_close":   0.0,
                }
                traded_today = True

    # Record equity for the final day
    if current_date is not None:
        equity_by_date[current_date] = equity

    # Force-close any trade still open at data end
    if open_trade is not None:
        ep    = open_trade["entry_price"]
        atr_e = open_trade["atr_at_entry"]
        exit_p = closes[-1]
        direction = open_trade["direction"]

        if direction == 2:
            mfe_atr = (open_trade["mfe_price"] - ep) / atr_e
            mae_atr = max(0.0, (ep - open_trade["mae_price"]) / atr_e)
        else:
            mfe_atr = (ep - open_trade["mfe_price"]) / atr_e
            mae_atr = max(0.0, (open_trade["mae_price"] - ep) / atr_e)

        gbu = float(gbpusd[-1])
        gbe = float(gbpeur[-1]) if is_de30 else None
        if not math.isfinite(gbu) or gbu <= 0:
            gbu = FALLBACK_GBPUSD

        pnl = _compute_pnl_gbp(
            instrument, direction, ep, exit_p,
            open_trade["units"], gbu, gbpeur_exit=gbe,
        )
        equity += pnl

        trades.append({
            "entry_time":        open_trade["entry_time"],
            "exit_time":         str(pd.Timestamp(ts_vals[-1])),
            "direction":         "LONG" if direction == 2 else "SHORT",
            "entry_price":       round(ep, 5),
            "exit_price":        round(exit_p, 5),
            "sl_price":          round(open_trade["sl_price"], 5),
            "units":             round(open_trade["units"], 2),
            "pnl_gbp":           round(pnl, 4),
            "exit_reason":       "DATA_END",
            "bars_held":         n - 1 - open_trade["entry_bar"],
            "mfe_price":         round(open_trade["mfe_price"], 5),
            "mae_price":         round(open_trade["mae_price"], 5),
            "mfe_atr":           round(mfe_atr, 4),
            "mae_atr":           round(mae_atr, 4),
            "confidence":        round(float(open_trade["confidence"]), 4),
            "atr_at_entry":      round(atr_e, 6),
            "trail_activated":   open_trade["trail_activated"],
            "exit_model_fired":  open_trade["exit_model_fired"],
            "p_exit_at_close":   round(open_trade["p_exit_at_close"], 4),
        })

    # ------------------------------------------------------------------
    # 8. Compute metrics
    # ------------------------------------------------------------------
    metrics = _compute_metrics(trades, equity_by_date)
    metrics["instrument"] = instrument
    metrics["split"]      = split
    metrics["n_bars"]     = n

    n_total       = len(trades)
    n_exit_model  = sum(1 for t in trades if t["exit_reason"] == "EXIT_MODEL")
    n_trail_sl    = sum(1 for t in trades if t["exit_reason"] == "TRAIL_SL")

    metrics["pct_exit_model"] = round(n_exit_model / n_total, 4) if n_total else 0.0
    metrics["pct_trail_sl"]   = round(n_trail_sl   / n_total, 4) if n_total else 0.0

    log.info(
        "[%s] trail=%.1f thr=%.2f | %d trades | win=%.1f%% | PF=%.2f | "
        "Sharpe=%.2f | DD=%.1f%% | P&L=£%.2f | EXIT_MODEL=%.0f%% TRAIL_SL=%.0f%%",
        instrument, trail_atr_mult, exit_threshold,
        metrics["total_trades"],
        metrics["win_rate"] * 100,
        metrics["profit_factor"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown"] * 100,
        metrics["total_pnl_gbp"],
        metrics["pct_exit_model"] * 100,
        metrics["pct_trail_sl"]   * 100,
    )

    # ------------------------------------------------------------------
    # 9. Save outputs
    # ------------------------------------------------------------------
    if _save:
        _TRADE_COLS = [
            "entry_time", "exit_time", "direction", "entry_price", "exit_price",
            "sl_price", "units", "pnl_gbp", "exit_reason", "bars_held",
            "mfe_price", "mae_price", "mfe_atr", "mae_atr",
            "confidence", "atr_at_entry",
            "trail_activated", "exit_model_fired", "p_exit_at_close",
        ]
        trades_path = _out_dir / f"{instrument}_trades.csv"
        with open(trades_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_TRADE_COLS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(trades)

        equity_path = _out_dir / f"{instrument}_equity.csv"
        with open(equity_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "equity"])
            for d, eq in sorted(equity_by_date.items()):
                writer.writerow([str(d), round(eq, 4)])

        metrics_path = _out_dir / f"{instrument}_metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)

    return metrics


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def sweep_exit_params(
    instrument: str,
    entry_model_path: str,
    exit_model_path: str,
    split: str = "holdout",
) -> pd.DataFrame:
    """
    Sweep trail_atr_mult × exit_threshold combinations (42 total).

    trail_atr_mult : [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    exit_threshold : [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    Saves sweep CSV to backtest/results_exit/{instrument}_param_sweep.csv
    Returns DataFrame sorted by Sharpe DESC.
    """
    trail_mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    thresholds  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    total       = len(trail_mults) * len(thresholds)

    log.info(
        "[%s] sweep_exit_params: %d combinations (trail × threshold)",
        instrument, total,
    )

    rows = []
    combo_n = 0

    for trail in trail_mults:
        for thr in thresholds:
            combo_n += 1
            m = run_backtest_with_exit(
                instrument, entry_model_path, exit_model_path,
                trail_atr_mult=trail,
                exit_threshold=thr,
                split=split,
                _save=False,       # suppress file writes during sweep
            )
            rows.append({
                "trail_atr_mult":   trail,
                "exit_threshold":   thr,
                "sharpe":           m["sharpe_ratio"],
                "win_rate":         m["win_rate"],
                "profit_factor":    m["profit_factor"],
                "max_dd":           m["max_drawdown"],
                "total_pnl_gbp":    m["total_pnl_gbp"],
                "total_trades":     m["total_trades"],
                "avg_hold_bars":    m["avg_hold_bars"],
                "pct_exit_model":   m["pct_exit_model"],
                "pct_trail_sl":     m["pct_trail_sl"],
            })

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    sweep_path = RESULTS_EXIT_DIR / f"{instrument}_param_sweep.csv"
    df.to_csv(sweep_path, index=False)
    log.info("[%s] Sweep saved -> %s", instrument, sweep_path.name)

    return df


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_with_exit(split: str = "holdout") -> pd.DataFrame:
    """
    For each instrument:
      1. Run sweep_exit_params() to find optimal trail × threshold.
      2. Run final backtest with winner params (saves files).
      3. Save winner params to backtest/results_exit/winner_params.json.

    Returns summary DataFrame.
    """
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "ml/models/metadata.json not found. Run ml/multi_combo_trainer.py first."
        )
    with open(meta_path) as fh:
        entry_meta = json.load(fh)

    winner_params: dict = {}
    summary_rows:  list[dict] = []

    for instrument in ALL_INSTRUMENTS:
        if instrument not in entry_meta:
            log.warning("[%s] No entry metadata — skipping.", instrument)
            continue

        entry_model = MODELS_DIR / f"entry_{instrument}.pkl"
        exit_model  = MODELS_DIR / f"exit_{instrument}.pkl"

        if not entry_model.exists():
            log.warning("[%s] Entry model missing — skipping.", instrument)
            continue
        if not exit_model.exists():
            log.warning("[%s] Exit model missing — skipping.", instrument)
            continue

        log.info("--- %s ---", instrument)
        try:
            sweep = sweep_exit_params(
                instrument, str(entry_model), str(exit_model), split=split,
            )

            winner = sweep.iloc[0]
            trail_w = float(winner["trail_atr_mult"])
            thr_w   = float(winner["exit_threshold"])

            log.info(
                "[%s] Winner: trail=%.1f thr=%.2f Sharpe=%.2f",
                instrument, trail_w, thr_w, float(winner["sharpe"]),
            )

            # Final run with winner params — saves files
            m = run_backtest_with_exit(
                instrument, str(entry_model), str(exit_model),
                trail_atr_mult=trail_w,
                exit_threshold=thr_w,
                split=split,
                _save=True,
            )

            winner_params[instrument] = {
                "trail_atr_mult": trail_w,
                "exit_threshold": thr_w,
            }
            summary_rows.append({
                "instrument":    instrument,
                "trail_atr_mult": trail_w,
                "exit_threshold": thr_w,
                **{k: m[k] for k in [
                    "total_trades", "win_rate", "profit_factor",
                    "sharpe_ratio", "max_drawdown", "total_pnl_gbp",
                    "pct_exit_model", "pct_trail_sl",
                ]},
            })

        except Exception as exc:
            log.error("[%s] run_all_with_exit FAILED: %s", instrument, exc,
                      exc_info=True)

    # Save winner params
    winner_path = RESULTS_EXIT_DIR / "winner_params.json"
    with open(winner_path, "w") as fh:
        json.dump(winner_params, fh, indent=2)
    log.info("Winner params saved -> %s", winner_path.name)

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        header = (
            f"\n{'Instrument':<15} {'Trail':>6} {'Thr':>5} {'Sharpe':>8}"
            f" {'Win%':>6} {'PF':>6} {'MaxDD':>7} {'P&L GBP':>10}"
            f" {'EXIT%':>7} {'TRSL%':>7}"
        )
        print(header)
        print("-" * len(header.strip()))
        for _, row in summary_df.iterrows():
            print(
                f"{row['instrument']:<15} {row['trail_atr_mult']:>6.1f}"
                f" {row['exit_threshold']:>5.2f}"
                f" {row['sharpe_ratio']:>8.2f}"
                f" {row['win_rate']:>5.1%}"
                f" {row['profit_factor']:>6.2f}"
                f" {row['max_drawdown']:>6.1%}"
                f" £{row['total_pnl_gbp']:>9.2f}"
                f" {row['pct_exit_model']:>6.1%}"
                f" {row['pct_trail_sl']:>6.1%}"
            )

    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_with_exit(split="holdout")
