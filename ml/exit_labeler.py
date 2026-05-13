"""
ml/exit_labeler.py — Oracle peak-exit labelling for Acieral Kairos Bot

Generates training labels for the exit model using oracle (look-ahead) labels
derived from the entry-only backtest trade log.

Label encoding
--------------
  0 = STAY  (hold the trade)
  1 = EXIT  (oracle peak profit bar — the best possible exit)

Algorithm (per trade)
---------------------
For each non-SL trade in the backtest results:
  1. Collect all H1 bars strictly after entry_time up to and including exit_time.
  2. Compute unrealised P&L (ATR-normalised) at each bar.
  3. Find the bar with maximum unrealised P&L (oracle peak).
  4. Label that bar EXIT=1, all others STAY=0.
  5. Skip trades with only 1 bar, or where max unrealised P&L <= 0.

Output columns
--------------
  time          — bar timestamp (UTC)
  trade_id      — sequential integer per instrument
  <FEATURE_COLS>  — 81 market features at that bar
  tc_bars_held           — bars elapsed since entry (1-indexed)
  tc_unrealised_atr      — unrealised P&L in ATR units (+ve = in profit)
  tc_mfe_atr             — running max unrealised P&L up to this bar
  tc_mae_atr             — running max adverse P&L up to this bar
  tc_pct_mfe_given_back  — (mfe - unrealised) / mfe, clipped [0,1]
  tc_direction           — 1 for LONG, -1 for SHORT
  tc_dist_to_sl_atr      — distance from current price to SL, in ATR units
                           (positive = price is above SL for LONG / below for SHORT)
  label         — 0 (STAY) or 1 (EXIT)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import HARD_CONSTRAINTS
from ml.labeler import FEATURE_COLS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CACHE_DIR      = Path(__file__).parent.parent / "data" / "cache"
RESULTS_DIR    = Path(__file__).parent.parent / "backtest" / "results"
EXIT_LABELS_DIR = Path(__file__).parent / "exit_labels"

EXIT_LABELS_DIR.mkdir(exist_ok=True)

ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]


# ---------------------------------------------------------------------------
# Core labeller
# ---------------------------------------------------------------------------

def label_exits(instrument: str) -> pd.DataFrame:
    """
    Generate oracle peak-exit labels for *instrument*.

    Loads the entry-only backtest trade log and marks the oracle peak profit
    bar within each trade as EXIT=1.  All other bars in the trade are STAY=0.

    Parameters
    ----------
    instrument : OANDA instrument name, e.g. 'EUR_USD'

    Returns
    -------
    DataFrame with columns: time, trade_id, <FEATURE_COLS>, tc_*, label
    One row per evaluated bar across all processed trades.
    """
    # ------------------------------------------------------------------
    # Load trade log
    # ------------------------------------------------------------------
    trades_path = RESULTS_DIR / f"{instrument}_trades.csv"
    if not trades_path.exists():
        raise FileNotFoundError(
            f"Trade log not found: {trades_path}\n"
            f"Run backtest.engine.run_backtest('{instrument}', ...) first."
        )

    trades = pd.read_csv(trades_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"]  = pd.to_datetime(trades["exit_time"],  utc=True)

    # Defensive: keep only closed trades
    trades = trades.dropna(subset=["exit_time", "exit_price"])

    n_total = len(trades)

    # ------------------------------------------------------------------
    # Load H1 features — full range (covers both train and holdout)
    # ------------------------------------------------------------------
    feat_path = CACHE_DIR / f"{instrument}_H1_features.parquet"
    feats = pd.read_parquet(feat_path)
    feats["time"] = pd.to_datetime(feats["time"], utc=True)
    feats = feats.sort_values("time").reset_index(drop=True)

    # Index by time for O(log n) slicing
    feats_times = feats["time"].to_numpy()  # numpy array for searchsorted

    # Verify all FEATURE_COLS are present
    missing = [c for c in FEATURE_COLS if c not in feats.columns]
    if missing:
        raise RuntimeError(f"[{instrument}] Missing feature columns in cache: {missing}")

    # ------------------------------------------------------------------
    # Process each trade
    # ------------------------------------------------------------------
    all_rows: list[dict] = []

    skipped_sl       = 0
    skipped_one_bar  = 0
    skipped_no_profit = 0

    for trade_id, trade in trades.iterrows():
        # Skip SL exits — no useful exit signal
        if trade["exit_reason"] == "SL":
            skipped_sl += 1
            continue

        direction    = trade["direction"]        # "LONG" or "SHORT"
        entry_price  = float(trade["entry_price"])
        sl_price     = float(trade["sl_price"])
        atr_at_entry = float(trade["atr_at_entry"])
        entry_time   = trade["entry_time"]
        exit_time    = trade["exit_time"]

        dir_mult = 1.0 if direction == "LONG" else -1.0
        tc_dir   = 1   if direction == "LONG" else -1

        # Find bars: entry_time < bar_time <= exit_time
        lo = np.searchsorted(feats_times, entry_time, side="right")   # first bar > entry
        hi = np.searchsorted(feats_times, exit_time,  side="right")   # one past exit bar

        if hi <= lo:
            skipped_one_bar += 1
            continue

        window = feats.iloc[lo:hi].reset_index(drop=True)
        n_bars = len(window)

        if n_bars < 2:
            skipped_one_bar += 1
            continue

        closes = window["close"].to_numpy(dtype=np.float64)

        # ------------------------------------------------------------------
        # Compute unrealised P&L at each bar (ATR units)
        # ------------------------------------------------------------------
        if atr_at_entry <= 0.0:
            skipped_no_profit += 1
            continue

        unrealised = dir_mult * (closes - entry_price) / atr_at_entry

        if unrealised.max() <= 0.0:
            skipped_no_profit += 1
            continue

        # Oracle peak: first bar with maximum unrealised P&L
        peak_idx = int(np.argmax(unrealised))

        # ------------------------------------------------------------------
        # Running MFE and MAE
        # ------------------------------------------------------------------
        mfe_running = np.maximum.accumulate(unrealised)
        adverse     = np.maximum(0.0, -unrealised)
        mae_running = np.maximum.accumulate(adverse)

        # tc_pct_mfe_given_back: how much of peak has been given back
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_given_back = np.where(
                mfe_running > 0,
                np.clip((mfe_running - unrealised) / mfe_running, 0.0, 1.0),
                0.0,
            )

        # tc_dist_to_sl_atr: positive = price still above SL (LONG) / below SL (SHORT)
        if direction == "LONG":
            dist_to_sl = (closes - sl_price) / atr_at_entry
        else:
            dist_to_sl = (sl_price - closes) / atr_at_entry

        # ------------------------------------------------------------------
        # Build rows
        # ------------------------------------------------------------------
        for j in range(n_bars):
            row = {
                "time":     window.at[j, "time"],
                "trade_id": trade_id,
            }

            # Market features
            for col in FEATURE_COLS:
                row[col] = window.at[j, col]

            # Trade-context features
            row["tc_bars_held"]          = j + 1
            row["tc_unrealised_atr"]     = float(unrealised[j])
            row["tc_mfe_atr"]            = float(mfe_running[j])
            row["tc_mae_atr"]            = float(mae_running[j])
            row["tc_pct_mfe_given_back"] = float(pct_given_back[j])
            row["tc_direction"]          = tc_dir
            row["tc_dist_to_sl_atr"]     = float(dist_to_sl[j])

            row["label"] = 1 if j == peak_idx else 0

            all_rows.append(row)

    if not all_rows:
        raise RuntimeError(
            f"[{instrument}] No exit label rows generated. "
            f"Check that backtest results exist and contain non-SL trades."
        )

    result = pd.DataFrame(all_rows)

    n_trades_used   = result["trade_id"].nunique()
    n_exit_bars     = int((result["label"] == 1).sum())
    n_stay_bars     = int((result["label"] == 0).sum())

    log.info(
        "[%s] label_exits: %d trades processed | %d SL skipped | "
        "%d single-bar skipped | %d no-profit skipped",
        instrument, n_trades_used, skipped_sl, skipped_one_bar, skipped_no_profit,
    )
    log.info(
        "[%s] %d total bars | EXIT=%d (%.2f%%) | STAY=%d",
        instrument, len(result), n_exit_bars,
        n_exit_bars / len(result) * 100, n_stay_bars,
    )

    exit_rows = result[result["label"] == 1]
    stay_rows = result[result["label"] == 0]
    log.info(
        "[%s] tc_pct_mfe_given_back  EXIT=%.3f  STAY=%.3f",
        instrument,
        exit_rows["tc_pct_mfe_given_back"].mean(),
        stay_rows["tc_pct_mfe_given_back"].mean(),
    )

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def label_all_exits() -> None:
    """
    Run label_exits() for all 8 instruments and save parquet files.

    Output: ml/exit_labels/{instrument}_exit_labels.parquet
    Prints a per-instrument summary table on completion.
    """
    log.info("=== label_all_exits() starting ===")

    summary_rows = []

    for instrument in ALL_INSTRUMENTS:
        log.info("--- %s ---", instrument)
        try:
            df = label_exits(instrument)

            out_path = EXIT_LABELS_DIR / f"{instrument}_exit_labels.parquet"
            df.to_parquet(out_path, index=False)
            log.info("[%s] Saved -> %s", instrument, out_path.name)

            exit_bars = df[df["label"] == 1]
            stay_bars = df[df["label"] == 0]

            summary_rows.append({
                "instrument":           instrument,
                "trades_processed":     df["trade_id"].nunique(),
                "total_bars":           len(df),
                "exit_bars":            len(exit_bars),
                "exit_pct":             len(exit_bars) / len(df) * 100,
                "mfe_given_back_exit":  exit_bars["tc_pct_mfe_given_back"].mean(),
                "mfe_given_back_stay":  stay_bars["tc_pct_mfe_given_back"].mean(),
            })

        except Exception as exc:
            log.error("[%s] label_exits FAILED: %s", instrument, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    if summary_rows:
        header = (
            f"\n{'Instrument':<15} {'Trades':>7} {'Bars':>7} {'EXIT%':>7}"
            f" {'MFE-gb EXIT':>12} {'MFE-gb STAY':>12}"
        )
        print(header)
        print("-" * len(header.strip()))
        for r in summary_rows:
            print(
                f"{r['instrument']:<15} {r['trades_processed']:>7} {r['total_bars']:>7}"
                f" {r['exit_pct']:>6.2f}%"
                f" {r['mfe_given_back_exit']:>12.3f}"
                f" {r['mfe_given_back_stay']:>12.3f}"
            )

    log.info("=== label_all_exits() complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    label_all_exits()
