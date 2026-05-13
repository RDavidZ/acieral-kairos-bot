"""
ml/labeler.py — Triple-barrier entry labelling for Acieral Kairos Bot

Label encoding
--------------
  0 = NO_TRADE
  1 = SHORT
  2 = LONG

Algorithm (per trading day)
----------------------------
For each bar k in the day:
  1. Collect the next N bars that fall within the session window.
     (For indices: bars with hour ≤ last valid session open hour.
      For forex: all bars in the day.)
  2. Compute:
       Long  profit  = (max_high[lookahead] - close[k]) / close[k]
       Short profit  = (close[k] - min_low[lookahead])  / close[k]
  3. A direction is valid only when its profit >= T and its
     own profit > the adverse leg (profit > |adverse|).
  4. Assign score = profit if valid, else -inf.

The single bar with the highest max(score_long, score_short) for the day
receives the LONG (2) or SHORT (1) label.  Every other bar is NO_TRADE (0).
If no bar has a valid score >= T the entire day is NO_TRADE.

Grid search
-----------
N_VALUES = [4, 6, 8, 10, 12, 16, 20]
T_VALUES = [0.0005, 0.001, 0.0015, 0.002]   (0.05% … 0.20%)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import HARD_CONSTRAINTS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CACHE_DIR   = Path(__file__).parent.parent / "data" / "cache"
LABELS_DIR  = Path(__file__).parent / "labels_train"
RESULTS_DIR = Path(__file__).parent / "multi_combo_results"

LABELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

ALL_INSTRUMENTS   = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
INDEX_INSTRUMENTS = HARD_CONSTRAINTS["INDEX_INSTRUMENTS"]
INDEX_SESSION_CLOSE = HARD_CONSTRAINTS["INDEX_SESSION_CLOSE_UTC"]

TRAINING_START = HARD_CONSTRAINTS["TRAINING_START"]     # "2017-01-01"
HOLDOUT_START  = HARD_CONSTRAINTS["HOLDOUT_START"]      # "2024-01-01"

# Last *valid* session open hour for index instruments.
# Session close "21:00" → last open hour is 20 (the 20:00 candle opens at 20:00,
# closes at 21:00 — the final session candle).
# Session close "16:30" → last open hour is 15.
SESSION_LAST_OPEN_HOUR = {
    inst: int(cs.split(":")[0]) - 1
    for inst, cs in INDEX_SESSION_CLOSE.items()
}

N_VALUES = [4, 6, 8, 10, 12, 16, 20]
T_VALUES = [0.0005, 0.001, 0.0015, 0.002]


# ---------------------------------------------------------------------------
# Feature column list  (81 columns matching feature_builder output)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Group 1 — FVG (12)
    "fvg_bull_exists", "fvg_bear_exists",
    "fvg_bull_dist_atr", "fvg_bear_dist_atr",
    "fvg_bull_size_atr", "fvg_bear_size_atr",
    "fvg_bull_age", "fvg_bear_age",
    "fvg_bull_fill_pct", "fvg_bear_fill_pct",
    "fvg_stacked_bull", "fvg_stacked_bear",
    # Group 2 — Swing structure (9)
    "swing_high_dist_atr", "swing_low_dist_atr",
    "swing_swept_high", "swing_swept_low",
    "sweep_high_mag_atr", "sweep_low_mag_atr",
    "bars_since_swing_high", "bars_since_swing_low",
    "structure_bias",
    # Group 3 — Session / time (10)
    "minutes_since_midnight", "hour_utc", "day_of_week",
    "session_asian", "session_london", "session_ny", "session_overlap",
    "is_london_first_3h", "is_ny_first_3h", "hours_since_session_open",
    # Group 4 — Higher-timeframe context (17)
    "d1_ema20_slope", "d1_close_vs_ema20",
    "d1_high_dist_atr", "d1_low_dist_atr",
    "d1_fvg_bull_exists", "d1_fvg_bear_exists", "d1_fvg_dist_atr",
    "d1_swing_bias",
    "h4_ema20_slope", "h4_close_vs_ema20",
    "h4_fvg_bull_exists", "h4_fvg_bear_exists", "h4_fvg_dist_atr",
    "h4_swing_bias",
    "htf_alignment",
    "w1_high_dist_atr", "w1_low_dist_atr",
    # Group 5 — Volatility / regime (8)
    "atr_ratio", "atr_ratio_delta",
    "candle_body_ratio", "wick_upper_ratio", "wick_lower_ratio",
    "range_vs_atr_5", "adx_14", "adx_delta",
    # Group 6 — Momentum (8)
    "rsi_14", "rsi_14_delta_3",
    "rsi_div_bull", "rsi_div_bear",
    "macd_hist", "macd_hist_delta",
    "roc_4", "roc_10",
    # Group 7 — Candle context (14)
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_4",
    "body_ratio_lag_1", "body_ratio_lag_2", "body_ratio_lag_3",
    "wick_upper_lag_1", "wick_lower_lag_1",
    "day_high_dist_atr", "day_low_dist_atr", "day_displacement_atr",
    "bull_candle_count", "bear_candle_count",
    # Group 8 — Supplementary (3)
    "vix_close", "vix_delta_1", "spy_volume_ratio",
]

assert len(FEATURE_COLS) == 81, f"Expected 81 feature cols, got {len(FEATURE_COLS)}"


# ---------------------------------------------------------------------------
# Core labelling engine
# ---------------------------------------------------------------------------

def _label_from_data(
    df: pd.DataFrame,
    instrument: str,
    N: int,
    T: float,
) -> np.ndarray:
    """
    Compute triple-barrier labels for every row of *df*.

    Parameters
    ----------
    df         : DataFrame with 'time', 'high', 'low', 'close' columns,
                 sorted by time, index reset.
    instrument : Used to determine session cap for indices.
    N          : Lookahead horizon (bars).
    T          : Profit threshold (fraction, e.g. 0.001 = 0.1%).

    Returns
    -------
    labels : np.ndarray of int8, shape (len(df),)
             0 = NO_TRADE, 1 = SHORT, 2 = LONG
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    hours  = df["time"].dt.hour.to_numpy(dtype=np.int32)
    n_rows = len(df)

    labels = np.zeros(n_rows, dtype=np.int8)

    last_hour = SESSION_LAST_OPEN_HOUR.get(instrument)  # None for forex

    # Day boundary positions
    date_strs  = df["time"].dt.date.astype(str).to_numpy()
    boundaries = np.concatenate(
        [[0], np.where(date_strs[1:] != date_strs[:-1])[0] + 1, [n_rows]]
    )

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        n_day = e - s

        day_hours = hours[s:e]

        # Valid lookahead target positions within this day
        if last_hour is not None:
            session_pos = np.where(day_hours <= last_hour)[0]
        else:
            session_pos = np.arange(n_day)

        if len(session_pos) < 2:
            continue   # nothing to label — no lookahead targets

        day_highs  = highs[s:e]
        day_lows   = lows[s:e]
        day_closes = closes[s:e]

        best_score = -np.inf
        best_k     = -1
        best_dir   = 0   # 1 = SHORT, 2 = LONG

        for k in range(n_day):
            # Find session positions strictly after k, take at most N
            idx_in_sess = np.searchsorted(session_pos, k + 1)
            la_cand = session_pos[idx_in_sess: idx_in_sess + N]

            if la_cand.size == 0:
                continue

            la_max_h = day_highs[la_cand].max()
            la_min_l = day_lows[la_cand].min()
            c = day_closes[k]

            if c <= 0.0:
                continue

            # Long
            bp_long = (la_max_h - c) / c
            wl_long = (la_min_l - c) / c   # negative = adverse down-leg
            valid_long = (bp_long >= T) and (abs(wl_long) < bp_long)
            score_long = bp_long if valid_long else -np.inf

            # Short
            bp_short = (c - la_min_l) / c
            wl_short = (c - la_max_h) / c   # negative = adverse up-leg
            valid_short = (bp_short >= T) and (abs(wl_short) < bp_short)
            score_short = bp_short if valid_short else -np.inf

            score = max(score_long, score_short)
            if score > best_score:
                best_score = score
                best_k = k
                best_dir = 2 if score_long >= score_short else 1

        if best_k >= 0 and best_score >= T:
            labels[s + best_k] = best_dir

    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_instrument(
    instrument: str,
    N: int,
    T: float,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels for *instrument* using lookahead N and
    threshold T.

    Returns a DataFrame containing FEATURE_COLS + 'label', restricted to the
    training period (< HOLDOUT_START).  Saves to
    ml/labels_train/{instrument}_N{N}_T{bps}bps_labels.parquet when save=True.

    Parameters
    ----------
    instrument : OANDA instrument name, e.g. 'EUR_USD'
    N          : Lookahead horizon in bars (e.g. 10)
    T          : Profit threshold as a fraction (e.g. 0.001 = 0.1%)
    save       : Write result to parquet (default True)
    """
    path = CACHE_DIR / f"{instrument}_H1_features.parquet"
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Restrict to training period only
    holdout_ts = pd.Timestamp(HOLDOUT_START, tz="UTC")
    df = df[df["time"] < holdout_ts].reset_index(drop=True)

    log.info(
        "  [%s] labelling %d training rows (N=%d, T=%.4f)",
        instrument, len(df), N, T,
    )

    labels = _label_from_data(df, instrument, N, T)
    df["label"] = labels

    # Verify FEATURE_COLS are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{instrument}] Missing feature columns: {missing}"
        )

    result = df[["time"] + FEATURE_COLS + ["label"]].copy()

    if save:
        bps = int(round(T * 10_000))
        fname = f"{instrument}_N{N}_T{bps}bps_labels.parquet"
        result.to_parquet(LABELS_DIR / fname, index=False)
        log.info("  [%s] Saved → %s", instrument, fname)

    return result


def run_grid_search(instrument: str) -> pd.DataFrame:
    """
    Run label_instrument over all (N, T) combinations and return a summary
    DataFrame with label distribution statistics.

    Columns: N, T_pct, pct_long, pct_short, pct_no_trade,
             n_long, n_short, n_no_trade, total_bars, labeled_days

    Saves to ml/multi_combo_results/{instrument}_grid.csv.
    """
    log.info("=== Grid search for %s ===", instrument)
    rows = []

    for N in N_VALUES:
        for T in T_VALUES:
            df = label_instrument(instrument, N, T, save=True)

            total   = len(df)
            n_long  = int((df["label"] == 2).sum())
            n_short = int((df["label"] == 1).sum())
            n_no    = int((df["label"] == 0).sum())

            # Days with at least one labelled bar
            labeled_days = int(
                df[df["label"] != 0]["time"].dt.date.nunique()
            )
            total_days = int(df["time"].dt.date.nunique())

            rows.append({
                "N":            N,
                "T_pct":        round(T * 100, 4),
                "pct_long":     round(n_long  / total * 100, 2),
                "pct_short":    round(n_short / total * 100, 2),
                "pct_no_trade": round(n_no    / total * 100, 2),
                "n_long":       n_long,
                "n_short":      n_short,
                "n_no_trade":   n_no,
                "total_bars":   total,
                "labeled_days": labeled_days,
                "total_days":   total_days,
                "pct_days_labeled": round(labeled_days / total_days * 100, 1),
            })

            log.info(
                "  N=%2d  T=%.4f  → LONG=%d (%.1f%%)  SHORT=%d (%.1f%%)"
                "  NO_TRADE=%d (%.1f%%)",
                N, T,
                n_long,  n_long  / total * 100,
                n_short, n_short / total * 100,
                n_no,    n_no    / total * 100,
            )

    grid = pd.DataFrame(rows)
    out  = RESULTS_DIR / f"{instrument}_grid.csv"
    grid.to_csv(out, index=False)
    log.info("  [%s] Grid saved → %s", instrument, out.name)
    return grid


def label_all_grid() -> None:
    """Run the full N×T grid search for all 8 instruments."""
    log.info("=== label_all_grid() starting ===")
    for instrument in ALL_INSTRUMENTS:
        run_grid_search(instrument)
    log.info("=== label_all_grid() complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    label_all_grid()
