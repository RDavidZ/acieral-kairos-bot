"""
strategy/swing_detector.py — Swing high/low detection for Acieral Kairos Bot

Two functions with different lookahead guarantees:

detect_swings(df, n)
    LOOKAHEAD version — for use ONLY in the labeller.
    A pivot high at bar i requires that high[i] is the maximum of the
    symmetric window [i-n, i+n].  This uses n future bars and must never
    be used as a live feature.

detect_swings_live(df, n)
    NO-LOOKAHEAD version — safe for feature engineering.
    A swing high at bar i means high[i] is strictly greater than every
    high in the preceding n bars [i-n, i-1].  No future bar is touched.
    Detects breakouts of the prior n-bar range rather than symmetric pivots.
"""

from collections import deque

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Labelling-only: symmetric pivot detection (uses lookahead)
# ---------------------------------------------------------------------------

def detect_swings(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Add symmetric pivot columns to *df*.

    Added columns
    -------------
    swing_high_pivot : bool — True at bar i if high[i] == max(high[i-n : i+n+1])
    swing_low_pivot  : bool — True at bar i if low[i]  == min(low[i-n : i+n+1])

    The first and last n rows are set to False (incomplete windows).

    WARNING: these columns look n bars into the future.  They must not be
    used as model features — only for labelling.
    """
    df = df.copy()
    window = 2 * n + 1

    # pandas rolling with center=True and min_periods=window gives NaN where
    # the full symmetric window is not available (first / last n rows).
    roll_max = df["high"].rolling(window, center=True, min_periods=window).max()
    roll_min = df["low"].rolling(window, center=True, min_periods=window).min()

    df["swing_high_pivot"] = (df["high"] == roll_max)
    df["swing_low_pivot"]  = (df["low"]  == roll_min)

    # Fill NaN edges (incomplete windows) with False
    df["swing_high_pivot"] = df["swing_high_pivot"].fillna(False)
    df["swing_low_pivot"]  = df["swing_low_pivot"].fillna(False)

    return df


# ---------------------------------------------------------------------------
# Live features: lookback-only swing detection (no lookahead)
# ---------------------------------------------------------------------------

def detect_swings_live(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Add live swing feature columns to *df*.

    Swing high definition (lookback-only):
        high[i] > max(high[i-1], ..., high[i-n])
    Swing low definition (lookback-only):
        low[i] < min(low[i-1], ..., low[i-n])

    Added columns
    -------------
    recent_swing_high         — price of the most recent lookback swing high
    recent_swing_low          — price of the most recent lookback swing low
    recent_swing_high_bar     — bars elapsed since that swing high
    recent_swing_low_bar      — bars elapsed since that swing low
    swing_swept_high          — 1 if within the last 5 bars any bar had
                                high > recent_swing_high AND close < recent_swing_high
    swing_swept_low           — 1 if within the last 5 bars any bar had
                                low < recent_swing_low AND close > recent_swing_low
    sweep_high_magnitude_atr  — max wick extension above the swept high level
                                (in ATR units); 0 if no sweep
    sweep_low_magnitude_atr   — max wick extension below the swept low level
    structure_bias            — rolling sum of last 6 swing-point comparisons:
                                +1 for HH (new swing high > previous swing high)
                                +1 for HL (new swing low  > previous swing low)
                                -1 for LH (new swing high < previous swing high)
                                -1 for LL (new swing low  < previous swing low)
                                Range roughly -6 to +6; positive = bullish.
    """
    df = df.copy()
    N = len(df)

    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    atrs   = df["atr_14"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 1: identify lookback-only swing points
    #   A bar is a swing high if its high > max of the PRECEDING n highs.
    #   Use shift(1).rolling(n) so bar i's own high is excluded.
    # ------------------------------------------------------------------
    preceding_max = df["high"].shift(1).rolling(n, min_periods=n).max().to_numpy()
    preceding_min = df["low"].shift(1).rolling(n, min_periods=n).min().to_numpy()

    # Strict greater-than / less-than to avoid false pivots in flat ranges
    is_swing_high = np.zeros(N, dtype=bool)
    is_swing_low  = np.zeros(N, dtype=bool)

    valid = ~np.isnan(preceding_max)
    is_swing_high[valid] = highs[valid] > preceding_max[valid]

    valid = ~np.isnan(preceding_min)
    is_swing_low[valid]  = lows[valid] < preceding_min[valid]

    # ------------------------------------------------------------------
    # Step 2: carry-forward most recent swing values and indices,
    #         and compute structure_bias from the rolling 6-event window.
    # ------------------------------------------------------------------
    recent_sh_val = np.full(N, np.nan)
    recent_sh_idx = np.full(N, -1, dtype=np.intp)
    recent_sl_val = np.full(N, np.nan)
    recent_sl_idx = np.full(N, -1, dtype=np.intp)
    structure_bias = np.zeros(N, dtype=np.float64)

    last_sh_val: float = np.nan
    last_sh_idx: int   = -1
    last_sl_val: float = np.nan
    last_sl_idx: int   = -1
    prev_sh_val: float = np.nan   # previous swing high (for HH/LH comparison)
    prev_sl_val: float = np.nan   # previous swing low  (for HL/LL comparison)

    events: deque[float] = deque(maxlen=6)   # last 6 ±1 structural events

    for i in range(N):
        if is_swing_high[i]:
            if not np.isnan(prev_sh_val):
                events.append(1.0 if highs[i] > prev_sh_val else -1.0)
            prev_sh_val = highs[i]
            last_sh_val = highs[i]
            last_sh_idx = i

        if is_swing_low[i]:
            if not np.isnan(prev_sl_val):
                events.append(1.0 if lows[i] > prev_sl_val else -1.0)
            prev_sl_val = lows[i]
            last_sl_val = lows[i]
            last_sl_idx = i

        recent_sh_val[i]  = last_sh_val
        recent_sh_idx[i]  = last_sh_idx
        recent_sl_val[i]  = last_sl_val
        recent_sl_idx[i]  = last_sl_idx
        structure_bias[i] = float(sum(events))

    # ------------------------------------------------------------------
    # Step 3: bars-elapsed since the most recent swing
    # ------------------------------------------------------------------
    idx_range = np.arange(N, dtype=np.float64)

    recent_sh_bar = np.where(recent_sh_idx >= 0, idx_range - recent_sh_idx, np.nan)
    recent_sl_bar = np.where(recent_sl_idx >= 0, idx_range - recent_sl_idx, np.nan)

    # ------------------------------------------------------------------
    # Step 4: vectorised sweep detection.
    #
    #   Key insight: a sweep bar j that wicks above the prior swing high
    #   will often *itself* become a new swing high, advancing recent_sh_val
    #   to highs[j] before any post-loop check runs.  Checking
    #   recent_sh_val[i] (current level at bar i) against highs[j] would
    #   therefore miss the sweep — the reference level has already moved.
    #
    #   Correct approach:
    #     prev_sh_val[j] = recent_sh_val[j-1]  (the swing high level that
    #     was current BEFORE bar j ran its own swing update).
    #
    #   A sweep occurs at bar j when:
    #     high[j] > prev_sh_val[j]  AND  close[j] < prev_sh_val[j]
    #
    #   We then roll that binary event over a 5-bar window so that bars
    #   up to 4 bars after the sweep still see swing_swept_high = 1.
    # ------------------------------------------------------------------

    # Level that was "most recent swing high" just BEFORE bar j processed
    prev_sh_val = np.concatenate([[np.nan], recent_sh_val[:-1]])
    prev_sl_val = np.concatenate([[np.nan], recent_sl_val[:-1]])

    has_sh = ~np.isnan(prev_sh_val)
    has_sl = ~np.isnan(prev_sl_val)

    is_sweep_high = has_sh & (highs  > prev_sh_val) & (closes < prev_sh_val)
    is_sweep_low  = has_sl & (lows   < prev_sl_val) & (closes > prev_sl_val)

    # Per-bar sweep magnitude (0 where no sweep)
    safe_atr = np.where(atrs > 0, atrs, np.nan)
    raw_high_mag = np.where(is_sweep_high, (highs - prev_sh_val) / safe_atr, 0.0)
    raw_low_mag  = np.where(is_sweep_low,  (prev_sl_val - lows)  / safe_atr, 0.0)
    raw_high_mag = np.nan_to_num(raw_high_mag)
    raw_low_mag  = np.nan_to_num(raw_low_mag)

    # Roll: within the last 5 bars, was there a sweep?  Take max of the
    # 5-bar window so both the flag and magnitude propagate forward.
    swept_high     = pd.Series(is_sweep_high.astype(float)) \
                       .rolling(5, min_periods=1).max().to_numpy()
    swept_low      = pd.Series(is_sweep_low.astype(float)) \
                       .rolling(5, min_periods=1).max().to_numpy()
    sweep_high_mag = pd.Series(raw_high_mag) \
                       .rolling(5, min_periods=1).max().to_numpy()
    sweep_low_mag  = pd.Series(raw_low_mag) \
                       .rolling(5, min_periods=1).max().to_numpy()

    # ------------------------------------------------------------------
    # Assign all columns
    # ------------------------------------------------------------------
    df["recent_swing_high"]        = recent_sh_val
    df["recent_swing_low"]         = recent_sl_val
    df["recent_swing_high_bar"]    = recent_sh_bar
    df["recent_swing_low_bar"]     = recent_sl_bar
    df["swing_swept_high"]         = swept_high
    df["swing_swept_low"]          = swept_low
    df["sweep_high_magnitude_atr"] = sweep_high_mag
    df["sweep_low_magnitude_atr"]  = sweep_low_mag
    df["structure_bias"]           = structure_bias

    return df
