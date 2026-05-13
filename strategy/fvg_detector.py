"""
strategy/fvg_detector.py — Fair Value Gap detection for Acieral Kairos Bot

FVG definitions (computed strictly at candle close — no lookahead):

  Bullish FVG formed at bar i when:  high[i-2] < low[i]
    The gap lies between the top of the two-bars-ago candle and the bottom
    of the current candle.  candle[i-1] (the "middle") does not bridge it.
    fvg_bull_bottom = high[i-2]
    fvg_bull_top    = low[i]

  Bearish FVG formed at bar i when:  low[i-2] > high[i]
    fvg_bear_top    = low[i-2]
    fvg_bear_bottom = high[i]

Fill tracking:
  - Bullish fill: price LOW enters the gap from above.
    fill_pct = (fvg_bull_top - running_min_low) / gap_size  clamped [0, 1]
  - Bearish fill: price HIGH enters the gap from below.
    fill_pct = (running_max_high - fvg_bear_bottom) / gap_size  clamped [0, 1]

State rules:
  - Only the MOST RECENT active FVG of each direction is tracked.
    A newer FVG always replaces the current one immediately.
  - When fill_pct >= 1.0 the FVG is invalidated; the slot is empty until
    the next FVG of that direction is created.
  - Columns are NaN while no active FVG exists.
"""

import numpy as np
import pandas as pd


def detect_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FVG columns to *df* (rows × columns stays the same — no rows removed).

    Added columns
    -------------
    fvg_bull_top, fvg_bull_bottom  — boundaries of the active bullish FVG
    fvg_bear_top, fvg_bear_bottom  — boundaries of the active bearish FVG
    fvg_bull_age    — bars elapsed since the active bullish FVG was created
    fvg_bear_age    — bars elapsed since the active bearish FVG was created
    fvg_bull_fill_pct — cumulative fill fraction of the active bullish FVG [0, 1)
    fvg_bear_fill_pct — cumulative fill fraction of the active bearish FVG [0, 1)

    All columns are NaN when no active FVG of that direction exists.
    """
    df = df.copy()
    n = len(df)

    highs  = df["high"].to_numpy(dtype=np.float64)
    lows   = df["low"].to_numpy(dtype=np.float64)

    # Output arrays (NaN by default)
    bull_top_arr      = np.full(n, np.nan)
    bull_bot_arr      = np.full(n, np.nan)
    bull_age_arr      = np.full(n, np.nan)
    bull_fill_arr     = np.full(n, np.nan)

    bear_top_arr      = np.full(n, np.nan)
    bear_bot_arr      = np.full(n, np.nan)
    bear_age_arr      = np.full(n, np.nan)
    bear_fill_arr     = np.full(n, np.nan)

    # -----------------------------------------------------------------------
    # Bullish FVG state
    # -----------------------------------------------------------------------
    bull_active  = False
    bull_top     = 0.0
    bull_bot     = 0.0
    bull_created = 0
    bull_min_low = 0.0   # running minimum low since FVG creation (tracks fill depth)

    # -----------------------------------------------------------------------
    # Bearish FVG state
    # -----------------------------------------------------------------------
    bear_active   = False
    bear_top      = 0.0
    bear_bot      = 0.0
    bear_created  = 0
    bear_max_high = 0.0  # running maximum high since FVG creation

    for i in range(n):
        # ------------------------------------------------------------------
        # 1. Check for new FVG creation at bar i (requires bars i-2, i-1, i)
        #    No lookahead: candle i is now closed.
        # ------------------------------------------------------------------
        if i >= 2:
            # Bullish: gap between top of bar i-2 and bottom of bar i
            if highs[i - 2] < lows[i]:
                bull_active  = True
                bull_bot     = highs[i - 2]
                bull_top     = lows[i]
                bull_created = i
                # At creation the low of bar i == bull_top → fill = 0
                bull_min_low = lows[i]

            # Bearish: gap between bottom of bar i-2 and top of bar i
            if lows[i - 2] > highs[i]:
                bear_active   = True
                bear_top      = lows[i - 2]
                bear_bot      = highs[i]
                bear_created  = i
                # At creation the high of bar i == bear_bot → fill = 0
                bear_max_high = highs[i]

        # ------------------------------------------------------------------
        # 2. Update running price extremes and compute fill fractions
        # ------------------------------------------------------------------
        if bull_active:
            if lows[i] < bull_min_low:
                bull_min_low = lows[i]

            gap = bull_top - bull_bot
            if gap > 0.0:
                # How far has low penetrated from the top of the gap downward
                fill = (bull_top - max(bull_min_low, bull_bot)) / gap
                fill = min(1.0, max(0.0, fill))
            else:
                fill = 1.0  # degenerate zero-width gap — treat as instantly filled

            if fill >= 1.0:
                bull_active = False          # FVG fully filled → invalidate
            else:
                bull_top_arr[i]  = bull_top
                bull_bot_arr[i]  = bull_bot
                bull_age_arr[i]  = float(i - bull_created)
                bull_fill_arr[i] = fill

        if bear_active:
            if highs[i] > bear_max_high:
                bear_max_high = highs[i]

            gap = bear_top - bear_bot
            if gap > 0.0:
                fill = (min(bear_max_high, bear_top) - bear_bot) / gap
                fill = min(1.0, max(0.0, fill))
            else:
                fill = 1.0

            if fill >= 1.0:
                bear_active = False
            else:
                bear_top_arr[i]  = bear_top
                bear_bot_arr[i]  = bear_bot
                bear_age_arr[i]  = float(i - bear_created)
                bear_fill_arr[i] = fill

    df["fvg_bull_top"]       = bull_top_arr
    df["fvg_bull_bottom"]    = bull_bot_arr
    df["fvg_bull_age"]       = bull_age_arr
    df["fvg_bull_fill_pct"]  = bull_fill_arr
    df["fvg_bear_top"]       = bear_top_arr
    df["fvg_bear_bottom"]    = bear_bot_arr
    df["fvg_bear_age"]       = bear_age_arr
    df["fvg_bear_fill_pct"]  = bear_fill_arr

    return df
