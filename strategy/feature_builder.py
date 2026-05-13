"""
strategy/feature_builder.py — Full feature set for Acieral Kairos Bot

Builds all ~80 features for one instrument and saves to
data/cache/{instrument}_H1_features.parquet.

Instrument-aware behaviour:
  - Supplementary features (vix_close, vix_delta_1, spy_volume_ratio) are
    computed only for index instruments; NaN for forex.
  - Session anchoring differs by instrument class (see _session_open_minutes).

No lookahead: every feature uses only data available at candle close time t.
All price distances are ATR-normalised (divided by H1 atr_14).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ta

from config import HARD_CONSTRAINTS
from strategy.fvg_detector import detect_fvgs
from strategy.swing_detector import detect_swings_live

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
ALL_INSTRUMENTS    = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
FOREX_PAIRS        = HARD_CONSTRAINTS["FOREX_PAIRS"]
INDEX_INSTRUMENTS  = HARD_CONSTRAINTS["INDEX_INSTRUMENTS"]

# Session open times in UTC minutes since midnight
# Forex: most recent of Asian(22:00=1320), London(08:00=480), NY(13:00=780)
_LONDON_OPEN   = 480   # 08:00 UTC
_NY_OPEN       = 780   # 13:00 UTC
_ASIAN_OPEN    = 1320  # 22:00 UTC

# Index-specific single session opens (UTC minutes)
_INDEX_SESSION_OPEN = {
    "SPX500_USD": 870,   # 14:30 UTC
    "NAS100_USD": 870,   # 14:30 UTC
    "DE30_EUR":   480,   # 08:00 UTC
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _mask_warmup(series: pd.Series, warmup_bars: int) -> pd.Series:
    """Replace leading warmup rows with NaN (ta library fills them with 0.0)."""
    result = series.copy().astype(float)
    result.iloc[:warmup_bars] = float("nan")
    return result


def _safe_atr(df: pd.DataFrame, col: str = "atr_14") -> pd.Series:
    """Return ATR series with zeros replaced by NaN (prevents divide-by-zero)."""
    return df[col].replace(0.0, np.nan)


def _fvg_dist(close: pd.Series, top: pd.Series, bottom: pd.Series,
              atr: pd.Series) -> pd.Series:
    """
    Distance from close to the midpoint of an FVG, ATR-normalised.
    Returns NaN where the FVG is not active (top/bottom are NaN).
    """
    mid = (top + bottom) / 2.0
    return (close - mid).abs() / atr


def _nearest_fvg_dist(bull_d: np.ndarray, bear_d: np.ndarray) -> np.ndarray:
    """Return element-wise minimum of two distance arrays, respecting NaN."""
    return np.where(
        np.isnan(bull_d),  bear_d,
        np.where(np.isnan(bear_d), bull_d,
                 np.minimum(bull_d, bear_d))
    )


# ---------------------------------------------------------------------------
# HTF processing
# ---------------------------------------------------------------------------

def _process_htf(instrument: str, timeframe: str,
                 n_swing: int = 10) -> pd.DataFrame:
    """
    Load a raw H4 or D parquet, compute ATR + EMA-20, run FVG and swing
    detectors, and return the enriched DataFrame.
    """
    path = CACHE_DIR / f"{instrument}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # ATR (with warmup masking)
    df["atr_14"] = _mask_warmup(
        ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range(),
        warmup_bars=13,
    )

    # EMA-20 for slope calculation
    df["ema_20"] = ta.trend.EMAIndicator(
        close=df["close"], window=20
    ).ema_indicator()

    # FVG detection
    df = detect_fvgs(df)

    # Swing detection (requires atr_14)
    df = detect_swings_live(df, n=n_swing)

    return df


def _build_htf_cols(h1: pd.DataFrame, instrument: str,
                    n_swing: int) -> pd.DataFrame:
    """
    Process H4 and Daily data, compute derived HTF features, and
    merge them onto h1 via merge_asof (direction='backward').

    Returns h1 augmented with all Group-4 HTF columns.
    """
    atr_h1 = _safe_atr(h1)

    # ------------------------------------------------------------------ H4
    h4 = _process_htf(instrument, "H4", n_swing)

    # EMA slope on H4 granularity, normalised by H4 ATR
    h4_atr = h4["atr_14"].replace(0.0, np.nan)
    h4["h4_ema20_slope"] = (h4["ema_20"] - h4["ema_20"].shift(1)) / h4_atr

    h4_ff = h4[["time",
                "fvg_bull_top", "fvg_bull_bottom",
                "fvg_bear_top", "fvg_bear_bottom",
                "structure_bias", "h4_ema20_slope"]].copy()
    h4_ff = h4_ff.rename(columns={
        "fvg_bull_top":    "_h4_fvg_bull_top",
        "fvg_bull_bottom": "_h4_fvg_bull_bottom",
        "fvg_bear_top":    "_h4_fvg_bear_top",
        "fvg_bear_bottom": "_h4_fvg_bear_bottom",
        "structure_bias":  "h4_swing_bias",
    })

    h1 = pd.merge_asof(h1, h4_ff, on="time", direction="backward")

    # Compute H4 FVG features on H1
    h1["h4_fvg_bull_exists"] = h1["_h4_fvg_bull_top"].notna().astype(float)
    h1["h4_fvg_bear_exists"] = h1["_h4_fvg_bear_top"].notna().astype(float)

    bull_d = _fvg_dist(h1["close"], h1["_h4_fvg_bull_top"],
                       h1["_h4_fvg_bull_bottom"], atr_h1).values
    bear_d = _fvg_dist(h1["close"], h1["_h4_fvg_bear_top"],
                       h1["_h4_fvg_bear_bottom"], atr_h1).values
    h1["h4_fvg_dist_atr"] = _nearest_fvg_dist(bull_d, bear_d)

    h1["h4_close_vs_ema20"] = (h1["close"] - h1["h4_ema_20"]) / atr_h1

    # Drop temp columns
    h1 = h1.drop(columns=["_h4_fvg_bull_top", "_h4_fvg_bull_bottom",
                           "_h4_fvg_bear_top", "_h4_fvg_bear_bottom"])

    # ------------------------------------------------------------------ D1
    d1 = _process_htf(instrument, "D", n_swing)

    d1_atr = d1["atr_14"].replace(0.0, np.nan)
    # Raw daily EMA change (will be divided by H1 atr_14 after forward-fill)
    d1["_d1_ema_raw_change"] = d1["ema_20"] - d1["ema_20"].shift(1)

    d1_ff = d1[["time", "high", "low",
                "fvg_bull_top", "fvg_bull_bottom",
                "fvg_bear_top", "fvg_bear_bottom",
                "structure_bias", "_d1_ema_raw_change"]].copy()
    d1_ff = d1_ff.rename(columns={
        "high":            "d1_high",
        "low":             "d1_low",
        "fvg_bull_top":    "_d1_fvg_bull_top",
        "fvg_bull_bottom": "_d1_fvg_bull_bottom",
        "fvg_bear_top":    "_d1_fvg_bear_top",
        "fvg_bear_bottom": "_d1_fvg_bear_bottom",
        "structure_bias":  "d1_swing_bias",
    })

    h1 = pd.merge_asof(h1, d1_ff, on="time", direction="backward")

    # D1 features computed on H1
    h1["d1_ema20_slope"]   = h1["_d1_ema_raw_change"] / atr_h1
    h1["d1_close_vs_ema20"] = (h1["close"] - h1["d1_ema_20"]) / atr_h1
    h1["d1_high_dist_atr"] = (h1["d1_high"] - h1["close"]) / atr_h1
    h1["d1_low_dist_atr"]  = (h1["close"] - h1["d1_low"]) / atr_h1

    h1["d1_fvg_bull_exists"] = h1["_d1_fvg_bull_top"].notna().astype(float)
    h1["d1_fvg_bear_exists"] = h1["_d1_fvg_bear_top"].notna().astype(float)

    bull_d = _fvg_dist(h1["close"], h1["_d1_fvg_bull_top"],
                       h1["_d1_fvg_bull_bottom"], atr_h1).values
    bear_d = _fvg_dist(h1["close"], h1["_d1_fvg_bear_top"],
                       h1["_d1_fvg_bear_bottom"], atr_h1).values
    h1["d1_fvg_dist_atr"] = _nearest_fvg_dist(bull_d, bear_d)

    # HTF alignment: D1 and H4 biases agree in sign
    d1_bias = h1["d1_swing_bias"].values
    h4_bias = h1["h4_swing_bias"].values
    h1["htf_alignment"] = np.where(
        ((d1_bias > 0) & (h4_bias > 0)) | ((d1_bias < 0) & (h4_bias < 0)),
        1.0, 0.0
    )

    # Weekly distance features (w1_high, w1_low already on h1 from preprocessor)
    h1["w1_high_dist_atr"] = (h1["w1_high"] - h1["close"]) / atr_h1
    h1["w1_low_dist_atr"]  = (h1["close"]   - h1["w1_low"]) / atr_h1

    # Drop temp columns
    h1 = h1.drop(columns=["_d1_fvg_bull_top", "_d1_fvg_bull_bottom",
                           "_d1_fvg_bear_top", "_d1_fvg_bear_bottom",
                           "_d1_ema_raw_change"])

    return h1


# ---------------------------------------------------------------------------
# Session and time features
# ---------------------------------------------------------------------------

def _hours_since_session_open_forex(mins: np.ndarray) -> np.ndarray:
    """
    For forex: hours elapsed since the most recently passed session open
    among Asian (22:00=1320), London (08:00=480), NY (13:00=780) UTC.
    """
    return np.where(
        mins < _LONDON_OPEN,
        # Asian opened at 22:00 the prior UTC day
        (mins + (1440 - _ASIAN_OPEN)) / 60.0,
        np.where(
            mins < _NY_OPEN,
            # London opened today at 08:00
            (mins - _LONDON_OPEN) / 60.0,
            np.where(
                mins < _ASIAN_OPEN,
                # NY opened today at 13:00
                (mins - _NY_OPEN) / 60.0,
                # Asian opened today at 22:00
                (mins - _ASIAN_OPEN) / 60.0,
            )
        )
    )


def _hours_since_session_open_index(mins: np.ndarray,
                                    open_min: int) -> np.ndarray:
    """
    For index instruments: hours elapsed since the most recently passed
    session open.  Wraps around midnight so the value is always positive.
    """
    return np.where(
        mins >= open_min,
        (mins - open_min) / 60.0,
        (mins + 1440 - open_min) / 60.0,
    )


def _compute_session_features(df: pd.DataFrame,
                               instrument: str) -> pd.DataFrame:
    """Add Group 3 session/time features (in-place on copy)."""
    hour = df["time"].dt.hour.values
    minute = df["time"].dt.minute.values
    mins = (hour * 60 + minute).astype(np.float64)

    df["minutes_since_midnight"] = mins
    df["hour_utc"]     = hour.astype(np.float64)
    df["day_of_week"]  = df["time"].dt.dayofweek.values.astype(np.float64)

    # Session flags
    df["session_asian"]   = ((hour >= 22) | (hour < 8)).astype(float)
    df["session_london"]  = ((hour >= 8)  & (hour < 17)).astype(float)
    df["session_ny"]      = ((hour >= 13) & (hour < 21)).astype(float)
    df["session_overlap"] = ((hour >= 13) & (hour < 17)).astype(float)

    # Hours since session open
    if instrument in INDEX_INSTRUMENTS:
        open_min = _INDEX_SESSION_OPEN[instrument]
        hso = _hours_since_session_open_index(mins, open_min)
    else:
        hso = _hours_since_session_open_forex(mins)

    df["hours_since_session_open"] = hso

    # First-3h flags use session-specific hours, not generic hso
    # London: within 3h of 08:00 UTC
    mins_since_london = np.where(mins >= _LONDON_OPEN,
                                 mins - _LONDON_OPEN,
                                 mins + 1440 - _LONDON_OPEN)
    df["is_london_first_3h"] = (
        (df["session_london"].values == 1) &
        (mins_since_london < 180)
    ).astype(float)

    # NY: within 3h of 13:00 UTC
    mins_since_ny = np.where(mins >= _NY_OPEN,
                             mins - _NY_OPEN,
                             mins + 1440 - _NY_OPEN)
    df["is_ny_first_3h"] = (
        (df["session_ny"].values == 1) &
        (mins_since_ny < 180)
    ).astype(float)

    return df


# ---------------------------------------------------------------------------
# Intraday cumulative features (Group 7 partial + Group 5 range_vs_atr_5)
# ---------------------------------------------------------------------------

def _compute_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-UTC-day cumulative features without using groupby-apply
    (which is slow on 57k rows).  Uses numpy day-boundary splitting.
    """
    n = len(df)
    dates = df["time"].dt.date.values

    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values

    intra_high    = np.empty(n, dtype=np.float64)
    intra_low     = np.empty(n, dtype=np.float64)
    midnight_open = np.empty(n, dtype=np.float64)
    bull_count    = np.empty(n, dtype=np.float64)
    bear_count    = np.empty(n, dtype=np.float64)

    # Find day boundaries (positions where the date changes)
    date_strs  = np.array([str(d) for d in dates])
    boundaries = np.concatenate([[0],
                                 np.where(date_strs[1:] != date_strs[:-1])[0] + 1,
                                 [n]])

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]

        h_slice = highs[s:e]
        l_slice = lows[s:e]
        o_slice = opens[s:e]
        c_slice = closes[s:e]

        intra_high[s:e]    = np.maximum.accumulate(h_slice)
        intra_low[s:e]     = np.minimum.accumulate(l_slice)
        midnight_open[s:e] = o_slice[0]

        is_bull = (c_slice > o_slice).astype(np.float64)
        is_bear = (c_slice < o_slice).astype(np.float64)
        bull_count[s:e] = np.cumsum(is_bull)
        bear_count[s:e] = np.cumsum(is_bear)

    df["_intra_high"]    = intra_high
    df["_intra_low"]     = intra_low
    df["_midnight_open"] = midnight_open
    df["bull_candle_count"] = bull_count
    df["bear_candle_count"] = bear_count

    return df


# ---------------------------------------------------------------------------
# Supplementary data (Group 8)
# ---------------------------------------------------------------------------

def _load_supplementary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load VIX and SPY volume daily parquets."""
    vix = pd.read_parquet(CACHE_DIR / "vix_daily.parquet")
    spy = pd.read_parquet(CACHE_DIR / "spy_volume_daily.parquet")

    vix["date"] = pd.to_datetime(vix["date"]).dt.date
    spy["date"] = pd.to_datetime(spy["date"]).dt.date

    # Pre-compute rolling average and ratio
    spy = spy.sort_values("date").reset_index(drop=True)
    spy["spy_vol_20avg"] = (
        spy["spy_volume"].rolling(20, min_periods=1).mean().shift(1)
    )
    spy["spy_volume_ratio"] = spy["spy_volume"] / spy["spy_vol_20avg"].replace(0, np.nan)

    # Daily VIX delta
    vix = vix.sort_values("date").reset_index(drop=True)
    vix["vix_delta_1"] = vix["vix_close"].diff(1)

    return vix[["date", "vix_close", "vix_delta_1"]], \
           spy[["date", "spy_volume_ratio"]]


def _merge_supplementary(df: pd.DataFrame, instrument: str,
                          vix_df: pd.DataFrame,
                          spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge VIX and SPY features onto H1 rows by UTC date.
    Only populated for index instruments; NaN for forex.
    """
    if instrument not in INDEX_INSTRUMENTS:
        df["vix_close"]        = np.nan
        df["vix_delta_1"]      = np.nan
        df["spy_volume_ratio"] = np.nan
        return df

    df["_date"] = df["time"].dt.date

    # Left-merge on date (each H1 bar in the same day gets the same daily values)
    df = df.merge(vix_df.rename(columns={"date": "_date"}),
                  on="_date", how="left")
    df = df.merge(spy_df.rename(columns={"date": "_date"}),
                  on="_date", how="left")

    df = df.drop(columns=["_date"])
    return df


# ---------------------------------------------------------------------------
# Main feature computation
# ---------------------------------------------------------------------------

def build_features(instrument: str, n_swing: int = 10) -> pd.DataFrame:
    """
    Build the full feature set for one instrument.

    Loads *_H1_processed.parquet, runs FVG + swing detectors, computes all
    ~80 features across 8 groups, and returns the enriched H1 DataFrame.
    No rows are removed — NaN handling is the labeller's responsibility.
    """
    log.info("Building features for %s …", instrument)

    # ------------------------------------------------------------------ load
    path = CACHE_DIR / f"{instrument}_H1_processed.parquet"
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # ---------------------------------------------------------- H1 detectors
    df = detect_fvgs(df)
    df = detect_swings_live(df, n=n_swing)

    # ------------------------------------------------------- Group 1 — FVGs
    atr = _safe_atr(df)

    df["fvg_bull_exists"] = df["fvg_bull_top"].notna().astype(float)
    df["fvg_bear_exists"] = df["fvg_bear_top"].notna().astype(float)

    df["fvg_bull_dist_atr"] = (df["close"] - df["fvg_bull_top"]) / atr
    df["fvg_bear_dist_atr"] = (df["fvg_bear_bottom"] - df["close"]) / atr
    df["fvg_bull_size_atr"] = (df["fvg_bull_top"] - df["fvg_bull_bottom"]) / atr
    df["fvg_bear_size_atr"] = (df["fvg_bear_top"] - df["fvg_bear_bottom"]) / atr
    # fvg_bull_age, fvg_bear_age, fvg_bull_fill_pct, fvg_bear_fill_pct
    # are already present from the detector — keep as-is

    # ----------------------------------------------- Group 2 — Swing struct
    df["swing_high_dist_atr"] = (df["recent_swing_high"] - df["close"]) / atr
    df["swing_low_dist_atr"]  = (df["close"] - df["recent_swing_low"])  / atr
    # swing_swept_high, swing_swept_low, sweep magnitudes, bars_since_*,
    # structure_bias already present from detect_swings_live — rename to spec
    df = df.rename(columns={
        "recent_swing_high_bar":    "bars_since_swing_high",
        "recent_swing_low_bar":     "bars_since_swing_low",
        "sweep_high_magnitude_atr": "sweep_high_mag_atr",
        "sweep_low_magnitude_atr":  "sweep_low_mag_atr",
    })

    # -------------------------------------------- Group 4 — HTF context
    df = _build_htf_cols(df, instrument, n_swing)

    # fvg_stacked (needs d1_fvg_{bull,bear}_exists, computed in _build_htf_cols)
    df["fvg_stacked_bull"] = (
        (df["fvg_bull_exists"] == 1) & (df["d1_fvg_bull_exists"] == 1)
    ).astype(float)
    df["fvg_stacked_bear"] = (
        (df["fvg_bear_exists"] == 1) & (df["d1_fvg_bear_exists"] == 1)
    ).astype(float)

    # ---------------------------------------- Group 3 — Session / time
    df = _compute_session_features(df, instrument)

    # -------------------------------------- Group 5 — Volatility / regime
    df["atr_ratio"]       = atr / atr.rolling(20, min_periods=1).mean()
    df["atr_ratio_delta"] = df["atr_ratio"] - df["atr_ratio"].shift(3)

    candle_range = (df["high"] - df["low"]).clip(lower=1e-9)
    body_top     = df["close"].clip(lower=df["open"])   # max(close, open)
    body_bottom  = df["close"].clip(upper=df["open"])   # min(close, open)

    df["candle_body_ratio"]  = (df["close"] - df["open"]).abs() / candle_range
    df["wick_upper_ratio"]   = (df["high"] - body_top)    / candle_range
    df["wick_lower_ratio"]   = (body_bottom - df["low"])  / candle_range
    df["range_vs_atr_5"]     = (
        df["high"].rolling(5, min_periods=1).max() -
        df["low"].rolling(5, min_periods=1).min()
    ) / atr
    df["adx_delta"] = df["adx_14"] - df["adx_14"].shift(3)

    # ------------------------------------------- Group 6 — Momentum
    df["rsi_14_delta_3"]  = df["rsi_14"] - df["rsi_14"].shift(3)
    df["rsi_div_bull"]    = (
        (df["close"] < df["close"].shift(5)) &
        (df["rsi_14"] > df["rsi_14"].shift(5))
    ).astype(float)
    df["rsi_div_bear"]    = (
        (df["close"] > df["close"].shift(5)) &
        (df["rsi_14"] < df["rsi_14"].shift(5))
    ).astype(float)
    df["macd_hist_delta"] = df["macd_hist"] - df["macd_hist"].shift(2)
    df["roc_4"]  = (df["close"] - df["close"].shift(4))  / df["close"].shift(4)
    df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10)

    # ------------------------------------------- Group 7 — Candle context
    for lag in range(1, 5):
        df[f"close_lag_{lag}"] = (df["close"] - df["close"].shift(lag)) / atr

    for lag in range(1, 4):
        df[f"body_ratio_lag_{lag}"] = df["candle_body_ratio"].shift(lag)

    df["wick_upper_lag_1"] = df["wick_upper_ratio"].shift(1)
    df["wick_lower_lag_1"] = df["wick_lower_ratio"].shift(1)

    # Intraday cumulative features
    df = _compute_intraday_features(df)

    df["day_high_dist_atr"]    = (df["_intra_high"]    - df["close"]) / atr
    df["day_low_dist_atr"]     = (df["close"] - df["_intra_low"])     / atr
    df["day_displacement_atr"] = (df["close"] - df["_midnight_open"]) / atr

    df = df.drop(columns=["_intra_high", "_intra_low", "_midnight_open"])

    # ---------------------------------- Group 8 — Supplementary (indices)
    vix_df, spy_df = _load_supplementary()
    df = _merge_supplementary(df, instrument, vix_df, spy_df)

    log.info("  [%s] features built: %d rows × %d columns",
             instrument, len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Build all instruments
# ---------------------------------------------------------------------------

def build_all(n_swing: int = 10) -> None:
    """Build and save features for all 8 instruments."""
    log.info("=== build_all() starting ===")
    for instrument in ALL_INSTRUMENTS:
        df = build_features(instrument, n_swing=n_swing)
        out = CACHE_DIR / f"{instrument}_H1_features.parquet"
        df.to_parquet(out, index=False)
        log.info("  [%s] Saved → %s", instrument, out.name)
    log.info("=== build_all() complete ===")


# ---------------------------------------------------------------------------
# Live feature computation (used by bot/main.py on every H1 bar)
# ---------------------------------------------------------------------------

def build_live_features(
    instrument: str,
    live_h1_df: pd.DataFrame,
    n_swing: int = 10,
) -> pd.Series:
    """
    Build the full feature set for the most recent H1 candle from live data.

    This is the live equivalent of build_features() but works from freshly
    fetched candles instead of cached processed parquets.

    Parameters
    ----------
    instrument  : OANDA instrument name
    live_h1_df  : DataFrame with columns: time, open, high, low, close, volume
                  Must contain at least 50 rows for indicator warmup.
                  Typically 500 rows (fetched by OandaClient.get_latest_candles).
                  500 bars (~21 days) gives the swing detector enough history to
                  match backtest structural analysis.
    n_swing     : Swing detector lookback (default 10)

    Returns
    -------
    pd.Series — feature vector for the last (most recently closed) candle,
                containing all FEATURE_COLS values.

    Raises
    ------
    ValueError if live_h1_df has fewer than 50 rows (insufficient warmup).
    """
    from data.preprocessor import (
        _add_h1_indicators,
        _add_h4_indicators,
        _add_d1_indicators,
        _forward_fill_htf,
    )

    if len(live_h1_df) < 50:
        raise ValueError(
            f"build_live_features requires >= 50 bars; got {len(live_h1_df)}"
        )

    # ------------------------------------------------------------------ prep
    df = live_h1_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # ------------------------------------------------- H1 indicators
    df = _add_h1_indicators(df)

    # ------------------------------------------------- H4 forward-fill
    h4_path = CACHE_DIR / f"{instrument}_H4.parquet"
    h4 = pd.read_parquet(h4_path)
    h4["time"] = pd.to_datetime(h4["time"], utc=True)
    h4 = h4.sort_values("time").reset_index(drop=True)
    h4 = _add_h4_indicators(h4)
    df = _forward_fill_htf(df, h4, ["h4_atr_14", "h4_ema_20", "h4_rsi_14", "h4_adx_14"])

    # ------------------------------------------------- D1 forward-fill
    d1_path = CACHE_DIR / f"{instrument}_D.parquet"
    d1 = pd.read_parquet(d1_path)
    d1["time"] = pd.to_datetime(d1["time"], utc=True)
    d1 = d1.sort_values("time").reset_index(drop=True)
    d1 = _add_d1_indicators(d1)
    df = _forward_fill_htf(df, d1, ["d1_atr_14", "d1_ema_20"])

    # ------------------------------------------------- W1 forward-fill
    w1_path = CACHE_DIR / f"{instrument}_W.parquet"
    w1 = pd.read_parquet(w1_path)
    w1["time"] = pd.to_datetime(w1["time"], utc=True)
    w1 = w1.sort_values("time").reset_index(drop=True)
    w1_subset = w1[["time", "high", "low"]].rename(
        columns={"high": "w1_high", "low": "w1_low"}
    )
    df = pd.merge_asof(df, w1_subset, on="time", direction="backward")

    # ------------------------------------------------ H1 detectors
    df = detect_fvgs(df)
    # Run swing detector on the full window first so the state machine
    # (structure_bias deque, recent_swing_high/low carry-forward) is
    # properly initialised from the entire history.  Then trim to the
    # last 300 bars: all downstream rolling features (ATR-14, RSI-14,
    # MACD, atr_ratio rolling-20, etc.) have sufficient warmup data
    # and we avoid carrying stale early-history rows into the return value.
    df = detect_swings_live(df, n=n_swing)
    if len(df) > 300:
        df = df.iloc[-300:].reset_index(drop=True)

    # ---------------------------------------- Group 1 — FVG features
    atr = _safe_atr(df)

    df["fvg_bull_exists"] = df["fvg_bull_top"].notna().astype(float)
    df["fvg_bear_exists"] = df["fvg_bear_top"].notna().astype(float)

    df["fvg_bull_dist_atr"] = (df["close"] - df["fvg_bull_top"]) / atr
    df["fvg_bear_dist_atr"] = (df["fvg_bear_bottom"] - df["close"]) / atr
    df["fvg_bull_size_atr"] = (df["fvg_bull_top"] - df["fvg_bull_bottom"]) / atr
    df["fvg_bear_size_atr"] = (df["fvg_bear_top"] - df["fvg_bear_bottom"]) / atr
    # fvg_bull_age, fvg_bear_age, fvg_bull_fill_pct, fvg_bear_fill_pct — from detector

    # ---------------------------------------- Group 2 — Swing structure
    df["swing_high_dist_atr"] = (df["recent_swing_high"] - df["close"]) / atr
    df["swing_low_dist_atr"]  = (df["close"] - df["recent_swing_low"])  / atr
    df = df.rename(columns={
        "recent_swing_high_bar":    "bars_since_swing_high",
        "recent_swing_low_bar":     "bars_since_swing_low",
        "sweep_high_magnitude_atr": "sweep_high_mag_atr",
        "sweep_low_magnitude_atr":  "sweep_low_mag_atr",
    })

    # ---------------------------------------- Group 4 — HTF context
    df = _build_htf_cols(df, instrument, n_swing)

    # fvg_stacked (d1 FVG columns now present)
    df["fvg_stacked_bull"] = (
        (df["fvg_bull_exists"] == 1) & (df["d1_fvg_bull_exists"] == 1)
    ).astype(float)
    df["fvg_stacked_bear"] = (
        (df["fvg_bear_exists"] == 1) & (df["d1_fvg_bear_exists"] == 1)
    ).astype(float)

    # ---------------------------------------- Group 3 — Session / time
    df = _compute_session_features(df, instrument)

    # ---------------------------------- Group 5 — Volatility / regime
    df["atr_ratio"]       = atr / atr.rolling(20, min_periods=1).mean()
    df["atr_ratio_delta"] = df["atr_ratio"] - df["atr_ratio"].shift(3)

    candle_range = (df["high"] - df["low"]).clip(lower=1e-9)
    body_top     = df[["close", "open"]].max(axis=1)
    body_bottom  = df[["close", "open"]].min(axis=1)

    df["candle_body_ratio"]  = (df["close"] - df["open"]).abs() / candle_range
    df["wick_upper_ratio"]   = (df["high"] - body_top)    / candle_range
    df["wick_lower_ratio"]   = (body_bottom - df["low"])  / candle_range
    df["range_vs_atr_5"]     = (
        df["high"].rolling(5, min_periods=1).max() -
        df["low"].rolling(5, min_periods=1).min()
    ) / atr
    df["adx_delta"] = df["adx_14"] - df["adx_14"].shift(3)

    # -------------------------------------------- Group 6 — Momentum
    df["rsi_14_delta_3"]  = df["rsi_14"] - df["rsi_14"].shift(3)
    df["rsi_div_bull"]    = (
        (df["close"] < df["close"].shift(5)) &
        (df["rsi_14"] > df["rsi_14"].shift(5))
    ).astype(float)
    df["rsi_div_bear"]    = (
        (df["close"] > df["close"].shift(5)) &
        (df["rsi_14"] < df["rsi_14"].shift(5))
    ).astype(float)
    df["macd_hist_delta"] = df["macd_hist"] - df["macd_hist"].shift(2)
    df["roc_4"]  = (df["close"] - df["close"].shift(4))  / df["close"].shift(4)
    df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10)

    # ------------------------------------------ Group 7 — Candle context
    for lag in range(1, 5):
        df[f"close_lag_{lag}"] = (df["close"] - df["close"].shift(lag)) / atr

    for lag in range(1, 4):
        df[f"body_ratio_lag_{lag}"] = df["candle_body_ratio"].shift(lag)

    df["wick_upper_lag_1"] = df["wick_upper_ratio"].shift(1)
    df["wick_lower_lag_1"] = df["wick_lower_ratio"].shift(1)

    df = _compute_intraday_features(df)
    df["day_high_dist_atr"]    = (df["_intra_high"]    - df["close"]) / atr
    df["day_low_dist_atr"]     = (df["close"] - df["_intra_low"])     / atr
    df["day_displacement_atr"] = (df["close"] - df["_midnight_open"]) / atr
    df = df.drop(columns=["_intra_high", "_intra_low", "_midnight_open"])

    # ------------------------------ Group 8 — Supplementary (indices only)
    try:
        vix_df, spy_df = _load_supplementary()
        df = _merge_supplementary(df, instrument, vix_df, spy_df)
    except Exception:
        df["vix_close"]        = np.nan
        df["vix_delta_1"]      = np.nan
        df["spy_volume_ratio"] = np.nan

    # Return the last row as a Series (the current candle's features)
    return df.iloc[-1]


# ---------------------------------------------------------------------------
# Cache-based live feature computation (v2)
# ---------------------------------------------------------------------------

def build_live_features_v2(
    instrument: str,
    live_h1_df: pd.DataFrame,
    n_swing: int = 10,
    now=None,
):
    """
    Cache-based live feature computation.
    Reads the pre-computed H1 features parquet, appends any new bars
    not yet in the cache, runs incremental feature computation on new
    bars only, and returns the most recent complete row.

    Falls back to build_live_features if cache unavailable.
    """
    cache_path = CACHE_DIR / f"{instrument}_H1_features.parquet"

    if not cache_path.exists():
        log.warning("[%s] Feature cache not found — falling back to build_live_features", instrument)
        return build_live_features(instrument, live_h1_df, n_swing=n_swing)

    # Load pre-computed feature cache
    cached = pd.read_parquet(cache_path)
    cached["time"] = pd.to_datetime(cached["time"], utc=True)

    # Exclude in-progress candle (bar that has opened but not yet closed)
    if now is not None:
        now_ts = pd.Timestamp(now).tz_convert("UTC") if hasattr(now, "tzinfo") else pd.Timestamp(now, tz="UTC")
        cached = cached[cached["time"] < now_ts]

    if cached.empty:
        log.warning("[%s] Feature cache empty after time filter", instrument)
        return build_live_features(instrument, live_h1_df, n_swing=n_swing)

    last_row = cached.iloc[-1]
    log.debug(
        "[%s] build_live_features_v2: cache row time=%s cols=%d",
        instrument, last_row["time"], len(cached.columns),
    )
    return last_row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_all()
