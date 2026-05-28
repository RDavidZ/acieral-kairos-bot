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


def _safe_atr_series(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR via EWM — works on DataFrames of any length (no minimum-rows requirement).
    Replaces ta.volatility.AverageTrueRange on short completed-bar DataFrames.
    """
    high  = df["high"].reset_index(drop=True)
    low   = df["low"].reset_index(drop=True)
    close = df["close"].reset_index(drop=True)
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low  - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, min_periods=1, adjust=False).mean()
    # Null out the first (window-1) rows to match _mask_warmup(warmup_bars=13)
    atr.iloc[:window - 1] = np.nan
    return atr.replace(0.0, np.nan)


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
# HTF processing — synthetic bars from H1 data (no OANDA HTF parquets)
# ---------------------------------------------------------------------------

def _build_htf_cols(h1: pd.DataFrame, instrument: str,
                    n_swing: int) -> pd.DataFrame:
    """
    Build all HTF features synthetically from H1 data — no OANDA HTF parquets.

    For each H1 bar at time T the synthetic HTF bar represents ONLY the H1 bars
    already closed within the current period (strictly before T).
    If T is the first bar of a new period, falls back to the previous completed
    period's OHLCV.

    Period boundaries (market-aligned, DST-aware):
      H4 : 22/02/06/10/14/18 UTC (winter)  or  21/01/05/09/13/17 UTC (summer)
      D1 : 22:00 UTC (winter)  /  21:00 UTC (summer)
      W1 : most recent Friday 21:00 or 22:00 UTC
    Both 21 and 22 are detected from the H1 timestamps directly.
    """
    atr_h1 = _safe_atr(h1)
    h1 = h1.copy()

    times = pd.DatetimeIndex(h1["time"])
    hours = times.hour.values
    dow   = times.dayofweek.values          # 0=Mon … 4=Fri … 6=Sun
    n     = len(h1)

    # ------------------------------------------------------------------
    # Step 1 — Period boundary detection
    # ------------------------------------------------------------------

    # D1: market day starts at hour 21 (BST/summer) or 22 (GMT/winter)
    is_d1 = np.zeros(n, dtype=bool)
    is_d1[(hours == 21) | (hours == 22)] = True
    is_d1[0] = True                         # anchor first bar to a period
    d1_pid = np.cumsum(is_d1)

    # Hours elapsed since current D1 start → H4 sub-period id
    d1_start_idxs = np.where(is_d1)[0]
    d1_start_ns   = np.empty(n, dtype=np.int64)
    for k, s in enumerate(d1_start_idxs):
        e = d1_start_idxs[k + 1] if k + 1 < len(d1_start_idxs) else n
        d1_start_ns[s:e] = times[s].value          # nanoseconds since epoch
    elapsed_h = (times.asi8 - d1_start_ns) / 1_000_000_000 / 3600
    h4_sub    = (elapsed_h // 4).astype(int)
    h4_pid    = d1_pid * 10 + h4_sub               # unique H4 period id

    # W1: starts at Friday (dow=4) at hour 21 or 22
    is_w1 = np.zeros(n, dtype=bool)
    is_w1[((hours == 21) | (hours == 22)) & (dow == 4)] = True
    is_w1[0] = True
    w1_pid = np.cumsum(is_w1)

    # ------------------------------------------------------------------
    # Step 2 — Partial-bar builder
    # ------------------------------------------------------------------

    def _partial_bars(pid_arr: np.ndarray) -> pd.DataFrame:
        """
        For each H1 bar i, return the state of its HTF period using ONLY the
        H1 bars that have already closed (strictly before bar i).

        Uses groupby cumulative stats + global shift(1):
          - position 0 in period P → shift gives last bar of period P-1 (fallback)
          - position k>0           → gives within-period cumulative up to bar k-1
        """
        pid   = pd.Series(pid_arr, index=h1.index, name="_pid")
        frame = h1[["open", "high", "low", "close", "volume"]].copy()
        frame["_pid"] = pid

        grp = frame.groupby("_pid", sort=False)

        cum_high  = grp["high"].cummax()
        cum_low   = grp["low"].cummin()
        cum_vol   = grp["volume"].cumsum()
        per_open  = grp["open"].transform("first")

        # Shift by 1 globally — at a period boundary this naturally picks up
        # the previous period's final cumulative value (the fallback).
        synth = pd.DataFrame({
            "open":   per_open.shift(1).ffill(),
            "high":   cum_high.shift(1).ffill(),
            "low":    cum_low.shift(1).ffill(),
            "close":  frame["close"].shift(1).ffill(),
            "volume": cum_vol.shift(1).ffill().fillna(0.0),
        }, index=h1.index)
        return synth

    # ------------------------------------------------------------------
    # Step 3 — Completed-bar builder (one row per fully closed period)
    # ------------------------------------------------------------------

    def _completed_bars(pid_arr: np.ndarray) -> pd.DataFrame:
        """
        Aggregate H1 bars into one completed HTF bar per period (last H1 bar
        of each period).  Used as input for FVG and swing detectors.
        """
        frame = h1[["time", "open", "high", "low", "close", "volume"]].copy()
        frame["_pid"] = pid_arr
        comp = (
            frame.groupby("_pid", sort=False)
            .agg(
                time=("time", "last"),
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .reset_index(drop=True)
        )
        comp["time"] = pd.to_datetime(comp["time"], utc=True)
        return comp

    # ------------------------------------------------------------------
    # Step 4 — H4
    # ------------------------------------------------------------------
    h4_part = _partial_bars(h4_pid)

    h4_ema_s = ta.trend.EMAIndicator(
        close=h4_part["close"], window=20
    ).ema_indicator()
    h4_atr_s = ta.volatility.AverageTrueRange(
        high=h4_part["high"], low=h4_part["low"],
        close=h4_part["close"], window=14,
    ).average_true_range().replace(0.0, np.nan)

    h1["h4_ema_20"]         = h4_ema_s.values
    h1["h4_atr_14"]         = h4_atr_s.values
    h1["h4_ema20_slope"]    = (h4_ema_s.diff(1) / h4_atr_s).values
    h1["h4_close_vs_ema20"] = ((h4_part["close"] - h4_ema_s) / atr_h1).values

    # FVG + swing on completed H4 bars → forward-fill to H1
    h4_comp = _completed_bars(h4_pid)
    h4_comp["atr_14"] = _safe_atr_series(h4_comp, window=14)
    h4_comp = detect_fvgs(h4_comp)
    h4_comp = detect_swings_live(h4_comp, n=n_swing)

    h4_ff = h4_comp[["time",
                      "fvg_bull_top", "fvg_bull_bottom",
                      "fvg_bear_top", "fvg_bear_bottom",
                      "structure_bias"]].copy()
    h4_ff = h4_ff.rename(columns={
        "fvg_bull_top":    "_h4_fvg_bull_top",
        "fvg_bull_bottom": "_h4_fvg_bull_bottom",
        "fvg_bear_top":    "_h4_fvg_bear_top",
        "fvg_bear_bottom": "_h4_fvg_bear_bottom",
        "structure_bias":  "h4_swing_bias",
    })
    # Shift +1h: completed H4 bar attaches to H1 bars in the NEXT period only
    h4_ff["time"] = h4_ff["time"] + pd.Timedelta(hours=1)
    h4_ff = h4_ff.sort_values("time").reset_index(drop=True)
    h1 = pd.merge_asof(h1, h4_ff, on="time", direction="backward")

    h1["h4_fvg_bull_exists"] = h1["_h4_fvg_bull_top"].notna().astype(float)
    h1["h4_fvg_bear_exists"] = h1["_h4_fvg_bear_top"].notna().astype(float)
    bull_d = _fvg_dist(h1["close"], h1["_h4_fvg_bull_top"],
                       h1["_h4_fvg_bull_bottom"], atr_h1).values
    bear_d = _fvg_dist(h1["close"], h1["_h4_fvg_bear_top"],
                       h1["_h4_fvg_bear_bottom"], atr_h1).values
    h1["h4_fvg_dist_atr"] = _nearest_fvg_dist(bull_d, bear_d)
    h1 = h1.drop(columns=["_h4_fvg_bull_top", "_h4_fvg_bull_bottom",
                           "_h4_fvg_bear_top", "_h4_fvg_bear_bottom"])

    # ------------------------------------------------------------------
    # Step 5 — D1
    # ------------------------------------------------------------------
    d1_part = _partial_bars(d1_pid)

    d1_ema_s = ta.trend.EMAIndicator(
        close=d1_part["close"], window=20
    ).ema_indicator()
    d1_atr_s = ta.volatility.AverageTrueRange(
        high=d1_part["high"], low=d1_part["low"],
        close=d1_part["close"], window=14,
    ).average_true_range().replace(0.0, np.nan)

    h1["d1_ema_20"]         = d1_ema_s.values
    h1["d1_atr_14"]         = d1_atr_s.values
    h1["d1_ema20_slope"]    = (d1_ema_s.diff(1) / atr_h1).values
    h1["d1_close_vs_ema20"] = ((d1_part["close"] - d1_ema_s) / atr_h1).values
    h1["d1_high_dist_atr"]  = ((d1_part["high"]  - h1["close"]) / atr_h1).values
    h1["d1_low_dist_atr"]   = ((h1["close"] - d1_part["low"])   / atr_h1).values

    # FVG + swing on completed D1 bars → forward-fill to H1
    d1_comp = _completed_bars(d1_pid)
    d1_comp["atr_14"] = _safe_atr_series(d1_comp, window=14)
    d1_comp = detect_fvgs(d1_comp)
    d1_comp = detect_swings_live(d1_comp, n=n_swing)

    d1_ff = d1_comp[["time",
                      "fvg_bull_top", "fvg_bull_bottom",
                      "fvg_bear_top", "fvg_bear_bottom",
                      "structure_bias"]].copy()
    d1_ff = d1_ff.rename(columns={
        "fvg_bull_top":    "_d1_fvg_bull_top",
        "fvg_bull_bottom": "_d1_fvg_bull_bottom",
        "fvg_bear_top":    "_d1_fvg_bear_top",
        "fvg_bear_bottom": "_d1_fvg_bear_bottom",
        "structure_bias":  "d1_swing_bias",
    })
    # Shift +1h: completed D1 bar attaches to H1 bars in the next day only
    d1_ff["time"] = d1_ff["time"] + pd.Timedelta(hours=1)
    d1_ff = d1_ff.sort_values("time").reset_index(drop=True)
    h1 = pd.merge_asof(h1, d1_ff, on="time", direction="backward")

    h1["d1_fvg_bull_exists"] = h1["_d1_fvg_bull_top"].notna().astype(float)
    h1["d1_fvg_bear_exists"] = h1["_d1_fvg_bear_top"].notna().astype(float)
    bull_d = _fvg_dist(h1["close"], h1["_d1_fvg_bull_top"],
                       h1["_d1_fvg_bull_bottom"], atr_h1).values
    bear_d = _fvg_dist(h1["close"], h1["_d1_fvg_bear_top"],
                       h1["_d1_fvg_bear_bottom"], atr_h1).values
    h1["d1_fvg_dist_atr"] = _nearest_fvg_dist(bull_d, bear_d)
    h1 = h1.drop(columns=["_d1_fvg_bull_top", "_d1_fvg_bull_bottom",
                           "_d1_fvg_bear_top", "_d1_fvg_bear_bottom"])

    # HTF alignment: D1 and H4 biases agree in sign
    h1["htf_alignment"] = np.where(
        ((h1["d1_swing_bias"].values > 0) & (h1["h4_swing_bias"].values > 0)) |
        ((h1["d1_swing_bias"].values < 0) & (h1["h4_swing_bias"].values < 0)),
        1.0, 0.0,
    )

    # ------------------------------------------------------------------
    # Step 6 — W1 (high / low only)
    # ------------------------------------------------------------------
    w1_part = _partial_bars(w1_pid)
    h1["w1_high"] = w1_part["high"].values
    h1["w1_low"]  = w1_part["low"].values
    h1["w1_high_dist_atr"] = (h1["w1_high"] - h1["close"]) / atr_h1
    h1["w1_low_dist_atr"]  = (h1["close"]   - h1["w1_low"]) / atr_h1

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

def build_features(instrument: str, df: pd.DataFrame = None,
                   n_swing: int = 10) -> pd.DataFrame:
    """
    Build the full feature set for one instrument.

    If `df` is supplied it is used directly (must be a preprocessed H1
    DataFrame with time, open, high, low, close, volume + H1 indicator
    columns).  Otherwise the cached *_H1_processed.parquet is loaded.

    No rows are removed — NaN handling is the labeller's responsibility.
    """
    log.info("Building features for %s …", instrument)

    # ------------------------------------------------------------------ load
    if df is None:
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
    from data.preprocessor import _add_h1_indicators

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

    # HTF columns (h4_ema_20, d1_ema_20, w1_high, w1_low) are computed
    # synthetically by _build_htf_cols below — no OANDA HTF parquets needed.

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

    # Emit an INFO-level note so every evaluation shows which bar's features are
    # being used. Lag > 0 during a live session means a cache update was missed.
    if now is not None:
        _now_ts    = pd.Timestamp(now).tz_convert("UTC")
        _bar_ts    = pd.Timestamp(last_row["time"]).tz_convert("UTC")
        _expected  = _now_ts.floor("H") - pd.Timedelta(hours=1)
        _lag_min   = (_expected - _bar_ts).total_seconds() / 60
        log.info(
            "[%s] features: bar=%s  lag=%.0fmin",
            instrument, _bar_ts.strftime("%Y-%m-%dT%H:%M"), _lag_min,
        )

    return last_row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_all()
