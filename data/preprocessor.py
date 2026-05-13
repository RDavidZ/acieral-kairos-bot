"""
data/preprocessor.py — Indicator computation and higher-TF forward-fill

For each instrument:
  - Computes technical indicators on H1, H4, and Daily data using `ta`
  - Forward-fills H4 and Daily indicators onto H1 rows via merge_asof
    (direction='backward' — only most recently *closed* higher-TF candle)
  - Forward-fills Weekly high/low the same way
  - Drops H1 warmup rows where atr_14 is NaN
  - Saves enriched H1 DataFrame to data/cache/{instrument}_H1_processed.parquet
"""

import logging
from pathlib import Path

import pandas as pd
import ta

from config import HARD_CONSTRAINTS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CACHE_DIR = Path(__file__).parent / "cache"
ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(instrument: str, timeframe: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{instrument}_{timeframe}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {path}. Run data/fetcher.py first."
        )
    df = pd.read_parquet(path)
    # Guarantee UTC-aware timestamps regardless of parquet storage
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _ensure_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce 'time' to UTC-aware if it isn't already."""
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")
    return df


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _mask_warmup(series: pd.Series, warmup_bars: int) -> pd.Series:
    """
    Replace the first `warmup_bars` values with NaN.

    The `ta` library fills warmup rows with 0.0 instead of NaN for ATR and
    ADX. This would corrupt forward-fills onto H1 rows. We explicitly nullify
    the warmup period using the known window sizes:
      - ATR(window):  warmup = window - 1
      - ADX(window):  warmup = 2 * window - 1
    """
    result = series.copy().astype(float)
    result.iloc[:warmup_bars] = float("nan")
    return result


def _add_h1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all H1-level indicators in place and return the DataFrame."""
    high, low, close = df["high"], df["low"], df["close"]

    df["atr_14"] = _mask_warmup(
        ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range(),
        warmup_bars=13,  # window - 1
    )

    df["ema_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

    df["rsi_5"]  = ta.momentum.RSIIndicator(close=close, window=5).rsi()
    df["rsi_10"] = ta.momentum.RSIIndicator(close=close, window=10).rsi()
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd_obj = ta.trend.MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9
    )
    df["macd_hist"] = macd_obj.macd_diff()

    df["adx_14"] = _mask_warmup(
        ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=14
        ).adx(),
        warmup_bars=27,  # 2 * window - 1
    )

    return df


def _add_h4_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute H4-level indicators; column names carry h4_ prefix."""
    high, low, close = df["high"], df["low"], df["close"]

    df["h4_atr_14"] = _mask_warmup(
        ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range(),
        warmup_bars=13,
    )

    df["h4_ema_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df["h4_rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["h4_adx_14"] = _mask_warmup(
        ta.trend.ADXIndicator(
            high=high, low=low, close=close, window=14
        ).adx(),
        warmup_bars=27,
    )

    return df


def _add_d1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Daily-level indicators; column names carry d1_ prefix."""
    high, low, close = df["high"], df["low"], df["close"]

    df["d1_atr_14"] = _mask_warmup(
        ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range(),
        warmup_bars=13,
    )

    df["d1_ema_20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()

    return df


# ---------------------------------------------------------------------------
# Higher-TF forward-fill onto H1
# ---------------------------------------------------------------------------

def _forward_fill_htf(
    h1: pd.DataFrame,
    htf: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """
    Left-join htf columns onto h1 by time using merge_asof with
    direction='backward'. This maps each H1 row to the most recently
    *closed* higher-TF candle whose close time is <= the H1 close time.

    Both DataFrames must be sorted by 'time' with UTC-aware timestamps.
    """
    htf_subset = htf[["time"] + cols].copy()

    merged = pd.merge_asof(
        h1,
        htf_subset,
        on="time",
        direction="backward",
    )
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(instrument: str) -> pd.DataFrame:
    """
    Load H1/H4/D/W parquets for `instrument`, compute indicators,
    forward-fill higher-TF columns onto H1 rows, and return the
    enriched H1 DataFrame (warmup rows dropped).
    """
    # --- Load all timeframes ---
    h1 = _load(instrument, "H1")
    h4 = _load(instrument, "H4")
    d1 = _load(instrument, "D")
    w1 = _load(instrument, "W")

    # --- H1 indicators ---
    h1 = _add_h1_indicators(h1)

    # --- H4 indicators ---
    h4 = _add_h4_indicators(h4)

    # --- Daily indicators ---
    d1 = _add_d1_indicators(d1)

    # --- Forward-fill H4 onto H1 ---
    h4_cols = ["h4_atr_14", "h4_ema_20", "h4_rsi_14", "h4_adx_14"]
    h1 = _forward_fill_htf(h1, h4, h4_cols)

    # --- Forward-fill Daily onto H1 ---
    d1_cols = ["d1_atr_14", "d1_ema_20"]
    h1 = _forward_fill_htf(h1, d1, d1_cols)

    # --- Forward-fill Weekly high/low onto H1 ---
    w1_subset = w1[["time", "high", "low"]].rename(
        columns={"high": "w1_high", "low": "w1_low"}
    )
    h1 = pd.merge_asof(
        h1,
        w1_subset,
        on="time",
        direction="backward",
    )

    # --- Drop warmup rows ---
    # Use adx_14 as the sentinel: it has the longest H1 warmup (27 bars),
    # so dropping its NaNs also covers atr_14 (13 bars) and all RSI/EMA windows.
    before = len(h1)
    h1 = h1.dropna(subset=["adx_14"]).reset_index(drop=True)
    dropped = before - len(h1)

    log.info(
        "  [%s] H1 rows: %d (dropped %d warmup)", instrument, len(h1), dropped
    )
    return h1


def preprocess_instrument(instrument: str) -> None:
    """Preprocess a single instrument — faster than preprocess_all() for hourly updates."""
    df = preprocess(instrument)
    out_path = CACHE_DIR / f"{instrument}_H1_processed.parquet"
    df.to_parquet(out_path, index=False)
    log.info("  [%s] Saved %d rows → %s", instrument, len(df), out_path.name)


def preprocess_all() -> None:
    """Preprocess all 8 instruments and save to *_H1_processed.parquet."""
    log.info("=== preprocess_all() starting ===")
    for instrument in ALL_INSTRUMENTS:
        log.info("Processing %s …", instrument)
        preprocess_instrument(instrument)
    log.info("=== preprocess_all() complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocess_all()
