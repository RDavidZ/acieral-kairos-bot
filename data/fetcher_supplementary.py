"""
data/fetcher_supplementary.py — VIX and SPY volume via yfinance

Fetches:
  - VIX daily close (^VIX) from 2017-01-01 to today
  - SPY daily volume from 2017-01-01 to today

Saves:
  - data/cache/vix_daily.parquet   — columns: date, vix_close
  - data/cache/spy_volume_daily.parquet — columns: date, spy_volume

These datasets will be merged onto index instrument features only
(SPX500_USD, NAS100_USD) in the feature builder, keyed on trading date.
DE30_EUR uses a different session and does not use these US-market
supplementary features.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CACHE_DIR = Path(__file__).parent / "cache"
FETCH_FROM = "2017-01-01"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str() -> str:
    return date.today().isoformat()


def _fetch_vix() -> pd.DataFrame:
    """Fetch ^VIX daily close from 2017-01-01 to today."""
    log.info("Fetching VIX (^VIX) daily close from %s …", FETCH_FROM)
    raw = yf.download(
        "^VIX",
        start=FETCH_FROM,
        end=_today_str(),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame for ^VIX")

    # yfinance returns a MultiIndex or flat index depending on version
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.reset_index()
    df.columns = ["date", "vix_close"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["vix_close"]).reset_index(drop=True)
    df["vix_close"] = df["vix_close"].astype(float).round(2)
    log.info("  VIX: %d rows (%s → %s)", len(df), df["date"].min(), df["date"].max())
    return df


def _fetch_spy_volume() -> pd.DataFrame:
    """Fetch SPY daily volume from 2017-01-01 to today."""
    log.info("Fetching SPY daily volume from %s …", FETCH_FROM)
    raw = yf.download(
        "SPY",
        start=FETCH_FROM,
        end=_today_str(),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame for SPY")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.reset_index()
    df.columns = ["date", "spy_volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.dropna(subset=["spy_volume"]).reset_index(drop=True)
    df["spy_volume"] = df["spy_volume"].astype("int64")
    log.info("  SPY volume: %d rows (%s → %s)", len(df), df["date"].min(), df["date"].max())
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_supplementary() -> None:
    """Fetch VIX close and SPY volume; save both to data/cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    vix_df = _fetch_vix()
    vix_path = CACHE_DIR / "vix_daily.parquet"
    vix_df.to_parquet(vix_path, index=False)
    log.info("  Saved → %s", vix_path.name)

    spy_df = _fetch_spy_volume()
    spy_path = CACHE_DIR / "spy_volume_daily.parquet"
    spy_df.to_parquet(spy_path, index=False)
    log.info("  Saved → %s", spy_path.name)

    log.info("=== fetch_supplementary() complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetch_supplementary()
