"""
data/fetcher.py — OANDA candle fetcher for Acieral Kairos Bot

Fetches H1, H4, D, W candles for all 8 instruments (5 forex + 3 indices).
Saves to data/cache/{instrument}_{timeframe}.parquet with incremental updates.

OANDA granularity mapping:
  H1 → "H1", H4 → "H4", D → "D", W → "W"
"""

import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

from config import HARD_CONSTRAINTS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "cache"
FETCH_FROM = "2017-01-01T00:00:00Z"
TIMEFRAMES = ["H1", "H4", "D", "W"]
MAX_CANDLES_PER_REQUEST = 5000
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2.0  # seconds; doubled on each retry

# JPY instruments use 3dp, all others 5dp
JPY_INSTRUMENTS = {"USD_JPY"}

FOREX_PAIRS = HARD_CONSTRAINTS["FOREX_PAIRS"]
INDEX_INSTRUMENTS = HARD_CONSTRAINTS["INDEX_INSTRUMENTS"]
ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_price(value: float, instrument: str) -> float:
    dp = 3 if instrument in JPY_INSTRUMENTS else 5
    return round(value, dp)


def _get_client() -> oandapyV20.API:
    api_key = os.environ.get("OANDA_API_KEY")
    environment = os.environ.get("OANDA_ENVIRONMENT", "practice")
    if not api_key:
        raise EnvironmentError("OANDA_API_KEY not set in environment / .env")
    return oandapyV20.API(access_token=api_key, environment=environment)


def _cache_path(instrument: str, timeframe: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{instrument}_{timeframe}.parquet"


def _load_cache(instrument: str, timeframe: str) -> pd.DataFrame | None:
    path = _cache_path(instrument, timeframe)
    if path.exists():
        return pd.read_parquet(path)
    return None


def _save_cache(df: pd.DataFrame, instrument: str, timeframe: str) -> None:
    path = _cache_path(instrument, timeframe)
    df.to_parquet(path, index=False)


def _last_cached_time(df: pd.DataFrame) -> str:
    """Return ISO-8601 UTC string of the latest cached candle close time."""
    ts = pd.to_datetime(df["time"]).max()
    # Advance by 1 second so we don't re-fetch the last candle
    ts = ts + pd.Timedelta(seconds=1)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_candles(raw_candles: list, instrument: str) -> pd.DataFrame:
    """Convert raw OANDA candle dicts to a tidy DataFrame."""
    rows = []
    for c in raw_candles:
        if not c.get("complete", False):
            # Skip the in-progress candle
            continue
        mid = c.get("mid", {})
        rows.append({
            "time":   c["time"],
            "open":   _round_price(float(mid.get("o", 0)), instrument),
            "high":   _round_price(float(mid.get("h", 0)), instrument),
            "low":    _round_price(float(mid.get("l", 0)), instrument),
            "close":  _round_price(float(mid.get("c", 0)), instrument),
            "volume": int(c.get("volume", 0)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _fetch_candles_paginated(
    client: oandapyV20.API,
    instrument: str,
    granularity: str,
    from_time: str,
) -> pd.DataFrame:
    """
    Fetch all complete candles from from_time to now, paginating in
    MAX_CANDLES_PER_REQUEST chunks. Returns a deduplicated, sorted DataFrame.
    """
    all_frames: list[pd.DataFrame] = []
    current_from = from_time

    while True:
        params = {
            "granularity": granularity,
            "from": current_from,
            "count": MAX_CANDLES_PER_REQUEST,
            "price": "M",  # mid prices
        }

        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                req = instruments.InstrumentsCandles(
                    instrument=instrument, params=params
                )
                client.request(req)
                raw = req.response.get("candles", [])
                break
            except Exception as exc:
                if attempt == RETRY_ATTEMPTS:
                    log.error(
                        "Failed to fetch %s %s after %d attempts: %s",
                        instrument, granularity, RETRY_ATTEMPTS, exc,
                    )
                    raise
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Attempt %d/%d failed for %s %s (%s). Retrying in %.1fs…",
                    attempt, RETRY_ATTEMPTS, instrument, granularity, exc, delay,
                )
                time.sleep(delay)

        df_page = _parse_candles(raw, instrument)

        if df_page.empty:
            break

        all_frames.append(df_page)

        # If we got fewer than MAX_CANDLES_PER_REQUEST complete candles, we're done
        if len(raw) < MAX_CANDLES_PER_REQUEST:
            break

        # Advance from_time to just after the last candle in this page
        last_ts = df_page["time"].max()
        current_from = (last_ts + pd.Timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    if not all_frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Per-instrument fetch
# ---------------------------------------------------------------------------

def _fetch_instrument(
    client: oandapyV20.API,
    instrument: str,
    timeframe: str,
) -> None:
    """Fetch (or incrementally update) one instrument+timeframe and save to cache."""
    cached = _load_cache(instrument, timeframe)

    if cached is not None and not cached.empty:
        from_time = _last_cached_time(cached)
        log.info(
            "  [%s %s] Incremental fetch from %s (cached rows: %d)",
            instrument, timeframe, from_time, len(cached),
        )
    else:
        from_time = FETCH_FROM
        log.info(
            "  [%s %s] Full fetch from %s",
            instrument, timeframe, from_time,
        )

    new_df = _fetch_candles_paginated(client, instrument, timeframe, from_time)

    if new_df.empty:
        log.info("  [%s %s] No new candles.", instrument, timeframe)
        return

    if cached is not None and not cached.empty:
        # Ensure cached times are datetime[UTC] before concat
        cached["time"] = pd.to_datetime(cached["time"], utc=True)
        combined = pd.concat([cached, new_df], ignore_index=True)
        combined = (
            combined
            .drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )
    else:
        combined = new_df

    _save_cache(combined, instrument, timeframe)
    log.info(
        "  [%s %s] Saved %d total rows (+%d new).",
        instrument, timeframe, len(combined), len(new_df),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_forex(client: oandapyV20.API | None = None) -> None:
    """Fetch H1/H4/D/W candles for all 5 forex pairs."""
    if client is None:
        client = _get_client()
    log.info("=== Fetching forex pairs ===")
    for pair in FOREX_PAIRS:
        for tf in TIMEFRAMES:
            _fetch_instrument(client, pair, tf)


def fetch_indices(client: oandapyV20.API | None = None) -> None:
    """
    Fetch H1/H4/D/W candles for all 3 equity index instruments.

    Indices have session gaps (weekends + overnight), but OANDA handles this
    transparently — we just request candles and store what is returned.
    The session-gap awareness lives in feature engineering, not here.
    """
    if client is None:
        client = _get_client()
    log.info("=== Fetching equity indices ===")
    for idx in INDEX_INSTRUMENTS:
        for tf in TIMEFRAMES:
            _fetch_instrument(client, idx, tf)


def fetch_all() -> None:
    """Fetch all instruments (forex + indices) across all timeframes."""
    client = _get_client()
    fetch_forex(client)
    fetch_indices(client)
    log.info("=== fetch_all() complete ===")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetch_all()
