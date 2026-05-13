"""
data/news_fetcher.py — ForexFactory news calendar fetcher

Fetches the ForexFactory calendar via the nfs.faireconomy.media JSON mirror.
Pulls both the current week and next week to handle events near the boundary.

Cached to data/cache/news_cache.json and refreshed every 12 hours.
On fetch failure the stale cache is used as a fallback (log warning only).

Event dict keys:
  title    : str       — e.g. "Non-Farm Payrolls"
  currency : str       — e.g. "USD"
  dt_utc   : datetime  — tz-aware UTC datetime of the event
  impact   : str       — "High" or "Medium"  (Low events are dropped)
"""

import json
import logging
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_FILE      = Path(__file__).parent / "cache" / "news_cache.json"
_CACHE_TTL_HOURS = 12

_FEED_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
]

_KEEP_IMPACTS = {"High", "Medium"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_events() -> list[dict]:
    """
    Return High and Medium impact news events (both current and next week).
    Refreshes from ForexFactory if cache is older than 12 hours.
    Falls back to stale cache on network failure.

    Returns list of dicts: {title, currency, dt_utc (UTC-aware), impact}.
    """
    if _cache_is_fresh():
        return _load_cache()

    try:
        events = _fetch_all()
        _save_cache(events)
        log.info("News cache refreshed — %d High/Medium events", len(events))
        return events
    except Exception as exc:
        log.warning("News fetch failed (%s) — using cached data", exc)
        cached = _load_cache()
        if cached:
            return cached
        log.warning("News cache empty — news filter disabled this cycle")
        return []


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_is_fresh() -> bool:
    if not _CACHE_FILE.exists():
        return False
    age_s = (
        datetime.now(timezone.utc)
        - datetime.fromtimestamp(_CACHE_FILE.stat().st_mtime, tz=timezone.utc)
    ).total_seconds()
    return age_s < _CACHE_TTL_HOURS * 3600


def _load_cache() -> list[dict]:
    if not _CACHE_FILE.exists():
        return []
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        return [_deserialise(e) for e in raw]
    except Exception as exc:
        log.warning("News cache read failed: %s", exc)
        return []


def _save_cache(events: list[dict]) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([_serialise(e) for e in events], f, indent=2)


def _serialise(e: dict) -> dict:
    return {**e, "dt_utc": e["dt_utc"].isoformat()}


def _deserialise(e: dict) -> dict:
    dt = datetime.fromisoformat(e["dt_utc"])
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return {**e, "dt_utc": dt}


# ---------------------------------------------------------------------------
# Feed fetch and parse
# ---------------------------------------------------------------------------

def _fetch_all() -> list[dict]:
    events: list[dict] = []
    for url in _FEED_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AcieralBot/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            events.extend(_parse(data))
            log.debug("Fetched %d events from %s", len(data), url)
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", url, exc)

    # Deduplicate by (currency, dt_utc, title)
    seen:   set        = set()
    unique: list[dict] = []
    for e in events:
        key = (e["currency"], e["dt_utc"].isoformat(), e["title"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def write_events_to_db(bot_id: str) -> int:
    """
    Write current cached news events to the dashboard DB.
    Called after each cache refresh so the dashboard can display the feed
    without importing from this module.
    Returns count of events written, or 0 on failure.
    """
    try:
        from dashboard.db import write_news_events
        from risk.news_filter import _CURRENCY_TO_INSTRUMENTS
    except ImportError:
        return 0

    try:
        events = get_events()
        if not events:
            return 0

        db_events = []
        for e in events:
            affected = _CURRENCY_TO_INSTRUMENTS.get(e["currency"], [])
            db_events.append({
                "title":       e["title"],
                "currency":    e["currency"],
                "impact":      e["impact"],
                "event_time":  e["dt_utc"],
                "instruments": affected,
            })

        return write_news_events(bot_id, db_events)
    except Exception as exc:
        log.warning(f"write_events_to_db failed: {exc}")
        return 0


def _parse(data: list[dict]) -> list[dict]:
    events = []
    for item in data:
        impact = item.get("impact", "")
        if impact not in _KEEP_IMPACTS:
            continue

        raw_dt = item.get("date", "")
        if not raw_dt:
            continue
        try:
            dt = datetime.fromisoformat(raw_dt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            continue

        # ForexFactory uses "country" for the currency code
        currency = (item.get("country") or item.get("currency") or "").upper().strip()
        if not currency:
            continue

        events.append({
            "title":    item.get("title", ""),
            "currency": currency,
            "dt_utc":   dt,
            "impact":   impact,
        })
    return events
