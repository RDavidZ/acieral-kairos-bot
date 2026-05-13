"""
risk/news_filter.py — News-based trading restrictions

Applies ForexFactory event timing rules to entry and exit decisions.
All times are UTC. Events are sourced from data.news_fetcher (12h cache).

Rules
-----
High impact events:
  - Block new entries on affected pairs within ±30 min of the event
  - Pre-close open trades when a High impact event is 0–15 min away

Medium impact events:
  - Reduce position size to 75% within ±15 min of the event

Currency → instruments mapping covers all OANDA forex pairs traded by
both Acieral Forex Bot and Acieral Kairos Bot, plus index instruments
that are sensitive to the same macro releases.
"""

import logging
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# ── Timing windows ────────────────────────────────────────────────────────────
HIGH_BLOCK_WINDOW_MIN  = 30    # block new entries ±30 min around High events
HIGH_CLOSE_LEAD_MIN    = 15    # close open trade when High event is ≤15 min away
MEDIUM_SIZE_WINDOW_MIN = 15    # reduce size within ±15 min of Medium events
MEDIUM_SIZE_MULT       = 0.75  # position size multiplier during Medium window

# ── Currency → OANDA instruments ─────────────────────────────────────────────
# USD: all USD-quoted forex pairs plus US index instruments
# EUR: EUR-quoted forex + German index
# Other currencies: their respective forex pairs only
_CURRENCY_TO_INSTRUMENTS: dict[str, list[str]] = {
    "USD": ["EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF",
            "SPX500_USD", "NAS100_USD"],
    "EUR": ["EUR_USD", "DE30_EUR"],
    "GBP": ["GBP_USD"],
    "AUD": ["AUD_USD"],
    "JPY": ["USD_JPY"],
    "CHF": ["USD_CHF"],
    "DEU": ["DE30_EUR"],   # ForexFactory country code for Germany
}

# Reverse mapping: instrument → list of currencies that affect it
_INSTRUMENT_TO_CURRENCIES: dict[str, list[str]] = {}
for _ccy, _instrs in _CURRENCY_TO_INSTRUMENTS.items():
    for _instr in _instrs:
        _INSTRUMENT_TO_CURRENCIES.setdefault(_instr, []).append(_ccy)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def should_block_entry(instrument: str, now: datetime) -> tuple[bool, str]:
    """
    Returns (blocked, reason).

    True if a High impact event affecting this instrument falls within
    ±HIGH_BLOCK_WINDOW_MIN minutes of now. The bot should skip entry entirely.
    """
    now_utc = _to_utc(now)

    for event in _events():
        if event["impact"] != "High":
            continue
        if instrument not in _instruments_for(event["currency"]):
            continue

        diff_min = abs((event["dt_utc"] - now_utc).total_seconds()) / 60
        if diff_min <= HIGH_BLOCK_WINDOW_MIN:
            mins_to  = int((event["dt_utc"] - now_utc).total_seconds() / 60)
            sign     = "in" if mins_to >= 0 else f"{abs(mins_to)}min ago"
            reason   = (
                f"High impact '{event['title']}' ({event['currency']}) "
                f"{'in ' + str(abs(mins_to)) + 'min' if mins_to >= 0 else sign} "
                f"— entry blocked ±{HIGH_BLOCK_WINDOW_MIN}min"
            )
            log.info("%s: news block — %s", instrument, reason)
            return True, reason

    return False, ""


def should_close_pre_news(instrument: str, now: datetime) -> tuple[bool, str]:
    """
    Returns (close_now, reason).

    True if a High impact event affecting this instrument is between 0 and
    HIGH_CLOSE_LEAD_MIN minutes in the future. The bot should close the open
    trade immediately to avoid being caught in the release spike.
    """
    now_utc = _to_utc(now)

    for event in _events():
        if event["impact"] != "High":
            continue
        if instrument not in _instruments_for(event["currency"]):
            continue

        mins_to = (event["dt_utc"] - now_utc).total_seconds() / 60
        if 0 < mins_to <= HIGH_CLOSE_LEAD_MIN:
            reason = (
                f"High impact '{event['title']}' ({event['currency']}) "
                f"in {mins_to:.0f}min — pre-news close "
                f"({HIGH_CLOSE_LEAD_MIN}min early)"
            )
            log.warning("%s: pre-news close — %s", instrument, reason)
            return True, reason

    return False, ""


def get_size_multiplier(instrument: str, now: datetime) -> float:
    """
    Returns 0.75 if a Medium impact event affecting this instrument falls
    within ±MEDIUM_SIZE_WINDOW_MIN minutes of now, else 1.0.

    High impact events already block entry entirely via should_block_entry,
    so this function only needs to handle Medium events.
    """
    now_utc = _to_utc(now)

    for event in _events():
        if event["impact"] != "Medium":
            continue
        if instrument not in _instruments_for(event["currency"]):
            continue

        diff_min = abs((event["dt_utc"] - now_utc).total_seconds()) / 60
        if diff_min <= MEDIUM_SIZE_WINDOW_MIN:
            log.info(
                "%s: medium impact '%s' (%s) within %dmin — "
                "size reduced to %.0f%%",
                instrument, event["title"], event["currency"],
                MEDIUM_SIZE_WINDOW_MIN, MEDIUM_SIZE_MULT * 100,
            )
            return MEDIUM_SIZE_MULT

    return 1.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _events() -> list[dict]:
    from data.news_fetcher import get_events
    return get_events()


def _instruments_for(currency: str) -> list[str]:
    return _CURRENCY_TO_INSTRUMENTS.get(currency.upper(), [])


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_next_event(
    instrument: str,
    now: datetime,
    hours_ahead: int = 24,
) -> dict | None:
    """Return the next High/Medium event affecting instrument within hours_ahead, or None."""
    try:
        currencies = _INSTRUMENT_TO_CURRENCIES.get(instrument, [])
        if not currencies:
            return None
        cutoff = _to_utc(now) + timedelta(hours=hours_ahead)
        now_utc = _to_utc(now)
        for e in sorted(_events(), key=lambda x: x["dt_utc"]):
            if e["dt_utc"] <= now_utc:
                continue
            if e["dt_utc"] > cutoff:
                break
            if e["impact"] not in ("High", "Medium"):
                continue
            if e["currency"] in currencies:
                return e
        return None
    except Exception:
        return None


def evaluate_news_status(instrument: str, now: datetime) -> dict:
    """
    Evaluate current news status for an instrument.
    Returns a dict with all fields needed for write_news_status().
    Does not write to DB — caller is responsible for that.

    Returns:
        {
            instrument: str,
            status: 'blocked' | 'reduced' | 'watch' | 'clear',
            reason: str | None,
            size_multiplier: float,
            next_event_time: datetime | None,
            next_event_title: str | None,
            next_event_currency: str | None,
        }
    """
    # Check blocked
    blocked, reason = should_block_entry(instrument, now)
    if blocked:
        next_event = _get_next_event(instrument, now)
        return {
            "instrument":          instrument,
            "status":              "blocked",
            "reason":              reason,
            "size_multiplier":     0.0,
            "next_event_time":     next_event.get("dt_utc") if next_event else None,
            "next_event_title":    next_event.get("title") if next_event else None,
            "next_event_currency": next_event.get("currency") if next_event else None,
        }

    # Check reduced
    multiplier = get_size_multiplier(instrument, now)
    if multiplier < 1.0:
        next_event = _get_next_event(instrument, now)
        return {
            "instrument":          instrument,
            "status":              "reduced",
            "reason":              f"News proximity — size {multiplier:.0%}",
            "size_multiplier":     multiplier,
            "next_event_time":     next_event.get("dt_utc") if next_event else None,
            "next_event_title":    next_event.get("title") if next_event else None,
            "next_event_currency": next_event.get("currency") if next_event else None,
        }

    # Check watch — next high-impact event within 2 hours
    next_event = _get_next_event(instrument, now, hours_ahead=2)
    if next_event:
        return {
            "instrument":          instrument,
            "status":              "watch",
            "reason":              None,
            "size_multiplier":     1.0,
            "next_event_time":     next_event.get("dt_utc"),
            "next_event_title":    next_event.get("title"),
            "next_event_currency": next_event.get("currency"),
        }

    # Clear — find next event regardless of window for display
    next_event = _get_next_event(instrument, now, hours_ahead=24)
    return {
        "instrument":          instrument,
        "status":              "clear",
        "reason":              None,
        "size_multiplier":     1.0,
        "next_event_time":     next_event.get("dt_utc") if next_event else None,
        "next_event_title":    next_event.get("title") if next_event else None,
        "next_event_currency": next_event.get("currency") if next_event else None,
    }
