"""
notifications/telegram_bot.py — Telegram alert layer for Acieral Kairos Bot

All methods are fire-and-forget: failures are logged but never raise.
If TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set, all send methods
are no-ops (bot.enabled == False).

Uses HTML parse_mode — format labels with <b>bold</b>.
"""

import logging
import os
from datetime import datetime

import requests

from config import HARD_CONSTRAINTS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HC              = HARD_CONSTRAINTS
FOREX_PAIRS      = _HC["FOREX_PAIRS"]
INDEX_INSTRUMENTS = _HC["INDEX_INSTRUMENTS"]

# Pip size per instrument (only used for forex display)
_PIP_SIZE: dict[str, float] = {
    "EUR_USD": 0.0001,
    "GBP_USD": 0.0001,
    "AUD_USD": 0.0001,
    "USD_JPY": 0.01,
    "USD_CHF": 0.0001,
}

# Display names for daily summary (indices only — forex shown as-is)
_DISPLAY: dict[str, str] = {
    "SPX500_USD": "SPX500",
    "NAS100_USD": "NAS100",
    "DE30_EUR":   "DE30",
}

# Human-readable exit reason labels
_REASON_LABEL: dict[str, str] = {
    "EXIT_MODEL":   "Exit model signal",
    "TRAIL_SL":     "Trailing stop",
    "SL":           "Stop loss",
    "EOD":          "End of day",
    "FRIDAY_CLOSE": "Friday close",
    "MAX_BARS":     "Max hold duration",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display_name(instrument: str) -> str:
    return _DISPLAY.get(instrument, instrument)


def _price_dp(instrument: str) -> int:
    if "JPY" in instrument:
        return 3
    if instrument in INDEX_INSTRUMENTS:
        return 2
    return 5


def _fmt_price(price: float, instrument: str) -> str:
    return f"{price:.{_price_dp(instrument)}f}"


def _sl_pips(sl_distance: float, instrument: str) -> float:
    """Convert SL distance to pips (forex) or points (indices)."""
    if instrument in INDEX_INSTRUMENTS:
        return sl_distance                          # points — no pip conversion
    pip = _PIP_SIZE.get(instrument, 0.0001)
    return sl_distance / pip


def _atr_to_pips(atr_units: float, atr: float, instrument: str) -> float:
    """Convert ATR-normalised units back to pips (forex) or points (indices)."""
    raw = atr_units * atr
    if instrument in INDEX_INSTRUMENTS:
        return raw
    pip = _PIP_SIZE.get(instrument, 0.0001)
    return raw / pip


def _to_lot_size(instrument: str, units: float) -> str:
    """Convert OANDA units to prop firm lot sizes for reference."""
    INDEX_INSTRUMENTS_SET = {"SPX500_USD", "NAS100_USD", "DE30_EUR"}
    if instrument in INDEX_INSTRUMENTS_SET:
        return f"{abs(units):.1f} contracts"
    else:
        std_lots   = abs(units) / 100000
        mini_lots  = abs(units) / 10000
        micro_lots = abs(units) / 1000
        if std_lots >= 0.01:
            return f"{std_lots:.2f} lots"
        elif mini_lots >= 0.01:
            return f"{mini_lots:.2f} mini lots"
        else:
            return f"{micro_lots:.2f} micro lots"


def _hold_str(bars_held: int) -> str:
    """Format bars_held (each bar = 1H) as 'Xh 00m' or '0m'."""
    if bars_held < 1:
        return "0m"
    return f"{bars_held}h 00m"


def _pnl_str(pnl_gbp: float) -> str:
    """Format P&L as '+£34.90' or '-£12.30'."""
    sign = "+" if pnl_gbp >= 0 else "-"
    return f"{sign}£{abs(pnl_gbp):.2f}"


# ---------------------------------------------------------------------------
# Prop-firm alert helpers (module-level for external verification)
# ---------------------------------------------------------------------------

_INDEX_SET = {"SPX500_USD", "NAS100_USD", "DE30_EUR"}


def _pip_value(instrument: str) -> float:
    """Return pip size for instrument."""
    if "JPY" in instrument:
        return 0.01
    elif instrument in _INDEX_SET:
        return 1.0          # points, not pips
    else:
        return 0.0001


def _sl_distance_pips(instrument: str, entry: float, sl: float) -> float:
    """Return SL distance in pips (forex) or points (indices), rounded to 1dp."""
    return round(abs(entry - sl) / _pip_value(instrument), 1)


def _entry_zone(instrument: str, entry: float, direction: str) -> tuple:
    """Return (low, high) entry zone — 5 pips either side for forex, 2 pts for indices."""
    tolerance = 2.0 if instrument in _INDEX_SET else 5 * _pip_value(instrument)
    return (round(entry - tolerance, 5), round(entry + tolerance, 5))


def _confidence_stars(confidence: float) -> str:
    """Return star rating based on confidence."""
    if confidence >= 0.90:
        return "⭐⭐⭐⭐⭐"
    elif confidence >= 0.80:
        return "⭐⭐⭐⭐"
    elif confidence >= 0.70:
        return "⭐⭐⭐"
    elif confidence >= 0.60:
        return "⭐⭐"
    else:
        return "⭐"


def _go_no_go(confidence: float, instrument: str) -> str:
    """Return go/no-go recommendation. GO if confidence >= 0.70, SKIP if below."""
    if confidence < 0.70:
        return "⛔ SKIP — Confidence below threshold"
    tolerance = "2 pts" if instrument in _INDEX_SET else "5 pips"
    return f"⚡ GO — Enter within {tolerance} of entry"


def _format_entry_zone(instrument: str, low: float, high: float) -> str:
    if instrument in _INDEX_SET:
        return f"{low:.1f} – {high:.1f}"
    elif "JPY" in instrument:
        return f"{low:.3f} – {high:.3f}"
    else:
        return f"{low:.5f} – {high:.5f}"


def _format_price(instrument: str, price: float) -> str:
    if instrument in _INDEX_SET:
        return f"{price:.1f}"
    elif "JPY" in instrument:
        return f"{price:.3f}"
    else:
        return f"{price:.5f}"


# ---------------------------------------------------------------------------
# Message builders (module-level for external verification)
# ---------------------------------------------------------------------------

def _build_trade_opened_message(
    instrument: str,
    direction: str,
    confidence: float,
    entry_price: float,
    sl_price: float,
    units: float,
    risk_gbp: float,
    candle_time,
) -> str:
    display    = instrument.replace("_", "")
    dir_emoji  = "📈" if direction == "LONG" else "📉"
    stars      = _confidence_stars(confidence)
    sl_pips    = _sl_distance_pips(instrument, entry_price, sl_price)
    pip_label  = "pts" if instrument in _INDEX_SET else "pips"
    zone_low, zone_high = _entry_zone(instrument, entry_price, direction)
    zone_str   = _format_entry_zone(instrument, zone_low, zone_high)
    sl_str     = _format_price(instrument, sl_price)
    lot_str    = _to_lot_size(instrument, units)
    go_str     = _go_no_go(confidence, instrument)
    time_str   = candle_time.strftime("%H:%M UTC") if hasattr(candle_time, "strftime") else str(candle_time)

    if instrument in _INDEX_SET:
        size_line  = f"Prop size: {lot_str} | Risk £{risk_gbp:.0f}"
        oanda_line = f"OANDA: {abs(int(units))} units"
    else:
        size_line  = f"FTMO 10k: {lot_str} | Risk £{risk_gbp:.0f}"
        oanda_line = f"OANDA: {int(abs(units)):,} units"

    return (
        f"✅ <b>SIGNAL | {display}</b>\n\n"
        f"Direction: {direction} {dir_emoji}  Confidence: {confidence * 100:.0f}% {stars}\n\n"
        f"Entry zone: {zone_str}\n"
        f"Stop loss: {sl_str} ({sl_pips:.1f} {pip_label})\n\n"
        f"{size_line}\n"
        f"{oanda_line}\n\n"
        f"{go_str}\n"
        f"🕐 {time_str} | Kairos v1"
    )


def _build_trade_closed_message(
    instrument: str,
    direction: str,
    close_price: float,
    pnl_gbp: float,
    exit_reason: str,
    bars_held: int,
    entry_price: float,
) -> str:
    display    = instrument.replace("_", "")
    pips       = _sl_distance_pips(instrument, entry_price, close_price)
    pip_label  = "pts" if instrument in _INDEX_SET else "pips"

    if pnl_gbp >= 0:
        result_line = f"✅ WIN  +£{pnl_gbp:.2f} (+{pips:.1f} {pip_label})"
    else:
        result_line = f"❌ LOSS  -£{abs(pnl_gbp):.2f} (-{pips:.1f} {pip_label})"

    hold_hours = bars_held // 4 if bars_held else 0
    hold_mins  = (bars_held % 4) * 15 if bars_held else 0
    hold_str   = f"{hold_hours}h {hold_mins:02d}m" if hold_hours > 0 else f"{hold_mins}m"

    reason_str = _REASON_LABEL.get(exit_reason, exit_reason)

    return (
        f"🔔 <b>CLOSED | {display}</b>\n\n"
        f"{result_line}\n\n"
        f"Exit: {_format_price(instrument, close_price)} | Reason: {reason_str}\n"
        f"Held: {hold_str} ({bars_held} bars)"
    )


# ---------------------------------------------------------------------------
# TelegramBot
# ---------------------------------------------------------------------------

class TelegramBot:
    """
    Telegram alert client for Acieral Kairos Bot.

    Credentials loaded from .env (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID).
    If either is absent, all send methods are silent no-ops.

    Uses HTML parse_mode — use <b>bold</b> for labels.
    """

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

        self.token    = token
        self.chat_id  = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled  = bool(token and chat_id)

        if not self.enabled:
            log.warning(
                "TelegramBot: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
                "all alerts disabled"
            )
        else:
            log.info("TelegramBot ready | chat_id=%s", chat_id)

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, message: str) -> bool:
        """
        POST a message to Telegram.

        Retries once on failure. Returns True on success, False on failure.
        Never raises.
        """
        if not self.enabled:
            return False

        payload = {
            "chat_id":    self.chat_id,
            "text":       message,
            "parse_mode": "HTML",
        }

        for attempt in range(1, 3):   # up to 2 attempts
            try:
                resp = requests.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=10,
                )
                if resp.status_code == 200:
                    return True
                log.warning(
                    "Telegram send failed (attempt %d/2): HTTP %d — %s",
                    attempt, resp.status_code, resp.text[:200],
                )
            except requests.RequestException as exc:
                log.warning("Telegram send error (attempt %d/2): %s", attempt, exc)

        log.error("Telegram: all retries exhausted — message not delivered")
        return False

    # ------------------------------------------------------------------
    # Trade opened
    # ------------------------------------------------------------------

    def send_trade_opened(
        self,
        instrument: str,
        direction: str,
        confidence: float,
        entry_price: float,
        sl_price: float,
        units: int,
        risk_gbp: float,
        now: datetime,
    ) -> None:
        msg = _build_trade_opened_message(
            instrument, direction, confidence, entry_price,
            sl_price, units, risk_gbp, now,
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Trade closed
    # ------------------------------------------------------------------

    def send_trade_closed(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        close_price: float,
        pnl_gbp: float,
        exit_reason: str,
        bars_held: int,
        mfe_atr: float,
        mae_atr: float,
        atr: float,
        now: datetime,
    ) -> None:
        msg = _build_trade_closed_message(
            instrument, direction, close_price, pnl_gbp,
            exit_reason, bars_held, entry_price,
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Day skipped
    # ------------------------------------------------------------------

    def send_skipped(
        self,
        instrument: str,
        best_confidence: float,
        threshold: float,
        reason: str | None = None,
    ) -> None:
        msg = (
            f"⏭ <b>SKIPPED</b> | {instrument}\n"
            f"Best confidence: {best_confidence:.2f} | Threshold: {threshold:.2f}"
        )
        if reason:
            msg += f"\n{reason}"
        self.send(msg)

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    def send_daily_summary(
        self,
        date_str: str,
        results: dict,
        equity: float,
        day_pnl: float,
        next_retrain_days: int,
    ) -> None:
        """
        results: {instrument: {'status': 'win'/'loss'/'skip', 'pnl': float}}
        """
        def _inst_str(instrument: str) -> str:
            r     = results.get(instrument, {})
            st    = r.get("status", "skip")
            pnl   = float(r.get("pnl", 0.0))
            name  = _display_name(instrument)
            if st == "win":
                return f"{name} ✅ +£{pnl:.2f}"
            elif st == "loss":
                return f"{name} ❌ -£{abs(pnl):.2f}"
            else:
                return f"{name} ⏭"

        forex_parts = " | ".join(_inst_str(i) for i in FOREX_PAIRS)
        index_parts = " | ".join(_inst_str(i) for i in INDEX_INSTRUMENTS)

        # Drawdown %
        day_start = equity - day_pnl
        dd_pct    = (day_pnl / day_start * 100) if day_start > 0 else 0.0
        dd_icon   = "✅" if abs(dd_pct) < 1.0 else "⚠️"
        dd_str    = f"{dd_pct:+.1f}%"

        msg = (
            f"📊 <b>DAILY SUMMARY</b> | {date_str}\n"
            f"Forex:   {forex_parts}\n"
            f"Indices: {index_parts}\n"
            f"P&L: {_pnl_str(day_pnl)} | Equity: £{equity:.2f} | "
            f"DD: {dd_str} {dd_icon}\n"
            f"Next retrain in: {next_retrain_days} trading days"
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def send_kill_switch(self, reason: str, equity: float) -> None:
        msg = (
            f"🚨 <b>KILL SWITCH ACTIVATED</b>\n"
            f"Reason: {reason}\n"
            f"Equity: £{equity:.2f}\n"
            f"All trading suspended for today."
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Trail activated
    # ------------------------------------------------------------------

    def send_trail_activated(
        self,
        instrument: str,
        direction: str,
        entry_price: float,
        current_price: float,
        unrealised_pnl: float,
        new_sl: float,
    ) -> None:
        display = instrument.replace("_", "")
        dir_emoji = "📈" if direction == "LONG" else "📉"
        msg = (
            f"🔒 <b>TRAIL ACTIVATED</b> | {display}\n\n"
            f"Direction: {direction} {dir_emoji}\n"
            f"Unrealised P&L: +£{unrealised_pnl:.2f}\n"
            f"Entry: {_format_price(instrument, entry_price)} → Now: {_format_price(instrument, current_price)}\n"
            f"Trail SL locked at: {_format_price(instrument, new_sl)}\n\n"
            f"Position protected — letting it run 🚀"
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Unrealised P&L milestone
    # ------------------------------------------------------------------

    def send_unrealised_milestone(
        self,
        instrument: str,
        direction: str,
        unrealised_pnl: float,
        milestone: float,
        entry_price: float,
        current_price: float,
    ) -> None:
        display = instrument.replace("_", "")
        dir_emoji = "📈" if direction == "LONG" else "📉"
        msg = (
            f"💰 <b>UNREALISED MILESTONE</b> | {display}\n\n"
            f"Direction: {direction} {dir_emoji}\n"
            f"Unrealised P&L: +£{unrealised_pnl:.2f}\n"
            f"Entry: {_format_price(instrument, entry_price)} → Now: {_format_price(instrument, current_price)}\n\n"
            f"🏆 £{milestone:.0f} milestone reached"
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Retrain complete
    # ------------------------------------------------------------------

    def send_retrain_complete(self, results: dict) -> None:
        """
        results: {instrument: {'mean_auc': float, 'n_folds': int}}
        """
        lines = "\n".join(
            f"{inst}: AUC {v['mean_auc']:.4f} ({v['n_folds']} folds)"
            for inst, v in sorted(results.items())
        )
        msg = (
            f"🔄 <b>RETRAIN COMPLETE</b>\n"
            f"{lines}\n"
            f"Next retrain in 20 trading days."
        )
        self.send(msg)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def send_startup(
        self,
        equity: float,
        active_instruments: list,
        environment: str,
    ) -> None:
        msg = (
            f"🚀 <b>ACIERAL KAIROS BOT STARTED</b>\n"
            f"Environment: {environment}\n"
            f"Equity: £{equity:.2f}\n"
            f"Active: {', '.join(active_instruments)}\n"
            f"Monitoring {len(active_instruments)} instruments on 1H candles."
        )
        self.send(msg)
