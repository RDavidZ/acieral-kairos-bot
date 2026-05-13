"""
risk/manager.py — Stateful risk enforcement for Acieral Kairos Bot

Sits above both entry and exit models.  All hard constraints from
HARD_CONSTRAINTS are enforced here regardless of model output.

Dependency order: data/ → strategy/ → ml/ → backtest/ → risk/ → execution/
"""

import logging
import threading
from datetime import datetime, timedelta

from config import HARD_CONSTRAINTS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants pulled from HARD_CONSTRAINTS
# ---------------------------------------------------------------------------

_HC = HARD_CONSTRAINTS

FOREX_PAIRS       = _HC["FOREX_PAIRS"]
INDEX_INSTRUMENTS = _HC["INDEX_INSTRUMENTS"]

MAX_OPEN_FOREX    = _HC["MAX_OPEN_TRADES_FOREX"]
MAX_OPEN_INDICES  = _HC["MAX_OPEN_TRADES_INDICES"]
MAX_OPEN_TOTAL    = _HC["MAX_OPEN_TRADES_TOTAL"]
DD_KILL_PCT       = _HC["DAILY_DRAWDOWN_KILL_PCT"]
COOLDOWN_HOURS    = _HC["REENTRY_COOLDOWN_HOURS"]
CONF_FLOOR        = _HC["REENTRY_CONFIDENCE_FLOOR"]
MAX_PER_DAY       = _HC["MAX_TRADES_PER_INSTRUMENT_PER_DAY"]
QUOTE_TYPE        = _HC["INSTRUMENT_QUOTE_TYPE"]
RISK_PCT          = _HC["RISK_PER_TRADE"]
MAX_RISK_GBP      = _HC["MAX_RISK_PER_TRADE_GBP"]
RISK_TIERS        = _HC["RISK_TIERS"]
MAX_UNITS         = _HC["MAX_UNITS_PER_TRADE"]


class RiskManager:
    """
    Stateful risk enforcement layer for the Acieral Kairos Bot.

    Instantiate once at bot startup with the current account equity.
    Call ``can_trade`` before every entry signal.
    Call ``register_open_trade`` / ``register_close_trade`` on every
    trade lifecycle event.
    Call ``daily_reset`` at UTC midnight.

    All hard constraints from HARD_CONSTRAINTS are enforced.
    Neither entry nor exit model output can override them.
    """

    def __init__(self, initial_equity: float, bot_id: str = "acieral_kairos_v1") -> None:
        self.equity             = initial_equity
        self.day_start_equity   = initial_equity
        self.bot_id             = bot_id
        self.kill_switch        = False
        self.kill_switch_reason: str | None = None
        self.drawdown_floor:    float       = initial_equity * (1.0 - DD_KILL_PCT)

        # Optional callback — fired once on kill switch activation (False → True).
        # Assign after construction to avoid circular imports (e.g. telegram).
        self.on_kill_switch: callable | None = None

        # Thread safety — streaming thread and scheduler thread both access trade state
        self._lock = threading.RLock()

        # Per-instrument state
        self._open_trades:      dict[str, dict]     = {}
        self._traded_today:     set[str]             = set()
        self._last_close_time:  dict[str, datetime] = {}
        self._pair_trade_count: dict[str, int]      = {}

        # Cross-instrument state
        self._open_forex:   int   = 0
        self._open_indices: int   = 0
        self._day_pnl:      float = 0.0

    # ------------------------------------------------------------------
    # Primary gate — call before every entry
    # ------------------------------------------------------------------

    def can_trade(
        self,
        instrument: str,
        confidence: float,
        now: datetime,
    ) -> tuple[bool, str]:
        """
        Check all hard constraints in order (fail fast).

        Parameters
        ----------
        instrument  : OANDA instrument name
        confidence  : entry model confidence for this signal (0–1)
        now         : current UTC datetime (tz-aware)

        Returns
        -------
        (True, '')          — trade permitted
        (False, reason_str) — trade blocked; reason_str describes why
        """

        # 1. Kill switch active
        if self.kill_switch:
            return False, f"Kill switch active: {self.kill_switch_reason}"

        # 2. Instrument already has an open trade
        if instrument in self._open_trades:
            return False, "Open trade exists"

        # 3. Instrument already traded today
        if instrument in self._traded_today:
            return False, "Already traded today"

        # 4. Per-instrument daily trade count
        if self._pair_trade_count.get(instrument, 0) >= MAX_PER_DAY:
            return False, "Daily trade limit reached"

        # 5. Re-entry cooldown + low confidence
        #    Cooldown alone does not block — only cooldown AND low confidence.
        if instrument in self._last_close_time:
            elapsed = now - self._last_close_time[instrument]
            if elapsed < timedelta(hours=COOLDOWN_HOURS) and confidence < CONF_FLOOR:
                return False, "Cooldown: low confidence"

        # 6. Forex concurrent open-trade limit
        if instrument in FOREX_PAIRS:
            if self._open_forex >= MAX_OPEN_FOREX:
                return False, "Forex limit reached"

        # 7. Index concurrent open-trade limit
        if instrument in INDEX_INSTRUMENTS:
            if self._open_indices >= MAX_OPEN_INDICES:
                return False, "Index limit reached"

        # 8. Total simultaneous open-trade limit
        if (self._open_forex + self._open_indices) >= MAX_OPEN_TOTAL:
            return False, "Total limit reached"

        # 9. Forex weekend close — no new entries Friday ≥ 21:00 UTC
        if instrument in FOREX_PAIRS:
            if now.weekday() == 4 and now.hour >= 21:
                return False, "Forex weekend close"

        # 10. All clear
        return True, ""

    # ------------------------------------------------------------------
    # Trade lifecycle
    # ------------------------------------------------------------------

    def register_open_trade(
        self,
        instrument: str,
        trade: dict,
        now: datetime,
    ) -> None:
        """
        Record a newly opened trade.

        Parameters
        ----------
        instrument : OANDA instrument name
        trade      : dict with keys: entry_price, direction, sl_price,
                     units, atr_at_entry, entry_time, confidence
        now        : current UTC datetime (tz-aware)
        """
        with self._lock:
            self._open_trades[instrument] = trade
            self._traded_today.add(instrument)
            self._pair_trade_count[instrument] = (
                self._pair_trade_count.get(instrument, 0) + 1
            )

            if instrument in FOREX_PAIRS:
                self._open_forex += 1
            else:
                self._open_indices += 1

        log.info(
            "[%s] Trade opened | dir=%s | entry=%.5f | sl=%.5f | units=%.0f | conf=%.2f",
            instrument,
            trade.get("direction"),
            trade.get("entry_price", 0.0),
            trade.get("sl_price",    0.0),
            trade.get("units",       0.0),
            trade.get("confidence",  0.0),
        )

    def register_close_trade(
        self,
        instrument: str,
        pnl_gbp: float,
        now: datetime,
    ) -> None:
        """
        Record a closed trade, update equity, and check kill switch.

        Parameters
        ----------
        instrument : OANDA instrument name
        pnl_gbp    : realised P&L in GBP (positive = profit, negative = loss)
        now        : current UTC datetime (tz-aware)
        """
        with self._lock:
            self.equity   += pnl_gbp
            self._day_pnl += pnl_gbp
            self._last_close_time[instrument] = now

            if instrument in self._open_trades:
                del self._open_trades[instrument]

            if instrument in FOREX_PAIRS:
                self._open_forex  = max(0, self._open_forex - 1)
            else:
                self._open_indices = max(0, self._open_indices - 1)

        log.info(
            "[%s] Trade closed | P&L=£%.2f | equity=£%.2f | day_pnl=£%.2f",
            instrument, pnl_gbp, self.equity, self._day_pnl,
        )

        # Daily drawdown kill switch — only fire on the False → True transition
        self.drawdown_floor = self.day_start_equity * (1.0 - DD_KILL_PCT)
        if not self.kill_switch and self.equity < self.drawdown_floor:
            self.kill_switch = True
            self.kill_switch_reason = (
                f"Daily drawdown exceeded: equity £{self.equity:.2f}"
            )
            log.warning("KILL SWITCH ACTIVATED — %s", self.kill_switch_reason)
            if self.on_kill_switch:
                try:
                    self.on_kill_switch()
                except Exception:
                    pass

    def update_sl(self, instrument: str, new_sl: float) -> None:
        """Update the stop-loss price for an open trade (trailing SL ratchet)."""
        with self._lock:
            if instrument in self._open_trades:
                self._open_trades[instrument]["sl_price"] = new_sl

    # ------------------------------------------------------------------
    # Day boundary
    # ------------------------------------------------------------------

    def daily_reset(self, now: datetime, new_equity: float | None = None) -> None:
        """
        Reset all daily state at UTC midnight.

        Parameters
        ----------
        now        : current UTC datetime
        new_equity : if provided, update self.equity to this value before
                     setting day_start_equity. Pass the freshly-fetched OANDA
                     balance so that day_start_equity is never set from a stale
                     in-memory figure. Omit to use current in-memory equity.

        Preserves: _open_trades, _last_close_time, equity.
        Resets:    _traded_today, _pair_trade_count, _day_pnl,
                   kill_switch, kill_switch_reason, day_start_equity.
        """
        if new_equity is not None:
            self.equity = new_equity
        self._traded_today.clear()
        self._pair_trade_count.clear()
        self._day_pnl           = 0.0
        self.kill_switch        = False
        self.kill_switch_reason = None
        self.day_start_equity   = self.equity
        self.drawdown_floor     = self.equity * (1.0 - DD_KILL_PCT)

        log.info(
            "Daily reset | equity=£%.2f | open_trades=%d | date=%s",
            self.equity,
            len(self._open_trades),
            now.strftime("%Y-%m-%d"),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_open_trade(self, instrument: str) -> dict | None:
        """Return the open trade dict for instrument, or None."""
        with self._lock:
            return self._open_trades.get(instrument)

    def get_all_open_trades(self) -> dict[str, dict]:
        """Return a snapshot of all open trade dicts keyed by instrument."""
        with self._lock:
            return dict(self._open_trades)

    def get_status(self) -> dict:
        """Return a full state snapshot for logging and dashboard writes."""
        return {
            "equity":             round(self.equity, 2),
            "day_pnl":            round(self._day_pnl, 2),
            "kill_switch":        self.kill_switch,
            "kill_switch_reason": self.kill_switch_reason,
            "open_trades":        len(self._open_trades),
            "open_forex":         self._open_forex,
            "open_indices":       self._open_indices,
            "traded_today":       sorted(self._traded_today),
        }

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _get_risk_pct(self, equity: float) -> float:
        """Return the tiered risk percentage for the given equity level."""
        for tier in RISK_TIERS:
            if equity <= tier["max_equity"]:
                return tier["risk_pct"]
        return RISK_TIERS[-1]["risk_pct"]

    def compute_position_size(
        self,
        instrument: str,
        sl_distance: float,
        entry_price: float,
        gbpusd_rate: float,
        gbpeur_rate: float | None = None,
    ) -> float:
        """
        Compute trade size (units) using the CLAUDE.md position-sizing formulas.

        Parameters
        ----------
        instrument   : OANDA instrument name
        sl_distance  : absolute price distance from entry to stop-loss
        entry_price  : trade entry price
        gbpusd_rate  : live GBP/USD rate at entry
        gbpeur_rate  : live GBP/EUR rate at entry — required for DE30_EUR

        Returns
        -------
        units (float) — caller is responsible for rounding to OANDA minimum

        Raises
        ------
        ValueError if sl_distance ≤ 0 or gbpeur_rate missing for DE30_EUR
        """
        if sl_distance <= 0:
            raise ValueError(f"sl_distance must be > 0, got {sl_distance}")

        risk_pct   = self._get_risk_pct(self.equity)
        risk_gbp   = min(self.equity * risk_pct, MAX_RISK_GBP[instrument])
        quote_type = QUOTE_TYPE[instrument]

        if quote_type == "usd_quote":
            # EUR_USD, GBP_USD, AUD_USD
            # units = risk_gbp / sl_distance_price
            units = risk_gbp / sl_distance

        elif quote_type == "usd_base":
            # USD_JPY, USD_CHF
            # sl_distance_usd = sl_distance / entry_price
            # units = (risk_gbp × gbpusd_rate) / sl_distance_usd
            sl_distance_usd = sl_distance / entry_price
            units = (risk_gbp * gbpusd_rate) / sl_distance_usd

        elif quote_type == "usd_index":
            # SPX500_USD, NAS100_USD
            # units = (risk_gbp × gbpusd_rate) / sl_distance_points
            units = (risk_gbp * gbpusd_rate) / sl_distance

        elif quote_type == "eur_index":
            # DE30_EUR
            if gbpeur_rate is None:
                raise ValueError("gbpeur_rate is required for DE30_EUR")
            units = (risk_gbp * gbpeur_rate) / sl_distance

        else:
            raise ValueError(
                f"Unknown quote_type '{quote_type}' for instrument '{instrument}'"
            )

        max_units = MAX_UNITS.get(instrument, 9999999)
        units = min(units, float(max_units))

        return units
