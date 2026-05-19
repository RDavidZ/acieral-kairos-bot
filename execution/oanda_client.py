"""
execution/oanda_client.py — OANDA v20 REST API wrapper for Acieral Kairos Bot

All public methods route through _request_with_retry, which handles
transient network and server errors with exponential backoff (3 retries:
2 s, 4 s, 8 s).

JPY instrument prices are rounded to 3 dp; all others to 5 dp.
OANDA environment is read from .env (must be 'practice' — never change
without explicit sign-off).
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import oandapyV20
import oandapyV20.endpoints.accounts    as acct_ep
import oandapyV20.endpoints.instruments as instr_ep
import oandapyV20.endpoints.orders      as order_ep
import oandapyV20.endpoints.pricing     as pricing_ep
import oandapyV20.endpoints.trades      as trade_ep
from oandapyV20.exceptions import V20Error

from config import HARD_CONSTRAINTS

log = logging.getLogger(__name__)

_HC = HARD_CONSTRAINTS
FOREX_PAIRS       = _HC["FOREX_PAIRS"]
INDEX_INSTRUMENTS = _HC["INDEX_INSTRUMENTS"]

# ---------------------------------------------------------------------------
# Index session hours (UTC) — open and close as (hour, minute) tuples
# ---------------------------------------------------------------------------

_SESSION_OPEN_UTC: dict[str, tuple[int, int]] = {
    "SPX500_USD": (14, 30),
    "NAS100_USD": (14, 30),
    "DE30_EUR":   ( 8,  0),
}

_SESSION_CLOSE_UTC: dict[str, tuple[int, int]] = {
    "SPX500_USD": (21,  0),
    "NAS100_USD": (21,  0),
    "DE30_EUR":   (16, 30),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_dp(instrument: str) -> int:
    """Return decimal places for price formatting: 3 for JPY, 5 for all others."""
    return 3 if "JPY" in instrument else 5


def _fmt_price(price: float, instrument: str) -> str:
    dp = _price_dp(instrument)
    return f"{price:.{dp}f}"


def _format_sl_price(instrument: str, price: float) -> str:
    index_instruments = {"SPX500_USD", "NAS100_USD", "DE30_EUR"}
    if instrument in index_instruments:
        return f"{price:.1f}"
    elif "JPY" in instrument:
        return f"{price:.3f}"
    else:
        return f"{price:.5f}"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OandaClient:
    """
    Thin, robust wrapper around oandapyV20.

    Usage
    -----
    client = OandaClient()          # loads credentials from .env
    account = client.get_account()
    df = client.get_latest_candles('EUR_USD', 'H1', count=100)
    bid, ask = client.get_current_price('EUR_USD')
    """

    def __init__(self) -> None:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # fall back to system environment

        api_key     = os.environ.get("OANDA_API_KEY")
        account_id  = os.environ.get("OANDA_ACCOUNT_ID")
        environment = os.environ.get("OANDA_ENVIRONMENT", "practice")

        if not api_key:
            raise ValueError("OANDA_API_KEY not set in environment / .env")
        if not account_id:
            raise ValueError("OANDA_ACCOUNT_ID not set in environment / .env")

        self.account_id  = account_id
        self.environment = environment

        # oandapyV20 uses "practice" or "live"
        env_str  = "practice" if environment == "practice" else "live"
        self.api = oandapyV20.API(access_token=api_key, environment=env_str)

        # Serialise all API calls — oandapyV20 sets request.response on the
        # endpoint object, which is not thread-safe across concurrent callers
        # (scheduler thread + PriceStream thread both call this client).
        self._api_lock = threading.Lock()

        log.info(
            "OandaClient ready | environment=%s | account=%s",
            self.environment, self.account_id,
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """
        Return account summary.

        Returns
        -------
        dict with keys: balance, unrealised_pl, nav, currency
        """
        req  = acct_ep.AccountSummary(self.account_id)
        resp = self._request_with_retry(req)
        acct = resp["account"]
        return {
            "balance":       float(acct["balance"]),
            "unrealised_pl": float(acct.get("unrealizedPL", 0.0)),
            "nav":           float(acct.get("NAV", acct["balance"])),
            "currency":      acct["currency"],
        }

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_latest_candles(
        self,
        instrument: str,
        granularity: str = "H1",
        count: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch the most recent ``count`` *completed* candles.

        Requests count+1 from OANDA to guard against an in-progress candle
        occupying the last slot, then filters for complete candles and
        returns the final ``count`` rows.

        Returns
        -------
        DataFrame with columns: time (UTC tz-aware), open, high, low,
        close, volume  — sorted ascending by time.
        """
        params = {
            "granularity": granularity,
            "count":       count + 1,   # +1 to absorb any in-progress candle
            "price":       "M",         # mid-price
        }
        req  = instr_ep.InstrumentsCandles(instrument, params=params)
        resp = self._request_with_retry(req)

        rows = []
        for c in resp.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            rows.append({
                "time":   pd.to_datetime(c["time"], utc=True),
                "open":   float(mid["o"]),
                "high":   float(mid["h"]),
                "low":    float(mid["l"]),
                "close":  float(mid["c"]),
                "volume": int(c["volume"]),
            })

        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

        if len(df) > count:
            df = df.tail(count).reset_index(drop=True)

        if df.empty:
            log.warning("[%s] get_latest_candles returned 0 complete candles", instrument)

        return df

    def get_current_price(self, instrument: str) -> tuple[float, float]:
        """
        Return (bid, ask) for ``instrument``.

        Uses PricingInfo — no candle latency.
        """
        req  = pricing_ep.PricingInfo(
            self.account_id, params={"instruments": instrument}
        )
        resp = self._request_with_retry(req)
        p    = resp["prices"][0]
        bid  = float(p["bids"][0]["price"])
        ask  = float(p["asks"][0]["price"])
        return bid, ask

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def market_order(
        self,
        instrument: str,
        units: float,
        sl_price: float,
    ) -> dict:
        """
        Place a market order with an attached GTC stop loss.

        Parameters
        ----------
        instrument : OANDA instrument name (e.g. 'EUR_USD')
        units      : positive = LONG, negative = SHORT
        sl_price   : stop-loss price level

        Returns
        -------
        dict: order_id, trade_id, fill_price, fill_time, units_filled, status

        Raises
        ------
        RuntimeError if the order does not fill (rejected, market closed, etc.)
        V20Error     for non-retryable API errors
        """
        sl_str = _format_sl_price(instrument, sl_price)

        if instrument in FOREX_PAIRS:
            units = int(round(units))
        elif instrument in INDEX_INSTRUMENTS:
            units = round(units, 2)

        body = {
            "order": {
                "type":         "MARKET",
                "instrument":   instrument,
                "units":        str(units),
                "timeInForce":  "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price":       sl_str,
                    "timeInForce": "GTC",
                },
            }
        }

        req  = order_ep.OrderCreate(self.account_id, data=body)
        resp = self._request_with_retry(req)

        fill = resp.get("orderFillTransaction", {})
        if not fill:
            raise RuntimeError(
                f"[{instrument}] Market order did not fill. Response: {resp}"
            )

        trade_opened = fill.get("tradeOpened", {})
        trade_id     = trade_opened.get("tradeID", fill.get("id", ""))

        log.info(
            "[%s] Market order filled | trade_id=%s | price=%s | units=%s",
            instrument, trade_id, fill.get("price"), fill.get("units"),
        )

        return {
            "order_id":     fill.get("orderID", ""),
            "trade_id":     trade_id,
            "fill_price":   float(fill.get("price", 0.0)),
            "fill_time":    fill.get("time", ""),
            "units_filled": int(float(fill.get("units", units))),
            "status":       "filled",
        }

    def get_closed_trade(self, trade_id: str) -> dict | None:
        """
        Fetch details of a closed trade via TradeDetails endpoint.
        Returns dict with close_price and pnl_usd, or None if not found.
        """
        try:
            from oandapyV20.endpoints.trades import TradeDetails
            r = TradeDetails(accountID=self.account_id, tradeID=str(trade_id))
            with self._api_lock:
                self.api.request(r)
            trade = r.response["trade"]
            state = trade.get("state", "")
            if state == "CLOSED":
                close_price = float(trade["averageClosePrice"])
                pnl_usd     = float(trade.get("realizedPL", 0.0))
                return {"close_price": close_price, "pnl_usd": pnl_usd}
            return None
        except Exception as exc:
            log.warning("get_closed_trade(%s) failed: %s", trade_id, exc)
            return None

    def get_trade_transaction(self, trade_id: str) -> dict | None:
        """
        Query transaction history to find the close transaction for a trade.

        Fallback for when TradeDetails returns None (e.g. stale/reconciled trade IDs
        that OANDA closed before the bot could record the fill price).

        Fetches transactions starting from just before the trade ID, then scans
        ORDER_FILL transactions for one that closed our trade.

        Returns dict with close_price and realised_pl_account (in account currency),
        or None if not found. Caller should compute GBP P&L from close_price using
        _compute_pnl_gbp — do not use realised_pl_account directly (it's in USD).
        """
        try:
            from oandapyV20.endpoints.transactions import TransactionsSinceID
            params = {"id": str(max(1, int(trade_id) - 10))}
            req = TransactionsSinceID(accountID=self.account_id, params=params)
            with self._api_lock:
                self.api.request(req)
            transactions = req.response.get("transactions", [])
            for txn in transactions:
                if txn.get("type") != "ORDER_FILL":
                    continue
                for tc in txn.get("tradesClosed", []):
                    if str(tc.get("tradeID")) == str(trade_id):
                        close_price = float(txn.get("price", 0))
                        realised_pl = float(txn.get("pl", 0))
                        log.info(
                            "get_trade_transaction(%s): found close at %.5f (pl=%s)",
                            trade_id, close_price, realised_pl,
                        )
                        return {
                            "close_price":          close_price,
                            "realised_pl_account":  realised_pl,
                        }
            log.warning("get_trade_transaction(%s): no matching ORDER_FILL found", trade_id)
            return None
        except Exception as exc:
            log.warning("get_trade_transaction(%s) failed: %s", trade_id, exc)
            return None

    def close_trade(self, trade_id: str, instrument: str) -> dict:
        """
        Close a specific trade by trade_id.

        Returns
        -------
        dict: trade_id, close_price, close_time, pnl, status
        """
        req  = trade_ep.TradeClose(self.account_id, trade_id)
        resp = self._request_with_retry(req)

        fill  = resp.get("orderFillTransaction", {})
        price = fill.get("price")
        pl    = fill.get("pl")

        log.info(
            "[%s] Trade closed | trade_id=%s | price=%s | pl=%s",
            instrument, trade_id, price, pl,
        )

        if not price:
            raise ValueError(
                f"close_trade({trade_id}): OANDA response missing fill price — "
                f"response keys: {list(resp.keys())}"
            )

        return {
            "trade_id":    trade_id,
            "close_price": float(price),
            "close_time":  fill.get("time", ""),
            "pnl":         float(pl) if pl is not None else 0.0,
            "status":      "closed",
        }

    def update_stop_loss(
        self,
        trade_id: str,
        instrument: str,
        new_sl_price: float,
    ) -> dict:
        """
        Update the stop-loss order attached to an open trade.

        Price precision: 3 dp for JPY, 5 dp for all others.

        Returns
        -------
        Raw response dict from OANDA (tradeOrdersModified transaction)
        """
        sl_str = _format_sl_price(instrument, new_sl_price)

        body = {
            "stopLoss": {
                "price":       sl_str,
                "timeInForce": "GTC",
            }
        }

        req  = trade_ep.TradeCRCDO(self.account_id, trade_id, data=body)
        resp = self._request_with_retry(req)

        log.debug(
            "[%s] SL updated | trade_id=%s | new_sl=%s",
            instrument, trade_id, sl_str,
        )

        return resp

    # ------------------------------------------------------------------
    # Open trades
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list[dict]:
        """
        Return all open trades on the account.

        Each dict: trade_id, instrument, units, open_price,
                   current_price, unrealised_pl, open_time, sl_price.
        """
        req  = trade_ep.OpenTrades(self.account_id)
        resp = self._request_with_retry(req)

        result = []
        for t in resp.get("trades", []):
            sl_order = t.get("stopLossOrder", {})
            sl_price = float(sl_order["price"]) if sl_order.get("price") else 0.0
            result.append({
                "trade_id":      t["id"],
                "instrument":    t["instrument"],
                "units":         float(t["currentUnits"]),
                "open_price":    float(t["price"]),
                "current_price": float(t.get("currentPriceAsk", t.get("price", 0.0))),
                "unrealised_pl": float(t.get("unrealizedPL", 0.0)),
                "open_time":     t["openTime"],
                "sl_price":      sl_price,
            })

        return result

    def get_open_trade(self, instrument: str) -> dict | None:
        """Return the open trade for a specific instrument, or None."""
        for t in self.get_open_trades():
            if t["instrument"] == instrument:
                return t
        return None

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_open(
        self,
        instrument: str,
        now: datetime | None = None,
    ) -> bool:
        """
        Return True if the instrument is tradeable at the given time.

        Forex  : open Mon 21:00 UTC → Fri 21:00 UTC (closed weekends).
        Indices: open only within their weekday session window:
                   SPX500_USD / NAS100_USD  14:30–21:00 UTC
                   DE30_EUR                  08:00–16:30 UTC

        Parameters
        ----------
        instrument : OANDA instrument name
        now        : UTC datetime — defaults to datetime.now(timezone.utc)
        """
        if now is None:
            now = datetime.now(timezone.utc)

        wd     = now.weekday()   # Monday=0 … Sunday=6
        hm     = (now.hour, now.minute)

        if instrument in FOREX_PAIRS:
            if wd == 5:                          # Saturday: always closed
                return False
            if wd == 6 and hm < (21, 0):         # Sunday before 21:00: closed
                return False
            if wd == 4 and hm >= (21, 0):        # Friday from 21:00: closed
                return False
            return True

        if instrument in INDEX_INSTRUMENTS:
            if wd >= 5:                          # Weekend: closed
                return False
            return True                          # Open all weekday hours — index CFDs trade 23h/day

        log.warning(
            "is_market_open: unknown instrument '%s' — returning True", instrument
        )
        return True

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _request_with_retry(self, request, max_retries: int = 3) -> dict:
        """
        Execute an oandapyV20 request with retry on transient errors.

        Retries on
        ----------
        - ConnectionError (network-level failure)
        - V20Error with HTTP status 503 or 504 (gateway / service unavailable)

        Backoff: 2 s → 4 s → 8 s.

        Raises
        ------
        The last exception if all retries are exhausted, or immediately
        for non-retryable V20Errors (e.g. 400, 401, 404).
        """
        delay     = 2.0
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                with self._api_lock:
                    self.api.request(request)
                return request.response

            except V20Error as exc:
                if exc.code in (503, 504):
                    last_exc = exc
                    log.warning(
                        "OANDA %d (attempt %d/%d) — retry in %.0fs | %s",
                        exc.code, attempt, max_retries, delay, exc,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error("OANDA non-retryable error %d: %s", exc.code, exc)
                    raise

            except ConnectionError as exc:
                last_exc = exc
                log.warning(
                    "ConnectionError (attempt %d/%d) — retry in %.0fs | %s",
                    attempt, max_retries, delay, exc,
                )
                time.sleep(delay)
                delay *= 2

        log.error("Max retries (%d) exhausted. Last error: %s", max_retries, last_exc)
        raise last_exc
