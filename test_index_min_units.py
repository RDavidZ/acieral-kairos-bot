"""
test_index_min_units.py

1. Queries OANDA instrument info for NAS100_USD, SPX500_USD, DE30_EUR
   to find minimum trade size and unit precision.
2. Tests placing the minimum unit trade on each, then immediately closes.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

import oandapyV20
import oandapyV20.endpoints.orders as order_ep
import oandapyV20.endpoints.trades as trades_ep
import oandapyV20.endpoints.pricing as pricing_ep
import oandapyV20.endpoints.instruments as instruments_ep
from oandapyV20.exceptions import V20Error
import os

API_KEY    = os.environ["OANDA_API_KEY"]
ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
ENV        = os.environ.get("OANDA_ENVIRONMENT", "practice")

assert ENV == "practice", f"ABORT: environment is '{ENV}'"

api = oandapyV20.API(access_token=API_KEY, environment=ENV)

SEP = "=" * 60

INSTRUMENTS = ["NAS100_USD", "SPX500_USD", "DE30_EUR"]
# SL offsets (points below ask) — wide enough not to hit immediately
SL_OFFSETS = {
    "NAS100_USD": 300.0,
    "SPX500_USD": 100.0,
    "DE30_EUR":   300.0,
}

# ---------------------------------------------------------------------------
# Step 1: Query instrument info for all three
# ---------------------------------------------------------------------------
print(SEP)
print("Step 1: OANDA instrument info")
print(SEP)

import oandapyV20.endpoints.accounts as accounts_ep
req = accounts_ep.AccountInstruments(
    ACCOUNT_ID,
    params={"instruments": ",".join(INSTRUMENTS)}
)
api.request(req)
instrument_info = {i["name"]: i for i in req.response["instruments"]}

for inst in INSTRUMENTS:
    info = instrument_info.get(inst, {})
    min_trade = info.get("minimumTradeSize", "?")
    trade_units_precision = info.get("tradeUnitsPrecision", "?")
    min_guaranteed_sl     = info.get("guaranteedStopLossOrderLevelRestriction", {})
    print(f"  {inst:<14}  minimumTradeSize={min_trade}  tradeUnitsPrecision={trade_units_precision}")

print()

# ---------------------------------------------------------------------------
# Step 2: Get current prices
# ---------------------------------------------------------------------------
print(SEP)
print("Step 2: Current prices")
print(SEP)

req = pricing_ep.PricingInfo(ACCOUNT_ID, params={"instruments": ",".join(INSTRUMENTS)})
api.request(req)
prices = {p["instrument"]: p for p in req.response["prices"]}

for inst in INSTRUMENTS:
    p = prices[inst]
    ask = float(p["asks"][0]["price"])
    bid = float(p["bids"][0]["price"])
    print(f"  {inst:<14}  bid={bid}  ask={ask}")

print()

# ---------------------------------------------------------------------------
# Step 3: Place minimum unit trade on each, then close immediately
# ---------------------------------------------------------------------------
print(SEP)
print("Step 3: Test minimum unit trade on each instrument")
print(SEP)

for inst in INSTRUMENTS:
    info      = instrument_info.get(inst, {})
    min_units = info.get("minimumTradeSize", "1")
    p         = prices[inst]
    ask       = float(p["asks"][0]["price"])
    sl_price  = round(ask - SL_OFFSETS[inst], 1)

    print(f"  [{inst}] placing {min_units} unit LONG @ ask={ask}  SL={sl_price}")

    body = {
        "order": {
            "type":         "MARKET",
            "instrument":   inst,
            "units":        str(min_units),
            "timeInForce":  "FOK",
            "positionFill": "DEFAULT",
            "stopLossOnFill": {
                "price":       str(sl_price),
                "timeInForce": "GTC",
            },
        }
    }

    try:
        req = order_ep.OrderCreate(ACCOUNT_ID, data=body)
        api.request(req)
        resp = req.response

        fill = resp.get("orderFillTransaction", {})
        if fill:
            trade_id     = fill.get("tradeOpened", {}).get("tradeID")
            fill_price   = fill.get("price")
            units_filled = fill.get("units")
            print(f"    ACCEPTED | trade_id={trade_id} | fill={fill_price} | units={units_filled}")

            # Close immediately
            close_req = trades_ep.TradeClose(ACCOUNT_ID, tradeID=trade_id)
            api.request(close_req)
            close = close_req.response.get("orderFillTransaction", {})
            print(f"    Closed   | price={close.get('price')} | P&L={close.get('pl')}")
        else:
            cancelled = resp.get("orderCancelTransaction", {})
            print(f"    REJECTED | reason={cancelled.get('reason', 'unknown')}")

    except V20Error as e:
        print(f"    V20Error: {e}")

    print()

print(SEP)
print("Done.")
