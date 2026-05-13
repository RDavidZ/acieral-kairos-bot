"""
test_nas100_fractional.py

Tests whether OANDA practice account accepts 0.5 units for NAS100_USD.
Places a 0.5 unit LONG, then immediately closes it.
Runs against practice account only.
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
from oandapyV20.exceptions import V20Error
import os

API_KEY    = os.environ["OANDA_API_KEY"]
ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID"]
ENV        = os.environ.get("OANDA_ENVIRONMENT", "practice")

assert ENV == "practice", f"ABORT: environment is '{ENV}' — only running on practice"

api = oandapyV20.API(access_token=API_KEY, environment=ENV)

SEP = "=" * 60

# ---------------------------------------------------------------------------
# 1. Get current NAS100 price
# ---------------------------------------------------------------------------
print(SEP)
print("Step 1: Get NAS100_USD current price")
req  = pricing_ep.PricingInfo(ACCOUNT_ID, params={"instruments": "NAS100_USD"})
api.request(req)
prices = req.response["prices"][0]
ask    = float(prices["asks"][0]["price"])
bid    = float(prices["bids"][0]["price"])
print(f"  bid={bid}  ask={ask}")

# Simple SL: 200 points below ask (intentionally wide so it doesn't hit)
sl_price = round(ask - 200.0, 1)
print(f"  SL price = {sl_price}")

# ---------------------------------------------------------------------------
# 2. Attempt 0.5 unit LONG order
# ---------------------------------------------------------------------------
print(SEP)
print("Step 2: Place 0.5 unit LONG order")

body = {
    "order": {
        "type":         "MARKET",
        "instrument":   "NAS100_USD",
        "units":        "0.5",          # fractional — key test
        "timeInForce":  "FOK",
        "positionFill": "DEFAULT",
        "stopLossOnFill": {
            "price":       str(sl_price),
            "timeInForce": "GTC",
        },
    }
}

try:
    req  = order_ep.OrderCreate(ACCOUNT_ID, data=body)
    api.request(req)
    resp = req.response

    fill = resp.get("orderFillTransaction", {})
    if fill:
        trade_id    = fill.get("tradeOpened", {}).get("tradeID")
        fill_price  = fill.get("price")
        units_filled = fill.get("units")
        print(f"  ACCEPTED by OANDA")
        print(f"  trade_id    = {trade_id}")
        print(f"  fill_price  = {fill_price}")
        print(f"  units_filled= {units_filled}")
    else:
        cancelled = resp.get("orderCancelTransaction", {})
        print(f"  REJECTED / not filled")
        print(f"  reason = {cancelled.get('reason', 'unknown')}")
        print(f"  full response: {resp}")
        sys.exit(0)

except V20Error as e:
    print(f"  V20Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Immediately close the trade
# ---------------------------------------------------------------------------
print(SEP)
print(f"Step 3: Close trade {trade_id}")

try:
    req  = trades_ep.TradeClose(ACCOUNT_ID, tradeID=trade_id)
    api.request(req)
    close = req.response.get("orderFillTransaction", {})
    close_price = close.get("price")
    pnl         = close.get("pl")
    print(f"  Closed at {close_price} | P&L = {pnl}")
except V20Error as e:
    print(f"  Close failed: {e}")

print(SEP)
print("RESULT: OANDA accepts 0.5 fractional units for NAS100_USD")
print("The bot can send fractional units — no need to round to integer.")
