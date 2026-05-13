"""
analysis/live_vs_backtest_check.py

Compare live practice trades from Supabase against what the 2026 backtest
predicted for the same instruments and dates.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "acieral-dashboard"))

# Load .env so dashboard.db can find DB credentials
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import pandas as pd

# ---------------------------------------------------------------------------
# Step 1 — Load practice trades
# ---------------------------------------------------------------------------

from dashboard.db import get_trades

practice = pd.DataFrame(get_trades(
    bot_id="acieral_kairos_v1", status="closed",
    trade_type="practice", limit=500
))
practice = practice[~practice["exit_reason"].isin(["RECONCILED", "MANUAL_CLOSE"])].copy()
practice["entry_time"] = pd.to_datetime(practice["entry_time"], utc=True).dt.floor("h")
practice["entry_date"] = practice["entry_time"].dt.date
practice["pnl_gbp"] = practice["pnl_gbp"].astype(float)
practice["entry_price"] = practice["entry_price"].astype(float)

# Normalise instrument name: DB stores e.g. "EUR_USD", backtest CSVs use same
# but column is named "pair" in DB
practice = practice.rename(columns={"pair": "instrument"})

# ---------------------------------------------------------------------------
# Step 2 — Load backtest trades (all instruments)
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "backtest" / "results_exit"
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF",
    "SPX500_USD", "NAS100_USD", "DE30_EUR",
]

bt_frames = []
for instr in INSTRUMENTS:
    path = RESULTS_DIR / f"{instr}_trades.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True).dt.floor("h")
    df["entry_date"] = df["entry_time"].dt.date
    df["instrument"] = instr
    bt_frames.append(df)

backtest = pd.concat(bt_frames, ignore_index=True)

# ---------------------------------------------------------------------------
# Step 3 — Match practice trades to backtest trades
# ---------------------------------------------------------------------------

SEP = "=" * 80

print(SEP)
print("LIVE vs BACKTEST COMPARISON")
print(SEP)
print(f"Practice trades (excl RECONCILED/MANUAL_CLOSE): {len(practice)}")
print(f"Backtest trades (2026 OOS): {len(backtest)}")
print()

results = []

for _, pt in practice.sort_values("entry_time").iterrows():
    instr = pt["instrument"]
    date  = pt["entry_date"]

    # Find backtest trades for same instrument and date
    bt_day = backtest[
        (backtest["instrument"] == instr) &
        (backtest["entry_date"] == date)
    ]

    row = {
        "instrument":      instr,
        "entry_date":      date,
        "live_dir":        pt["direction"],
        "live_hour":       pt["entry_time"].hour,
        "live_price":      pt["entry_price"],
        "live_exit":       pt["exit_reason"],
        "live_win":        pt["pnl_gbp"] > 0,
        "live_pnl":        pt["pnl_gbp"],
        "match_found":     not bt_day.empty,
    }

    if bt_day.empty:
        row.update({
            "direction_match": None,
            "same_candle":     None,
            "price_close":     None,
            "outcome_match":   None,
            "bt_dir":          None,
            "bt_hour":         None,
            "bt_price":        None,
            "bt_exit":         None,
            "bt_win":          None,
            "bt_pnl":          None,
        })
    else:
        bt = bt_day.iloc[0]  # at most 1 trade per instrument per day
        bt_win = bt["pnl_gbp"] > 0

        price_tol = abs(pt["entry_price"] - bt["entry_price"]) / bt["entry_price"]

        row.update({
            "direction_match": pt["direction"] == bt["direction"],
            "same_candle":     pt["entry_time"].hour == bt["entry_time"].hour,
            "price_close":     price_tol <= 0.001,
            "outcome_match":   (pt["pnl_gbp"] > 0) == bt_win,
            "bt_dir":          bt["direction"],
            "bt_hour":         bt["entry_time"].hour,
            "bt_price":        bt["entry_price"],
            "bt_exit":         bt["exit_reason"],
            "bt_win":          bt_win,
            "bt_pnl":          bt["pnl_gbp"],
        })

    results.append(row)

df_res = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Step 4 — Per-trade detail
# ---------------------------------------------------------------------------

print(SEP)
print("PER-TRADE DETAIL")
print(SEP)

hdr = (f"{'Date':<12} {'Instr':<12} {'LiveDir':<8} {'LiveH':>5} "
       f"{'LivePnL':>8} {'Match':>6} {'DirOK':>6} {'SameCdl':>8} "
       f"{'PxClose':>8} {'OutcOK':>7} {'BtDir':<8} {'BtH':>4} {'BtPnL':>8} {'Anomaly'}")
print(hdr)
print("-" * 110)

anomalies = []

for _, r in df_res.iterrows():
    match_str  = "YES"  if r["match_found"]     else "NO"
    dir_str    = ("Y" if r["direction_match"] else "N") if r["match_found"] else "-"
    cdl_str    = ("Y" if r["same_candle"]     else "N") if r["match_found"] else "-"
    px_str     = ("Y" if r["price_close"]     else "N") if r["match_found"] else "-"
    out_str    = ("Y" if r["outcome_match"]   else "N") if r["match_found"] else "-"
    bt_dir_str = str(r["bt_dir"]) if r["match_found"] else "-"
    bt_h_str   = str(int(r["bt_hour"])) if r["match_found"] else "-"
    bt_pnl_str = f"£{r['bt_pnl']:+.2f}" if r["match_found"] else "-"
    live_pnl_s = f"£{r['live_pnl']:+.2f}"

    anomaly = ""
    if not r["match_found"]:
        anomaly = "UNEXPECTED ENTRY"
        anomalies.append((r["entry_date"], r["instrument"], anomaly))
    elif not r["direction_match"]:
        anomaly = "DIRECTION MISMATCH"
        anomalies.append((r["entry_date"], r["instrument"], anomaly))

    print(
        f"{str(r['entry_date']):<12} {r['instrument']:<12} {r['live_dir']:<8} "
        f"{r['live_hour']:>5} {live_pnl_s:>8} {match_str:>6} {dir_str:>6} "
        f"{cdl_str:>8} {px_str:>8} {out_str:>7} {bt_dir_str:<8} {bt_h_str:>4} "
        f"{bt_pnl_str:>8}  {anomaly}"
    )

# ---------------------------------------------------------------------------
# Step 5 — Missed signals (backtest traded, live did not)
# ---------------------------------------------------------------------------

print()
print(SEP)
print("MISSED SIGNALS — backtest traded, live skipped")
print(SEP)

# Dates + instruments covered by live trades
live_keys = set(zip(df_res["instrument"], df_res["entry_date"]))

# Date range of live trading
live_min = practice["entry_date"].min()
live_max = practice["entry_date"].max()

# Backtest trades within that date range
bt_in_range = backtest[
    (backtest["entry_date"] >= live_min) &
    (backtest["entry_date"] <= live_max)
]

missed = []
for _, bt in bt_in_range.iterrows():
    key = (bt["instrument"], bt["entry_date"])
    if key not in live_keys:
        missed.append(bt)

if missed:
    missed_df = pd.DataFrame(missed)
    print(f"{'Date':<12} {'Instrument':<14} {'Dir':<6} {'Hour':>5} {'BtPnL':>10}  BtExit")
    print("-" * 60)
    for _, m in missed_df.iterrows():
        print(f"{str(m['entry_date']):<12} {m['instrument']:<14} {m['direction']:<6} "
              f"{m['entry_time'].hour:>5} £{m['pnl_gbp']:>+9.2f}  {m['exit_reason']}")
else:
    print("  None — live bot traded all days that backtest predicted.")

# ---------------------------------------------------------------------------
# Step 6 — Summary statistics
# ---------------------------------------------------------------------------

print()
print(SEP)
print("SUMMARY STATISTICS")
print(SEP)

n_total   = len(df_res)
n_matched = df_res["match_found"].sum()
n_unmatched = n_total - n_matched

matched = df_res[df_res["match_found"]]

dir_agree  = matched["direction_match"].sum() if n_matched else 0
cdl_agree  = matched["same_candle"].sum()     if n_matched else 0
px_agree   = matched["price_close"].sum()     if n_matched else 0
out_agree  = matched["outcome_match"].sum()   if n_matched else 0

print(f"  Total practice trades checked : {n_total}")
print(f"  Matches found                 : {n_matched} ({n_matched/n_total:.0%})")
print(f"  Unexpected entries (no match) : {n_unmatched}")
print(f"  Missed signals                : {len(missed)}")
print()
if n_matched:
    print(f"  Of {n_matched} matched trades:")
    print(f"    Direction agreement : {dir_agree}/{n_matched} ({dir_agree/n_matched:.0%})")
    print(f"    Same candle (hour)  : {cdl_agree}/{n_matched} ({cdl_agree/n_matched:.0%})")
    print(f"    Price within 0.1%   : {px_agree}/{n_matched} ({px_agree/n_matched:.0%})")
    print(f"    Outcome agreement   : {out_agree}/{n_matched} ({out_agree/n_matched:.0%})")

# ---------------------------------------------------------------------------
# Step 7 — Anomaly summary
# ---------------------------------------------------------------------------

print()
print(SEP)
print("ANOMALIES")
print(SEP)

dir_mismatches = [(d, i, a) for d, i, a in anomalies if a == "DIRECTION MISMATCH"]
unexpected     = [(d, i, a) for d, i, a in anomalies if a == "UNEXPECTED ENTRY"]

if dir_mismatches:
    print("  DIRECTION MISMATCHES:")
    for d, i, a in dir_mismatches:
        row = df_res[(df_res["entry_date"] == d) & (df_res["instrument"] == i)].iloc[0]
        print(f"    {d} {i}: live={row['live_dir']} bt={row['bt_dir']}")
else:
    print("  No direction mismatches.")

print()
if unexpected:
    print("  UNEXPECTED ENTRIES (live traded, backtest skipped):")
    for d, i, a in unexpected:
        row = df_res[(df_res["entry_date"] == d) & (df_res["instrument"] == i)].iloc[0]
        print(f"    {d} {i}: live={row['live_dir']} P&L=£{row['live_pnl']:+.2f}")
else:
    print("  No unexpected entries.")

print()
if missed:
    print("  MISSED SIGNALS (backtest traded, live skipped):")
    for _, m in pd.DataFrame(missed).iterrows():
        print(f"    {m['entry_date']} {m['instrument']}: bt={m['direction']} P&L=£{m['pnl_gbp']:+.2f}")
else:
    print("  No missed signals.")

print()
