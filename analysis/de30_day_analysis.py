"""
analysis/de30_day_analysis.py

Drill-down analysis of DE30_EUR 2026 backtest performance by day of week.
Reads backtest/results_2026/DE30_EUR_trades.csv.
"""

import math
from pathlib import Path

import pandas as pd

TRADES_PATH = Path(__file__).parent.parent / "backtest" / "results_2026" / "DE30_EUR_trades.csv"
DAY_NAMES   = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
SEP         = "=" * 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def profit_factor(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = pnl[pnl < 0].abs().sum()
    if gl == 0:
        return gp if gp > 0 else 0.0
    return gp / gl


def sharpe(pnl: pd.Series, exit_times: pd.Series) -> float:
    if len(pnl) < 2:
        return 0.0
    daily = pnl.groupby(exit_times.dt.date).sum()
    std = daily.std()
    if std == 0:
        return 0.0
    return float(daily.mean() / std * math.sqrt(252))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(TRADES_PATH, parse_dates=["entry_time", "exit_time"])
df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
df["dow"]        = df["entry_time"].dt.dayofweek
df["dow_name"]   = df["dow"].map(DAY_NAMES)
df["entry_hour"] = df["entry_time"].dt.hour
df["win"]        = df["pnl_gbp"] > 0

print(SEP)
print("DE30_EUR 2026 BACKTEST — DAY OF WEEK DRILL-DOWN")
print(SEP)
print(f"Total trades: {len(df)}")
print(f"Date range  : {df['entry_time'].min().date()} -> {df['entry_time'].max().date()}")


# ---------------------------------------------------------------------------
# Analysis 5 — Sample size warning (shown up front)
# ---------------------------------------------------------------------------

print()
print(SEP)
print("ANALYSIS 5 — SAMPLE SIZE CHECK")
print(SEP)

for dow, name in DAY_NAMES.items():
    n = (df["dow"] == dow).sum()
    flag = "  ** INDICATIVE ONLY (n < 10) **" if n < 10 else ""
    print(f"  {name:<12}: {n:>3} trades{flag}")


# ---------------------------------------------------------------------------
# Analysis 1 — Per day of week metrics
# ---------------------------------------------------------------------------

print()
print(SEP)
print("ANALYSIS 1 — PER DAY OF WEEK METRICS")
print(SEP)

header = (f"{'Day':<12} {'Trades':>6} {'Win%':>6} {'PF':>6} "
          f"{'AvgP&L':>8} {'TotalP&L':>10} {'AvgBars':>8}")
print(header)
print("-" * 60)

for dow in range(5):
    name = DAY_NAMES[dow]
    sub  = df[df["dow"] == dow]
    if sub.empty:
        print(f"{name:<12}     0      -      -        -          -        -")
        continue
    n      = len(sub)
    wr     = sub["win"].mean() * 100
    pf     = profit_factor(sub["pnl_gbp"])
    avg    = sub["pnl_gbp"].mean()
    total  = sub["pnl_gbp"].sum()
    avg_bh = sub["bars_held"].mean()
    print(f"{name:<12} {n:>6} {wr:>6.1f} {pf:>6.2f} {avg:>8.2f} {total:>10.2f} {avg_bh:>8.1f}")

print()
print("Exit reason breakdown by day:")
print("-" * 60)

exit_pivot = (
    df.groupby(["dow_name", "exit_reason"])
    .size()
    .unstack(fill_value=0)
)
# Order by day
exit_pivot = exit_pivot.reindex(
    [DAY_NAMES[d] for d in range(5) if DAY_NAMES[d] in exit_pivot.index]
)
print(exit_pivot.to_string())


# ---------------------------------------------------------------------------
# Analysis 2 — Individual Monday and Friday trades
# ---------------------------------------------------------------------------

for dow, name in [(0, "Monday"), (4, "Friday")]:
    sub = df[df["dow"] == dow].sort_values("entry_time")
    print()
    print(SEP)
    print(f"ANALYSIS 2 — INDIVIDUAL {name.upper()} TRADES ({len(sub)} total)")
    print(SEP)

    if sub.empty:
        print(f"  No {name} trades in this period.")
        continue

    hdr = (f"  {'Entry (UTC)':<20} {'Exit (UTC)':<20} {'Dir':<6} "
           f"{'EntryPx':>9} {'ExitPx':>9} {'P&L':>8} "
           f"{'Result':<6} {'Reason':<14} {'Bars':>5} {'Conf':>6}")
    print(hdr)
    print("  " + "-" * 108)

    for _, r in sub.iterrows():
        result = "WIN" if r["win"] else "LOSS"
        entry_str = r["entry_time"].strftime("%Y-%m-%d %H:%M")
        exit_str  = r["exit_time"].strftime("%Y-%m-%d %H:%M")
        print(
            f"  {entry_str:<20} {exit_str:<20} {r['direction']:<6} "
            f"{r['entry_price']:>9.2f} {r['exit_price']:>9.2f} {r['pnl_gbp']:>8.2f} "
            f"{result:<6} {str(r['exit_reason']):<14} {int(r['bars_held']):>5} "
            f"{r['confidence']:>6.3f}"
        )

    total_pnl = sub["pnl_gbp"].sum()
    wr        = sub["win"].mean() * 100
    print(f"\n  Summary: {len(sub)} trades | Win%={wr:.1f}% | Total P&L=£{total_pnl:.2f}")


# ---------------------------------------------------------------------------
# Analysis 3 — Intraday timing on Monday and Friday
# ---------------------------------------------------------------------------

print()
print(SEP)
print("ANALYSIS 3 — INTRADAY ENTRY HOUR DISTRIBUTION (Monday + Friday)")
print(SEP)

mf = df[df["dow"].isin([0, 4])].copy()

if mf.empty:
    print("  No Monday/Friday trades.")
else:
    for dow, name in [(0, "Monday"), (4, "Friday")]:
        sub = df[df["dow"] == dow]
        if sub.empty:
            continue
        print(f"\n  {name}:")
        print(f"  {'Hour (UTC)':>10} {'Trades':>8} {'Win%':>8} {'AvgP&L':>10}")
        print("  " + "-" * 42)

        for hour in sorted(sub["entry_hour"].unique()):
            h_sub = sub[sub["entry_hour"] == hour]
            n     = len(h_sub)
            wr    = h_sub["win"].mean() * 100
            avg   = h_sub["pnl_gbp"].mean()
            print(f"  {hour:>10}   {n:>6}   {wr:>6.1f}%   {avg:>8.2f}")


# ---------------------------------------------------------------------------
# Analysis 4 — Mon+Fri vs Tue+Wed+Thu
# ---------------------------------------------------------------------------

print()
print(SEP)
print("ANALYSIS 4 — MON+FRI vs TUE+WED+THU")
print(SEP)

mf_df  = df[df["dow"].isin([0, 4])]
mid_df = df[df["dow"].isin([1, 2, 3])]

def group_stats(g: pd.DataFrame, label: str) -> None:
    n     = len(g)
    if n == 0:
        print(f"  {label}: no trades")
        return
    wr    = g["win"].mean() * 100
    pf    = profit_factor(g["pnl_gbp"])
    avg   = g["pnl_gbp"].mean()
    total = g["pnl_gbp"].sum()
    sh    = sharpe(g["pnl_gbp"], g["exit_time"])
    print(f"  {label:<20}: {n:>4} trades | Win%={wr:.1f}% | PF={pf:.2f} | "
          f"AvgP&L=£{avg:.2f} | TotalP&L=£{total:.2f} | Sharpe={sh:.2f}")

group_stats(mf_df,  "Mon + Fri")
group_stats(mid_df, "Tue + Wed + Thu")

pct_mf  = len(mf_df)  / len(df) * 100 if len(df) else 0
pct_mid = len(mid_df) / len(df) * 100 if len(df) else 0
print(f"\n  Mon+Fri account for {len(mf_df)}/{len(df)} trades ({pct_mf:.0f}%) "
      f"and £{mf_df['pnl_gbp'].sum():.2f} P&L "
      f"({mf_df['pnl_gbp'].sum() / df['pnl_gbp'].sum() * 100:.0f}% of total)")
print(f"  Tue-Thu account for {len(mid_df)}/{len(df)} trades ({pct_mid:.0f}%) "
      f"and £{mid_df['pnl_gbp'].sum():.2f} P&L "
      f"({mid_df['pnl_gbp'].sum() / df['pnl_gbp'].sum() * 100:.0f}% of total)")

print()
