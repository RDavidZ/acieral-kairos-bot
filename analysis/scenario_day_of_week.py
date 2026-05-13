"""
analysis/scenario_day_of_week.py

Compares 4 day-of-week scenarios across all 8 instruments using
2026 backtest results from backtest/results_2026/.

Scenarios
---------
1. Full week        — all trades (baseline)
2. No Monday        — exclude entry_time weekday == 0
3. No Friday        — exclude entry_time weekday == 4
4. No Mon or Fri    — exclude both

Metrics per scenario (per-instrument and combined)
---------------------------------------------------
Trades, Win%, PF, Sharpe (annualised), Max DD%, P&L GBP, Trades removed
"""

import math
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "backtest" / "results_2026"
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF",
    "SPX500_USD", "NAS100_USD", "DE30_EUR",
]

SCENARIOS = {
    "Full week":       lambda df: df,
    "No Monday":       lambda df: df[df["_dow"] != 0],
    "No Friday":       lambda df: df[df["_dow"] != 4],
    "No Mon or Fri":   lambda df: df[~df["_dow"].isin([0, 4])],
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def sharpe(pnl_series: pd.Series) -> float:
    """Annualised Sharpe from a daily P&L series."""
    if len(pnl_series) < 2:
        return 0.0
    std = pnl_series.std()
    if std == 0:
        return 0.0
    return float(pnl_series.mean() / std * math.sqrt(252))


def max_drawdown(pnl_series: pd.Series) -> float:
    """Max drawdown as a fraction of cumulative peak."""
    cum = pnl_series.cumsum()
    peak = cum.cummax()
    dd = (peak - cum) / peak.replace(0, float("nan"))
    val = dd.max()
    return float(val) if not math.isnan(val) else 0.0


def compute_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return dict(trades=0, win_pct=0.0, pf=0.0, sharpe=0.0,
                    max_dd=0.0, pnl=0.0)
    wins = (df["pnl_gbp"] > 0).sum()
    gross_profit = df.loc[df["pnl_gbp"] > 0, "pnl_gbp"].sum()
    gross_loss   = df.loc[df["pnl_gbp"] < 0, "pnl_gbp"].abs().sum()
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    # Daily P&L for Sharpe and DD
    daily = (
        df.groupby(df["exit_time"].dt.date)["pnl_gbp"]
        .sum()
        .sort_index()
    )
    sh  = sharpe(daily)
    mdd = max_drawdown(daily) * 100.0

    return dict(
        trades   = n,
        win_pct  = wins / n * 100.0,
        pf       = pf,
        sharpe   = sh,
        max_dd   = mdd,
        pnl      = df["pnl_gbp"].sum(),
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

all_dfs: dict[str, pd.DataFrame] = {}
for instr in INSTRUMENTS:
    path = RESULTS_DIR / f"{instr}_trades.csv"
    if not path.exists():
        print(f"WARNING: {path} not found — skipping {instr}")
        continue
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df["_dow"]       = df["entry_time"].dt.dayofweek   # 0=Mon … 6=Sun
    df["instrument"] = instr
    all_dfs[instr]   = df

# ---------------------------------------------------------------------------
# Per-instrument results
# ---------------------------------------------------------------------------

per_instrument_rows = []
baseline_trades: dict[str, int] = {}

for instr, df in all_dfs.items():
    baseline_n = len(df)
    baseline_trades[instr] = baseline_n

    for scen_name, scen_fn in SCENARIOS.items():
        filtered = scen_fn(df)
        m = compute_metrics(filtered)
        removed    = baseline_n - m["trades"]
        removed_pct = removed / baseline_n * 100.0 if baseline_n > 0 else 0.0
        per_instrument_rows.append(dict(
            instrument   = instr,
            scenario     = scen_name,
            trades       = m["trades"],
            removed      = removed,
            removed_pct  = removed_pct,
            win_pct      = m["win_pct"],
            pf           = m["pf"],
            sharpe       = m["sharpe"],
            max_dd       = m["max_dd"],
            pnl          = m["pnl"],
        ))

per_df = pd.DataFrame(per_instrument_rows)

# ---------------------------------------------------------------------------
# Combined results (pool all instruments, then compute metrics)
# ---------------------------------------------------------------------------

combined_all = pd.concat(all_dfs.values(), ignore_index=True)
baseline_total = len(combined_all)

combined_rows = []
for scen_name, scen_fn in SCENARIOS.items():
    filtered = scen_fn(combined_all)
    m = compute_metrics(filtered)
    removed     = baseline_total - m["trades"]
    removed_pct = removed / baseline_total * 100.0 if baseline_total > 0 else 0.0
    combined_rows.append(dict(
        scenario    = scen_name,
        trades      = m["trades"],
        removed     = removed,
        removed_pct = removed_pct,
        win_pct     = m["win_pct"],
        pf          = m["pf"],
        sharpe      = m["sharpe"],
        max_dd      = m["max_dd"],
        pnl         = m["pnl"],
    ))

combined_df = pd.DataFrame(combined_rows)

# ---------------------------------------------------------------------------
# Print Table 1 — Per-instrument
# ---------------------------------------------------------------------------

print("=" * 100)
print("TABLE 1 — PER-INSTRUMENT METRICS BY SCENARIO")
print("=" * 100)

header = f"{'Instrument':<14} {'Scenario':<16} {'Trades':>6} {'Removed':>9} {'Win%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'P&L GBP':>10}"
print(header)
print("-" * 100)

for instr in INSTRUMENTS:
    if instr not in all_dfs:
        continue
    subset = per_df[per_df["instrument"] == instr]
    for _, row in subset.iterrows():
        removed_str = f"{int(row['removed'])} ({row['removed_pct']:.0f}%)"
        print(
            f"{row['instrument']:<14} {row['scenario']:<16} "
            f"{int(row['trades']):>6} {removed_str:>9} "
            f"{row['win_pct']:>6.1f} {row['pf']:>6.2f} "
            f"{row['sharpe']:>7.2f} {row['max_dd']:>7.1f} "
            f"{row['pnl']:>10.2f}"
        )
    print()

# ---------------------------------------------------------------------------
# Print Table 2 — Combined
# ---------------------------------------------------------------------------

print("=" * 100)
print("TABLE 2 — COMBINED (ALL 8 INSTRUMENTS) BY SCENARIO")
print("=" * 100)

header2 = f"{'Scenario':<16} {'Trades':>6} {'Removed':>12} {'Win%':>6} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>7} {'P&L GBP':>10}"
print(header2)
print("-" * 100)

for _, row in combined_df.iterrows():
    removed_str = f"{int(row['removed'])} ({row['removed_pct']:.0f}%)"
    print(
        f"{row['scenario']:<16} {int(row['trades']):>6} {removed_str:>12} "
        f"{row['win_pct']:>6.1f} {row['pf']:>6.2f} "
        f"{row['sharpe']:>7.2f} {row['max_dd']:>7.1f} "
        f"{row['pnl']:>10.2f}"
    )

# ---------------------------------------------------------------------------
# Best scenario analysis
# ---------------------------------------------------------------------------

print()
print("=" * 100)
print("SCENARIO ANALYSIS")
print("=" * 100)

best_sharpe_row = combined_df.loc[combined_df["sharpe"].idxmax()]
best_pnl_row    = combined_df.loc[combined_df["pnl"].idxmax()]

print(f"\nBest combined Sharpe : {best_sharpe_row['scenario']}  "
      f"(Sharpe={best_sharpe_row['sharpe']:.2f}, P&L=£{best_sharpe_row['pnl']:.2f})")
print(f"Best combined P&L    : {best_pnl_row['scenario']}  "
      f"(P&L=£{best_pnl_row['pnl']:.2f}, Sharpe={best_pnl_row['sharpe']:.2f})")

# ---------------------------------------------------------------------------
# Per-instrument Sharpe impact
# ---------------------------------------------------------------------------

print()
print("PER-INSTRUMENT SHARPE IMPACT vs FULL WEEK BASELINE")
print("-" * 100)
print(f"{'Instrument':<14} {'Baseline':>8} {'No Mon':>8} {'No Fri':>8} {'No Mon+Fri':>10}  Best scenario")
print("-" * 100)

for instr in INSTRUMENTS:
    if instr not in all_dfs:
        continue
    subset = per_df[per_df["instrument"] == instr].set_index("scenario")
    base  = subset.loc["Full week",     "sharpe"]
    no_m  = subset.loc["No Monday",     "sharpe"]
    no_f  = subset.loc["No Friday",     "sharpe"]
    no_mf = subset.loc["No Mon or Fri", "sharpe"]

    def delta(v):
        d = v - base
        return f"{v:.2f} ({'+' if d > 0 else '-'}{abs(d):.2f})"

    best_scen = max(
        [("Full week", base), ("No Monday", no_m),
         ("No Friday", no_f), ("No Mon or Fri", no_mf)],
        key=lambda x: x[1]
    )[0]

    print(f"{instr:<14} {base:>8.2f} {delta(no_m):>14} {delta(no_f):>14} {delta(no_mf):>16}  {best_scen}")

print()
