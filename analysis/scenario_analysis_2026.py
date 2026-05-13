"""
analysis/scenario_analysis_2026.py -- 2026 out-of-sample scenario analysis

Read-only: replays trades from backtest/results_2026/ under 4 equity/risk scenarios.
Runs two modes:
  - fixed:    position sizing uses starting equity throughout (no compounding)
  - compound: position sizing uses running equity (original behaviour)

Results saved to:
  analysis/results_2026_fixed/
  analysis/results_2026_compound/
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR  = Path(__file__).parent.parent / "backtest" / "results_2026"
OUT_FIXED    = Path(__file__).parent / "results_2026_fixed"
OUT_COMPOUND = Path(__file__).parent / "results_2026_compound"

QUOTE_TYPE = {
    "EUR_USD":    "usd_quote",
    "GBP_USD":    "usd_quote",
    "AUD_USD":    "usd_quote",
    "USD_JPY":    "usd_base",
    "USD_CHF":    "usd_base",
    "SPX500_USD": "usd_index",
    "NAS100_USD": "usd_index",
    "DE30_EUR":   "eur_index",
}

GBPUSD_RATE = 1.28
GBPEUR_RATE = 1.17

UNIT_CAPS = {
    "SPX500_USD": 5,
    "NAS100_USD": 2,
    "DE30_EUR":   2,
}

SCENARIO_KEYS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Risk tier
# ---------------------------------------------------------------------------

def get_risk_pct(equity: float) -> float:
    if equity < 20_000:
        return 0.0050
    elif equity < 50_000:
        return 0.0040
    elif equity < 100_000:
        return 0.0030
    else:
        return 0.0020


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def compute_units(instrument: str, sizing_equity: float, entry_price: float,
                  sl_price: float, apply_caps: bool) -> tuple[int, float]:
    """Return (units, risk_gbp). sizing_equity is the equity used for sizing only."""
    risk_pct    = get_risk_pct(sizing_equity)
    risk_gbp    = sizing_equity * risk_pct
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return 0, risk_gbp

    qt = QUOTE_TYPE[instrument]

    if qt == "usd_quote":
        units_f = risk_gbp / sl_distance
    elif qt == "usd_base":
        units_f = (risk_gbp * GBPUSD_RATE) / (sl_distance / entry_price)
    elif qt == "usd_index":
        units_f = (risk_gbp * GBPUSD_RATE) / sl_distance
    elif qt == "eur_index":
        units_f = (risk_gbp * GBPEUR_RATE) / sl_distance
    else:
        return 0, risk_gbp

    units = max(1, round(units_f))

    if apply_caps and instrument in UNIT_CAPS:
        units = min(units, UNIT_CAPS[instrument])

    return units, risk_gbp


# ---------------------------------------------------------------------------
# P&L computation
# ---------------------------------------------------------------------------

def compute_pnl(instrument: str, direction: str, entry_price: float,
                exit_price: float, units: int, risk_gbp: float) -> float:
    direction_sign = 1 if direction == "LONG" else -1
    qt = QUOTE_TYPE[instrument]

    if qt == "usd_quote":
        pnl = (exit_price - entry_price) * units / GBPUSD_RATE * direction_sign
    elif qt == "usd_base":
        pnl = ((exit_price - entry_price) / exit_price) * units / GBPUSD_RATE * direction_sign
    elif qt == "usd_index":
        pnl = (exit_price - entry_price) * units / GBPUSD_RATE * direction_sign
    elif qt == "eur_index":
        pnl = (exit_price - entry_price) * units / GBPEUR_RATE * direction_sign
    else:
        pnl = 0.0

    if pnl < -risk_gbp:
        pnl = -risk_gbp

    return round(pnl, 4)


# ---------------------------------------------------------------------------
# Load and merge all trades
# ---------------------------------------------------------------------------

def load_all_trades() -> pd.DataFrame:
    frames = []
    for instrument in QUOTE_TYPE:
        path = RESULTS_DIR / f"{instrument}_trades.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found -- skipping")
            continue
        df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
        df["instrument"] = instrument
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("entry_time").reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Run one scenario
# ---------------------------------------------------------------------------

def run_scenario(trades: pd.DataFrame, start_equity: float,
                 apply_caps: bool, label: str, fixed_sizing: bool) -> dict:
    """
    fixed_sizing=True  : units always sized from start_equity (no compounding)
    fixed_sizing=False : units sized from running equity (compounding)
    """
    running_equity = start_equity
    equity_curve   = []
    daily_pnl: dict[str, float] = {}
    results        = []
    sizing_failures = []
    risk_breaches   = []

    for _, row in trades.iterrows():
        instrument  = row["instrument"]
        entry_price = float(row["entry_price"])
        exit_price  = float(row["exit_price"])
        sl_price    = float(row["sl_price"])
        direction   = str(row["direction"])

        sizing_equity = start_equity if fixed_sizing else running_equity

        units, risk_gbp = compute_units(
            instrument, sizing_equity, entry_price, sl_price, apply_caps
        )

        if units == 0:
            sizing_failures.append({
                "instrument": instrument,
                "entry_time": row["entry_time"],
                "sizing_equity": round(sizing_equity, 2),
            })
            continue

        pnl = compute_pnl(instrument, direction, entry_price, exit_price, units, risk_gbp)

        if pnl < 0 and abs(pnl) > risk_gbp * 1.05:
            risk_breaches.append({
                "instrument": instrument,
                "entry_time": str(row["entry_time"]),
                "pnl":        round(pnl, 2),
                "risk_gbp":   round(risk_gbp, 2),
                "excess_pct": round((abs(pnl) / risk_gbp - 1) * 100, 1),
            })

        running_equity += pnl
        equity_curve.append(running_equity)

        exit_date = str(row["exit_time"])[:10] if pd.notna(row["exit_time"]) else str(row["entry_time"])[:10]
        daily_pnl[exit_date] = daily_pnl.get(exit_date, 0.0) + pnl

        results.append({
            "instrument":   instrument,
            "entry_time":   row["entry_time"],
            "exit_time":    row["exit_time"],
            "direction":    direction,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "sl_price":     sl_price,
            "units":        units,
            "risk_gbp":     round(risk_gbp, 4),
            "pnl":          pnl,
            "exit_reason":  row["exit_reason"],
            "equity_after": round(running_equity, 4),
        })

    res_df = pd.DataFrame(results)

    total_trades = len(res_df)
    if total_trades == 0:
        return {"label": label, "error": "no trades"}

    total_pnl     = res_df["pnl"].sum()
    wins          = (res_df["pnl"] > 0).sum()
    win_rate      = wins / total_trades * 100
    gross_profit  = res_df[res_df["pnl"] > 0]["pnl"].sum()
    gross_loss    = abs(res_df[res_df["pnl"] < 0]["pnl"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    daily_series = pd.Series(list(daily_pnl.values()))
    sharpe = (daily_series.mean() / daily_series.std() * np.sqrt(252)) if daily_series.std() > 0 else 0.0

    equity_arr = np.array(equity_curve)
    peak       = np.maximum.accumulate(equity_arr)
    drawdowns  = (equity_arr - peak) / peak * 100
    max_dd     = drawdowns.min()

    inst_breakdown = {}
    for inst, grp in res_df.groupby("instrument"):
        inst_breakdown[inst] = {
            "trades":    len(grp),
            "win_rate":  round((grp["pnl"] > 0).mean() * 100, 1),
            "total_pnl": round(grp["pnl"].sum(), 2),
            "avg_risk":  round(grp["risk_gbp"].mean(), 2),
            "max_loss":  round(grp["pnl"].min(), 2),
            "max_win":   round(grp["pnl"].max(), 2),
        }

    return {
        "label":            label,
        "start_equity":     start_equity,
        "end_equity":       round(running_equity, 2),
        "total_return_pct": round((running_equity - start_equity) / start_equity * 100, 2),
        "total_trades":     total_trades,
        "win_rate":         round(win_rate, 1),
        "profit_factor":    round(profit_factor, 3),
        "sharpe":           round(sharpe, 3),
        "max_dd_pct":       round(max_dd, 2),
        "total_pnl":        round(total_pnl, 2),
        "inst_breakdown":   inst_breakdown,
        "sizing_failures":  sizing_failures,
        "risk_breaches":    risk_breaches,
        "res_df":           res_df,
        "daily_pnl":        daily_pnl,
    }


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(scenarios: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-scenario trade-level CSVs
    for s in scenarios:
        if "error" in s:
            continue
        key = s["label"].split(" ")[1]   # "A", "B", "C", "D"
        s["res_df"].to_csv(out_dir / f"scenario_{key}_trades.csv", index=False)

    # 2. Scenario summary CSV
    summary_rows = []
    for s in scenarios:
        if "error" in s:
            continue
        summary_rows.append({
            "scenario":         s["label"].split(" ")[1],
            "label":            s["label"],
            "start_equity":     s["start_equity"],
            "end_equity":       s["end_equity"],
            "total_return_pct": s["total_return_pct"],
            "total_pnl":        s["total_pnl"],
            "total_trades":     s["total_trades"],
            "win_rate":         s["win_rate"],
            "profit_factor":    s["profit_factor"],
            "sharpe":           s["sharpe"],
            "max_dd_pct":       s["max_dd_pct"],
            "sizing_failures":  len(s["sizing_failures"]),
            "risk_breaches":    len(s["risk_breaches"]),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "scenario_summary.csv", index=False)

    # 3. Per-instrument breakdown CSV (all scenarios flat)
    inst_rows = []
    for s in scenarios:
        if "error" in s:
            continue
        key = s["label"].split(" ")[1]
        for inst, b in s["inst_breakdown"].items():
            inst_rows.append({
                "scenario":   key,
                "instrument": inst,
                **b,
            })
    pd.DataFrame(inst_rows).to_csv(out_dir / "per_instrument_summary.csv", index=False)

    # 4. Daily P&L CSVs per scenario
    for s in scenarios:
        if "error" in s:
            continue
        key = s["label"].split(" ")[1]
        daily_df = pd.DataFrame([
            {"date": d, "pnl": round(v, 4)}
            for d, v in sorted(s["daily_pnl"].items())
        ])
        daily_df.to_csv(out_dir / f"scenario_{key}_daily_pnl.csv", index=False)

    print(f"  Saved to {out_dir}")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_scenario_summary(s: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {s['label']}")
    print(f"{'='*60}")
    if "error" in s:
        print(f"  ERROR: {s['error']}")
        return
    print(f"  Starting equity  : GBP {s['start_equity']:>12,.2f}")
    print(f"  Ending equity    : GBP {s['end_equity']:>12,.2f}")
    print(f"  Total return     : {s['total_return_pct']:>+8.2f}%")
    print(f"  Total P&L        : GBP {s['total_pnl']:>+12.2f}")
    print(f"  Total trades     : {s['total_trades']:>8}")
    print(f"  Win rate         : {s['win_rate']:>8.1f}%")
    print(f"  Profit factor    : {s['profit_factor']:>8.3f}")
    print(f"  Sharpe (daily)   : {s['sharpe']:>8.3f}")
    print(f"  Max drawdown     : {s['max_dd_pct']:>+8.2f}%")

    print(f"\n  {'Instrument':<14} {'Trades':>6} {'Win%':>6} {'P&L GBP':>12} {'AvgRisk':>8} {'MaxLoss':>10} {'MaxWin':>10}")
    print(f"  {'-'*72}")
    for inst, b in sorted(s["inst_breakdown"].items()):
        print(f"  {inst:<14} {b['trades']:>6} {b['win_rate']:>5.1f}% "
              f"{b['total_pnl']:>+12.2f} {b['avg_risk']:>8.2f} "
              f"{b['max_loss']:>+10.2f} {b['max_win']:>+10.2f}")

    if s["sizing_failures"]:
        print(f"\n  SIZING FAILURES ({len(s['sizing_failures'])}):")
        for f in s["sizing_failures"]:
            print(f"    {f['instrument']} @ {f['entry_time']} | sizing_equity=GBP {f['sizing_equity']:,.2f}")
    else:
        print(f"\n  Sizing failures  : none")

    if s["risk_breaches"]:
        print(f"\n  RISK BREACHES ({len(s['risk_breaches'])}):")
        for b in s["risk_breaches"]:
            print(f"    {b['instrument']} @ {b['entry_time']} | "
                  f"loss=GBP {b['pnl']:.2f} vs risk=GBP {b['risk_gbp']:.2f} (+{b['excess_pct']}%)")
    else:
        print(f"  Risk breaches    : none")


def print_comparison_table(scenarios: list[dict]) -> None:
    print(f"\n\n{'='*88}")
    print("  SCENARIO COMPARISON")
    print(f"{'='*88}")

    col_w = 16
    labels = [s["label"].split("--")[0].strip() for s in scenarios]
    header = f"  {'Metric':<26}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print(f"  {'-'*(26 + col_w * len(scenarios))}")

    def row(name, key, fmt="{:.2f}", prefix=""):
        vals = []
        for s in scenarios:
            v = s.get(key, "N/A")
            vals.append(f"{prefix}{fmt.format(v)}" if v != "N/A" else "N/A")
        print(f"  {name:<26}" + "".join(f"{v:>{col_w}}" for v in vals))

    row("Start equity GBP",    "start_equity",     "{:,.0f}",  "")
    row("End equity GBP",      "end_equity",       "{:,.0f}",  "")
    row("Total return %",      "total_return_pct", "{:+.2f}",  "")
    row("Total P&L GBP",       "total_pnl",        "{:+,.2f}", "")
    row("Total trades",        "total_trades",     "{:d}",     "")
    row("Win rate %",          "win_rate",         "{:.1f}",   "")
    row("Profit factor",       "profit_factor",    "{:.3f}",   "")
    row("Sharpe (daily)",      "sharpe",           "{:.3f}",   "")
    row("Max drawdown %",      "max_dd_pct",       "{:.2f}",   "")

    sf_vals = [str(len(s.get("sizing_failures", []))) for s in scenarios]
    rb_vals = [str(len(s.get("risk_breaches",   []))) for s in scenarios]
    print(f"  {'Sizing failures':<26}" + "".join(f"{v:>{col_w}}" for v in sf_vals))
    print(f"  {'Risk breaches':<26}"   + "".join(f"{v:>{col_w}}" for v in rb_vals))

    print(f"{'='*88}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trades = load_all_trades()
    print(f"Loaded {len(trades)} trades across {trades['instrument'].nunique()} instruments")
    print(f"Date range: {trades['entry_time'].min().date()} to {trades['entry_time'].max().date()}")

    scenario_defs = [
        (50_000,  False, "SCENARIO A -- GBP 50k, tiered risk, no caps"),
        (50_000,  True,  "SCENARIO B -- GBP 50k, tiered risk, unit caps"),
        (100_000, False, "SCENARIO C -- GBP 100k, tiered risk, no caps"),
        (100_000, True,  "SCENARIO D -- GBP 100k, tiered risk, unit caps"),
    ]

    # ---- FIXED SIZING ----
    print("\n" + "#"*70)
    print("  MODE: FIXED SIZING (sizing equity = starting equity, no compounding)")
    print("#"*70)

    fixed_scenarios = [
        run_scenario(trades, eq, caps, label, fixed_sizing=True)
        for eq, caps, label in scenario_defs
    ]
    for s in fixed_scenarios:
        print_scenario_summary(s)
    print_comparison_table(fixed_scenarios)
    save_results(fixed_scenarios, OUT_FIXED)

    # ---- COMPOUNDING ----
    print("\n" + "#"*70)
    print("  MODE: COMPOUNDING (sizing equity = running equity)")
    print("#"*70)

    compound_scenarios = [
        run_scenario(trades, eq, caps, label, fixed_sizing=False)
        for eq, caps, label in scenario_defs
    ]
    for s in compound_scenarios:
        print_scenario_summary(s)
    print_comparison_table(compound_scenarios)
    save_results(compound_scenarios, OUT_COMPOUND)


if __name__ == "__main__":
    print("Acieral Kairos -- 2026 Out-of-Sample Scenario Analysis")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"GBPUSD={GBPUSD_RATE}  GBPEUR={GBPEUR_RATE}")
    main()
