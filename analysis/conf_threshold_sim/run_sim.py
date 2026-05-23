"""
analysis/conf_threshold_sim/run_sim.py

Confidence Threshold Comparison
================================
Tests three entry confidence threshold strategies over a specific date window:

  curve     — CONFIDENCE_CURVE from DISCOVERED_PARAMS (current BT + old live behaviour)
  flat_0.5  — flat 0.50 across all hours/instruments
  flat_0.6  — flat 0.60 (small safety floor, compromise option)

Test window: 2026-05-19 → 2026-05-23 (week of live divergence observation)

All other parameters are identical: production entry + exit models (retrained 2026-05-22),
production trail/exit params.

Run from repo root:
  venv/Scripts/python analysis/conf_threshold_sim/run_sim.py
"""

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
SIM_ROOT  = Path(__file__).parent

sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from config import HARD_CONSTRAINTS, DISCOVERED_PARAMS
import backtest.engine_with_exit as _bte

ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
MODELS_DIR      = REPO_ROOT / "ml" / "models"

# ---------------------------------------------------------------------------
# Test window — week of live divergence observation
# ---------------------------------------------------------------------------
TEST_START = "2026-05-19"
TEST_END   = "2026-05-23"   # exclusive

# ---------------------------------------------------------------------------
# Scenarios: name → override for DISCOVERED_PARAMS["CONFIDENCE_CURVE"]
# ---------------------------------------------------------------------------

def _flat_curve(threshold: float) -> dict:
    return {
        inst: {h: threshold for h in range(24)}
        for inst in ALL_INSTRUMENTS
    }

SCENARIOS = {
    "curve":    None,               # use real CONFIDENCE_CURVE unchanged
    "flat_0.5": _flat_curve(0.50),
    "flat_0.6": _flat_curve(0.60),
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_scenario(name: str, patched_curve) -> dict[str, dict]:
    log.info("=== Scenario: %s ===", name)

    # Patch HOLDOUT_START to TEST_START so the BT engine starts from May 19
    original_holdout = _bte.HOLDOUT_START
    _bte.HOLDOUT_START = TEST_START

    # Patch confidence curve if needed
    original_curve = _bte.DISCOVERED_PARAMS.get("CONFIDENCE_CURVE")
    if patched_curve is not None:
        _bte.DISCOVERED_PARAMS["CONFIDENCE_CURVE"] = patched_curve

    results = {}
    for instrument in ALL_INSTRUMENTS:
        entry_model = MODELS_DIR / f"entry_{instrument}.pkl"
        exit_model  = MODELS_DIR / f"exit_{instrument}.pkl"

        if not entry_model.exists() or not exit_model.exists():
            log.warning("[%s] model missing — skipping", instrument)
            results[instrument] = {"error": "model missing"}
            continue

        params     = DISCOVERED_PARAMS["INSTRUMENT_PARAMS"][instrument]
        trail_mult = float(params["trail_atr_mult"])
        exit_thr   = float(params["exit_threshold"])

        try:
            m = _bte.run_backtest_with_exit(
                instrument=instrument,
                entry_model_path=str(entry_model),
                exit_model_path=str(exit_model),
                trail_atr_mult=trail_mult,
                exit_threshold=exit_thr,
                split="holdout",
                end_date=TEST_END,
                _save=False,
            )
            results[instrument] = m
            log.info(
                "[%s] trades=%d  win=%.1f%%  pnl=£%.2f  sharpe=%.2f  dd=%.1f%%",
                instrument,
                m["total_trades"], m["win_rate"] * 100,
                m["total_pnl_gbp"], m["sharpe_ratio"],
                m["max_drawdown"] * 100,
            )
        except Exception as exc:
            log.error("[%s] FAILED: %s", instrument, exc, exc_info=True)
            results[instrument] = {"error": str(exc)}

    # Restore
    _bte.HOLDOUT_START = original_holdout
    if patched_curve is not None:
        _bte.DISCOVERED_PARAMS["CONFIDENCE_CURVE"] = original_curve

    return results


def generate_report(all_results: dict[str, dict[str, dict]]) -> str:
    scenario_names = list(SCENARIOS.keys())
    lines = []
    lines.append("# Confidence Threshold Comparison — May 19–22 2026")
    lines.append("")
    lines.append(f"**Test window:** {TEST_START} to {TEST_END}  |  **Models:** retrained 2026-05-22 (corrected HTF features)")
    lines.append("")
    lines.append("## Per-Instrument Results")
    lines.append("")

    for instrument in ALL_INSTRUMENTS:
        lines.append(f"### {instrument}")
        lines.append("")
        lines.append("| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |")
        lines.append("|----------|--------|------|-------|--------|--------|----|")
        for sc in scenario_names:
            m = all_results[sc].get(instrument, {"error": "not run"})
            if "error" in m:
                lines.append(f"| {sc} | — | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {sc} "
                    f"| {m['total_trades']} "
                    f"| {m['win_rate']*100:.1f}% "
                    f"| £{m['total_pnl_gbp']:.2f} "
                    f"| {m['sharpe_ratio']:.2f} "
                    f"| {m['max_drawdown']*100:.1f}% "
                    f"| {m.get('profit_factor', 0):.2f} |"
                )
        lines.append("")

    lines.append("## Portfolio Summary")
    lines.append("")
    lines.append("| Scenario | Total P&L £ | Avg Win% | Avg Sharpe | Avg Max DD | Total Trades |")
    lines.append("|----------|-------------|----------|------------|------------|--------------|")

    for sc in scenario_names:
        pnls, wins, sharpes, dds, trades = [], [], [], [], []
        for instrument in ALL_INSTRUMENTS:
            m = all_results[sc].get(instrument, {})
            if "error" not in m and m:
                pnls.append(m.get("total_pnl_gbp", 0))
                wins.append(m.get("win_rate", 0))
                sharpes.append(m.get("sharpe_ratio", 0))
                dds.append(m.get("max_drawdown", 0))
                trades.append(m.get("total_trades", 0))
        if pnls:
            lines.append(
                f"| {sc} "
                f"| £{sum(pnls):.2f} "
                f"| {sum(wins)/len(wins)*100:.1f}% "
                f"| {sum(sharpes)/len(sharpes):.2f} "
                f"| {sum(dds)/len(dds)*100:.1f}% "
                f"| {sum(trades)} |"
            )
        else:
            lines.append(f"| {sc} | — | — | — | — | 0 |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `curve`: CONFIDENCE_CURVE from DISCOVERED_PARAMS (per-hour, per-instrument ~0.74–0.83+)")
    lines.append("- `flat_0.5`: flat 0.50 — more permissive, no hour-based filtering")
    lines.append("- `flat_0.6`: flat 0.60 — small safety floor")
    lines.append("- Feature parquets: rebuilt 2026-05-22 with corrected H4/D1 merge-shift")
    lines.append("- Only the entry confidence gate differs between scenarios")

    return "\n".join(lines)


if __name__ == "__main__":
    SIM_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for scenario_name, patched_curve in SCENARIOS.items():
        all_results[scenario_name] = run_scenario(scenario_name, patched_curve)

    results_path = SIM_ROOT / "results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    report = generate_report(all_results)
    report_path = SIM_ROOT / "report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    print()
    print(report)
