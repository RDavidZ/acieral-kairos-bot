"""
backtest/run_holdout_curve.py

Re-run the 2024-2025 holdout backtest using:
  - The CONFIDENCE_CURVE entry threshold (already in engine_with_exit.py)
  - Winner params from results_exit/winner_params.json (same as original holdout)
  - Date range: 2024-01-01 to 2025-12-31 (exclusive of 2026)
  - Output: backtest/results_holdout_curve/

Does NOT run the param sweep again — uses existing winners.
Does NOT modify config, models, or live files.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine_with_exit import run_backtest_with_exit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR       = Path(__file__).parent.parent / "ml" / "models"
RESULTS_IN_DIR   = Path(__file__).parent / "results_exit"
RESULTS_OUT_DIR  = Path(__file__).parent / "results_holdout_curve"
WINNER_PATH      = RESULTS_IN_DIR / "winner_params.json"

END_DATE = "2026-01-01"   # exclusive — covers 2024-01-01 to 2025-12-31

INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF",
    "SPX500_USD", "NAS100_USD", "DE30_EUR",
]

SEP = "=" * 80


def main():
    if not WINNER_PATH.exists():
        log.error("winner_params.json not found at %s", WINNER_PATH)
        sys.exit(1)

    with open(WINNER_PATH) as fh:
        winner_params = json.load(fh)

    summary_rows = []

    for instrument in INSTRUMENTS:
        if instrument not in winner_params:
            log.warning("[%s] No winner params — skipping", instrument)
            continue

        wp = winner_params[instrument]
        trail = float(wp["trail_atr_mult"])
        thr   = float(wp["exit_threshold"])

        entry_model = MODELS_DIR / f"entry_{instrument}.pkl"
        exit_model  = MODELS_DIR / f"exit_{instrument}.pkl"

        if not entry_model.exists() or not exit_model.exists():
            log.warning("[%s] Model(s) missing — skipping", instrument)
            continue

        log.info("[%s] Running holdout curve backtest trail=%.1f thr=%.2f", instrument, trail, thr)
        try:
            m = run_backtest_with_exit(
                instrument,
                str(entry_model),
                str(exit_model),
                trail_atr_mult=trail,
                exit_threshold=thr,
                split="holdout",
                _save=True,
                end_date=END_DATE,
                out_dir=RESULTS_OUT_DIR,
            )
            summary_rows.append({
                "instrument":    instrument,
                "trail":         trail,
                "thr":           thr,
                **{k: m[k] for k in [
                    "total_trades", "win_rate", "profit_factor",
                    "sharpe_ratio", "max_drawdown", "total_pnl_gbp",
                ]},
            })
            log.info(
                "[%s] Done | trades=%d win=%.1f%% sharpe=%.2f pnl=£%.2f",
                instrument, m["total_trades"], m["win_rate"] * 100,
                m["sharpe_ratio"], m["total_pnl_gbp"],
            )
        except Exception as exc:
            log.error("[%s] FAILED: %s", instrument, exc, exc_info=True)

    if not summary_rows:
        log.error("No instruments completed successfully.")
        sys.exit(1)

    # Print summary
    print("\n" + SEP)
    print("HOLDOUT CURVE RESULTS: 2024-01-01 to 2025-12-31")
    print(SEP)
    hdr = (f"{'Instrument':<14} {'Trades':>6} {'Win%':>6} {'Sharpe':>7} "
           f"{'PF':>6} {'MaxDD%':>7} {'P&L GBP':>10}")
    print(hdr)
    print("-" * 65)
    total_trades = 0; total_pnl = 0.0
    for r in summary_rows:
        total_trades += r["total_trades"]
        total_pnl    += r["total_pnl_gbp"]
        print(
            f"{r['instrument']:<14} {r['total_trades']:>6} "
            f"{r['win_rate']*100:>6.1f} {r['sharpe_ratio']:>7.2f} "
            f"{r['profit_factor']:>6.2f} {r['max_drawdown']*100:>6.2f}% "
            f"£{r['total_pnl_gbp']:>9.2f}"
        )
    print("-" * 65)
    print(f"{'TOTAL':<14} {total_trades:>6} {'':>6} {'':>7} {'':>6} {'':>7} £{total_pnl:>9.2f}")
    print()


if __name__ == "__main__":
    main()
