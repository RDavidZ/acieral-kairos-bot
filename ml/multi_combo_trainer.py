"""
ml/multi_combo_trainer.py — N/T combo selection for Acieral Kairos Bot

Selects the best (N, T) labelling combination per instrument by:
  1. Taking the top-3 combos by labeled_days from the grid search results.
  2. Training a walk-forward model for each combo.
  3. Running a holdout backtest for each combo.
  4. Choosing the winner by highest holdout Sharpe ratio.

Intermediate models are cached as entry_{instrument}_N{N}_{T_str}.pkl
so the full run can be resumed after interruption.

Full run: 3 combos × 8 instruments = up to 24 training runs (~10-15 min each).
"""

import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from config import HARD_CONSTRAINTS
from ml.labeler import label_instrument
from ml.trainer import train_instrument
from backtest.engine import run_backtest

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODELS_DIR      = Path(__file__).parent / "models"
RESULTS_DIR     = Path(__file__).parent / "multi_combo_results"
LABELS_DIR      = Path(__file__).parent / "labels_train"
ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t_to_bps_str(T: float) -> str:
    """0.0005 -> 'T5bps',  0.002 -> 'T20bps'."""
    return f"T{int(round(T * 10_000))}bps"


def _load_grid(instrument: str) -> pd.DataFrame:
    """
    Load the N/T grid CSV.  Tries _grid.csv first (labeler output),
    then _grid_summary.csv (alternative naming).
    """
    for suffix in ("_grid.csv", "_grid_summary.csv"):
        path = RESULTS_DIR / f"{instrument}{suffix}"
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Grid CSV not found for {instrument} in {RESULTS_DIR}.\n"
        f"Run ml.labeler.run_grid_search('{instrument}') first."
    )


def _select_top3(grid: pd.DataFrame) -> list[tuple[int, float]]:
    """
    Return the 3 (N, T) combos with the most labeled trading days.
    Tie-break: lower N first (smaller lookahead = stricter signal).

    The grid column T_pct stores the threshold as a percentage
    (e.g. 0.05 means 0.05% = T=0.0005).  Convert back to raw float.
    """
    ranked = (
        grid
        .sort_values(["labeled_days", "N"], ascending=[False, True])
        .drop_duplicates(subset=["N", "T_pct"])
        .head(3)
    )
    return [
        (int(row["N"]), round(float(row["T_pct"]) / 100.0, 6))
        for _, row in ranked.iterrows()
    ]


def _combo_model_path(instrument: str, N: int, T: float) -> Path:
    return MODELS_DIR / f"entry_{instrument}_N{N}_{_t_to_bps_str(T)}.pkl"


def _combo_meta_path(instrument: str, N: int, T: float) -> Path:
    return MODELS_DIR / f"entry_{instrument}_N{N}_{_t_to_bps_str(T)}_meta.json"


def _load_combo_meta(instrument: str, N: int, T: float) -> dict:
    path = _combo_meta_path(instrument, N, T)
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save_combo_meta(instrument: str, N: int, T: float, meta: dict) -> None:
    with open(_combo_meta_path(instrument, N, T), "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Single-instrument combo selection
# ---------------------------------------------------------------------------

def run_multi_combo(
    instrument: str,
    _combo_counter: list | None = None,
    _total_combos: int = 0,
) -> dict:
    """
    Evaluate all top-3 combos for *instrument* and select the winner.

    Parameters
    ----------
    instrument      : OANDA instrument name
    _combo_counter  : mutable [int] used by run_all_multi_combo for progress
    _total_combos   : total combo count across all instruments (for progress)

    Returns
    -------
    dict with keys:
      instrument, combos (list of per-combo dicts), winner (best combo dict)
    """
    log.info("=== run_multi_combo(%s) ===", instrument)

    grid = _load_grid(instrument)
    top3 = _select_top3(grid)
    log.info("[%s] Top-3 combos selected: %s", instrument,
             [(n, _t_to_bps_str(t)) for n, t in top3])

    combo_results: list[dict] = []

    for N, T in top3:
        t_str = _t_to_bps_str(T)

        # Progress banner
        if _combo_counter is not None:
            _combo_counter[0] += 1
            total_str = f"/{_total_combos}" if _total_combos else ""
            log.info(
                "[%d%s] Training %s  N=%d  T=%s",
                _combo_counter[0], total_str, instrument, N, t_str,
            )
        else:
            log.info("  --- Combo  %s  N=%d  T=%s ---", instrument, N, t_str)

        result: dict = {
            "N": N,
            "T": T,
            "mean_val_auc":   None,
            "holdout_sharpe": None,
            "holdout_trades": None,
            "holdout_win_rate": None,
            "holdout_pf":     None,
            "holdout_max_dd": None,
            "holdout_pnl_gbp": None,
            "status": "pending",
        }

        try:
            # ----------------------------------------------------------
            # (a) Ensure label file exists
            # ----------------------------------------------------------
            label_path = LABELS_DIR / f"{instrument}_N{N}_{t_str}_labels.parquet"
            if not label_path.exists():
                log.info("    Labels missing — generating N=%d %s ...", N, t_str)
                label_instrument(instrument, N, T, save=True)
            else:
                log.info("    Labels OK: %s", label_path.name)

            # ----------------------------------------------------------
            # (b) Train walk-forward model (with combo cache)
            # ----------------------------------------------------------
            combo_pkl = _combo_model_path(instrument, N, T)

            if combo_pkl.exists():
                log.info("    Cached model found: %s — skipping training.", combo_pkl.name)
                cached_meta = _load_combo_meta(instrument, N, T)
                result["mean_val_auc"] = cached_meta.get("mean_val_auc")
            else:
                log.info("    Training walk-forward model ...")
                train_res = train_instrument(instrument, N, T)
                result["mean_val_auc"] = train_res["mean_val_auc"]

                # trainer.py saves to entry_{instrument}.pkl — copy to combo slot
                default_pkl = MODELS_DIR / f"entry_{instrument}.pkl"
                shutil.copy(default_pkl, combo_pkl)
                log.info("    Model cached: %s", combo_pkl.name)

                # Save combo-level sidecar metadata
                _save_combo_meta(instrument, N, T, {
                    "mean_val_auc": train_res["mean_val_auc"],
                    "std_val_auc":  train_res["std_val_auc"],
                    "n_folds":      train_res["n_folds"],
                    "best_params":  train_res["best_params"],
                })

            # ----------------------------------------------------------
            # (c) Holdout backtest
            # ----------------------------------------------------------
            log.info("    Running holdout backtest ...")
            bt = run_backtest(
                instrument, str(combo_pkl), N, T, split="holdout"
            )
            result.update({
                "holdout_sharpe":   bt["sharpe_ratio"],
                "holdout_trades":   bt["total_trades"],
                "holdout_win_rate": bt["win_rate"],
                "holdout_pf":       bt["profit_factor"],
                "holdout_max_dd":   bt["max_drawdown"],
                "holdout_pnl_gbp":  bt["total_pnl_gbp"],
                "status":           "ok",
            })

            log.info(
                "    Result: Sharpe=%.2f  trades=%d  win=%.1f%%  P&L=£%.2f",
                bt["sharpe_ratio"], bt["total_trades"],
                bt["win_rate"] * 100, bt["total_pnl_gbp"],
            )

        except Exception as exc:
            log.error(
                "    [%s] N=%d T=%s FAILED: %s",
                instrument, N, T, exc, exc_info=True,
            )
            result["status"] = f"FAILED: {exc}"

        combo_results.append(result)

    # ------------------------------------------------------------------
    # Select winner by holdout Sharpe (tiebreak: higher P&L)
    # ------------------------------------------------------------------
    ok = [r for r in combo_results
          if r["status"] == "ok" and r["holdout_sharpe"] is not None]

    if not ok:
        raise RuntimeError(
            f"[{instrument}] All combos failed — cannot select winner.\n"
            f"Errors: {[r['status'] for r in combo_results]}"
        )

    winner = max(ok, key=lambda x: (x["holdout_sharpe"], x["holdout_pnl_gbp"]))

    # ------------------------------------------------------------------
    # Save winner model as entry_{instrument}.pkl
    # ------------------------------------------------------------------
    winner_pkl = _combo_model_path(instrument, winner["N"], winner["T"])
    final_pkl  = MODELS_DIR / f"entry_{instrument}.pkl"
    shutil.copy(winner_pkl, final_pkl)
    log.info(
        "[%s] Winner: N=%d %s  Sharpe=%.2f  P&L=£%.2f -> %s",
        instrument, winner["N"], _t_to_bps_str(winner["T"]),
        winner["holdout_sharpe"], winner["holdout_pnl_gbp"], final_pkl.name,
    )

    # Re-run backtest for winner to refresh backtest/results/ files
    log.info("[%s] Saving winner backtest results ...", instrument)
    try:
        run_backtest(instrument, str(final_pkl), winner["N"], winner["T"],
                     split="holdout")
    except Exception as exc:
        log.warning("[%s] Winner backtest re-run failed: %s", instrument, exc)

    # ------------------------------------------------------------------
    # Save combo results CSV
    # ------------------------------------------------------------------
    results_df   = pd.DataFrame(combo_results)
    results_path = RESULTS_DIR / f"{instrument}_combo_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("[%s] Combo results -> %s", instrument, results_path.name)

    # ------------------------------------------------------------------
    # Update metadata.json
    # ------------------------------------------------------------------
    meta_path = MODELS_DIR / "metadata.json"
    metadata: dict = {}
    if meta_path.exists():
        with open(meta_path) as fh:
            metadata = json.load(fh)

    existing = metadata.get(instrument, {})
    metadata[instrument] = {
        **existing,
        "N":               winner["N"],
        "T":               winner["T"],
        "mean_val_auc":    winner.get("mean_val_auc"),
        "holdout_sharpe":  winner["holdout_sharpe"],
        "holdout_pnl_gbp": winner["holdout_pnl_gbp"],
        "combo_selection": "multi_combo",
    }
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return {
        "instrument": instrument,
        "combos":     combo_results,
        "winner":     winner,
    }


# ---------------------------------------------------------------------------
# Batch runner — all instruments
# ---------------------------------------------------------------------------

def run_all_multi_combo() -> pd.DataFrame:
    """
    Run run_multi_combo() for all 8 instruments sequentially.
    Prints a progress banner for each combo and a summary table on completion.
    Saves combined summary to ml/multi_combo_results/all_instruments_summary.csv.

    Returns the summary DataFrame.
    """
    log.info("=== run_all_multi_combo() starting ===")

    # First pass: load grids and count total combos for progress display
    instrument_plans: list[tuple[str, pd.DataFrame, list]] = []
    for instrument in ALL_INSTRUMENTS:
        try:
            grid = _load_grid(instrument)
            top3 = _select_top3(grid)
            instrument_plans.append((instrument, grid, top3))
        except FileNotFoundError as exc:
            log.warning("[%s] Grid CSV not found — skipping. (%s)", instrument, exc)

    total_combos  = sum(len(t) for _, _, t in instrument_plans)
    combo_counter = [0]   # mutable so run_multi_combo can increment it

    log.info(
        "Total combos to evaluate: %d across %d instruments",
        total_combos, len(instrument_plans),
    )

    summary_rows: list[dict] = []

    for instrument, _grid, _top3 in instrument_plans:
        log.info("--- Starting %s ---", instrument)
        try:
            res    = run_multi_combo(instrument, combo_counter, total_combos)
            winner = res["winner"]
            summary_rows.append({
                "instrument":      instrument,
                "winner_N":        winner["N"],
                "winner_T":        winner["T"],
                "mean_val_auc":    winner.get("mean_val_auc"),
                "holdout_sharpe":  winner["holdout_sharpe"],
                "holdout_trades":  winner["holdout_trades"],
                "holdout_win_rate": winner["holdout_win_rate"],
                "holdout_pf":      winner["holdout_pf"],
                "holdout_max_dd":  winner["holdout_max_dd"],
                "holdout_pnl_gbp": winner["holdout_pnl_gbp"],
            })
        except Exception as exc:
            log.error("[%s] run_multi_combo FAILED: %s", instrument, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        out_path = RESULTS_DIR / "all_instruments_summary.csv"
        summary_df.to_csv(out_path, index=False)
        log.info("Summary saved -> %s", out_path.name)

        header = (
            f"\n{'Instrument':<15} {'N':>4} {'T':>7} {'WF AUC':>9}"
            f" {'H-Sharpe':>10} {'H-P&L GBP':>12}"
        )
        print(header)
        print("-" * len(header.strip()))
        for _, row in summary_df.iterrows():
            auc_str = f"{row['mean_val_auc']:.4f}" if row["mean_val_auc"] else "  n/a"
            print(
                f"{row['instrument']:<15} {int(row['winner_N']):>4}"
                f" {row['winner_T']:>7.4f} {auc_str:>9}"
                f" {row['holdout_sharpe']:>10.2f}"
                f" {row['holdout_pnl_gbp']:>11.2f}"
            )

    log.info("=== run_all_multi_combo() complete ===")
    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_multi_combo()
