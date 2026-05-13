"""
ml/config_generator.py — DISCOVERED_PARAMS writer for Acieral Kairos Bot

Reads backtest results, sweep winner params, and trained model metadata,
then writes the DISCOVERED_PARAMS block to config.py.

This is the final step of the ML pipeline.  The execution layer reads
DISCOVERED_PARAMS at runtime — it never hardcodes strategy parameters.

Run order:
  Phase 2 completes → run generate_config() → config.py updated →
  Phase 3 (execution layer) can start.
"""

import json
import logging
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import HARD_CONSTRAINTS
from ml.labeler import FEATURE_COLS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR          = Path(__file__).parent.parent
CONFIG_PATH       = ROOT_DIR / "config.py"
MODELS_DIR        = Path(__file__).parent / "models"
LABELS_TRAIN_DIR  = Path(__file__).parent / "labels_train"
CACHE_DIR         = ROOT_DIR / "data" / "cache"
RESULTS_EXIT_DIR  = ROOT_DIR / "backtest" / "results_exit"
RESULTS_DIR       = ROOT_DIR / "backtest" / "results"

# ---------------------------------------------------------------------------
# Constants (filters for ACTIVE_INSTRUMENTS)
# ---------------------------------------------------------------------------

MIN_SHARPE     = 1.5
MAX_DRAWDOWN   = 0.15
MIN_TRADES     = 50

ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
HOLDOUT_START   = HARD_CONSTRAINTS["HOLDOUT_START"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _bps_str(T: float) -> str:
    return f"T{int(round(T * 10_000))}bps"


def _ensure_winner_params(entry_meta: dict) -> dict:
    """
    Load winner_params.json if it exists.  Otherwise, derive winners per
    instrument from their param_sweep.csv (highest Sharpe row), run the
    final backtest with winner params to populate metrics.json, then save
    winner_params.json.
    """
    winner_path = RESULTS_EXIT_DIR / "winner_params.json"

    if winner_path.exists():
        log.info("Loaded winner_params.json")
        return _load_json(winner_path)

    log.info("winner_params.json not found — deriving from sweep CSVs ...")

    from backtest.engine_with_exit import run_backtest_with_exit

    winner_params: dict = {}

    for instrument in ALL_INSTRUMENTS:
        sweep_csv = RESULTS_EXIT_DIR / f"{instrument}_param_sweep.csv"
        entry_model = MODELS_DIR / f"entry_{instrument}.pkl"
        exit_model  = MODELS_DIR / f"exit_{instrument}.pkl"

        if not sweep_csv.exists():
            log.info("[%s] No sweep CSV — skipping.", instrument)
            continue
        if not entry_model.exists() or not exit_model.exists():
            log.info("[%s] Models missing — skipping.", instrument)
            continue

        sweep = pd.read_csv(sweep_csv)
        if sweep.empty:
            log.warning("[%s] Sweep CSV is empty — skipping.", instrument)
            continue

        best = sweep.sort_values("sharpe", ascending=False).iloc[0]
        trail = float(best["trail_atr_mult"])
        thr   = float(best["exit_threshold"])

        log.info(
            "[%s] Winner from sweep: trail=%.1f thr=%.2f Sharpe=%.2f",
            instrument, trail, thr, float(best["sharpe"]),
        )

        # Run final backtest with winner params → saves metrics.json
        instr_meta = entry_meta.get(instrument, {})
        N = int(instr_meta.get("N", 10))
        T = float(instr_meta.get("T", 0.002))

        run_backtest_with_exit(
            instrument, str(entry_model), str(exit_model),
            trail_atr_mult=trail,
            exit_threshold=thr,
            split="holdout",
            _save=True,
        )

        winner_params[instrument] = {
            "trail_atr_mult": trail,
            "exit_threshold": thr,
        }

    with open(winner_path, "w") as fh:
        json.dump(winner_params, fh, indent=2)
    log.info("Saved winner_params.json (%d instruments)", len(winner_params))

    return winner_params


def _compute_confidence_curve(
    instrument: str,
    entry_model,
    N: int,
    T: float,
) -> dict[int, float]:
    """
    Compute a per-hour confidence threshold from training-period data.

    For each hour_utc (0-23):
      - Find labeled bars in the training set at that hour.
      - Run model predictions; identify bars where prediction matches label.
      - mean_correct_conf = mean confidence of those correct predictions.
      - threshold = max(0.50, mean_correct_conf × 0.85)
      - If fewer than 10 samples at that hour: use the global mean threshold.

    Returns dict {hour_int: threshold_float} for all 24 hours.
    """
    bps_str   = _bps_str(T)
    label_path = LABELS_TRAIN_DIR / f"{instrument}_N{N}_{bps_str}_labels.parquet"

    if not label_path.exists():
        log.warning(
            "[%s] Training labels not found at %s — using flat 0.5 curve.",
            instrument, label_path.name,
        )
        return {h: 0.5 for h in range(24)}

    df = pd.read_parquet(label_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["hour_utc"] = df["time"].dt.hour

    # Only consider labeled bars (where ground truth signal exists)
    labeled = df[df["label"] != 0].copy()
    if labeled.empty:
        return {h: 0.5 for h in range(24)}

    X_labeled = np.nan_to_num(
        labeled[FEATURE_COLS].to_numpy(dtype=np.float32), nan=0.0
    )
    probas       = entry_model.predict_proba(X_labeled)
    pred_classes = np.argmax(probas, axis=1)
    confs        = np.max(probas, axis=1)

    labeled = labeled.reset_index(drop=True)
    labeled["pred_class"] = pred_classes
    labeled["confidence"] = confs
    labeled["correct"]    = labeled["pred_class"] == labeled["label"]

    correct = labeled[labeled["correct"]]
    global_mean = float(correct["confidence"].mean()) if not correct.empty else 0.6
    global_thr  = max(0.50, global_mean * 0.85)

    curve: dict[int, float] = {}
    for hour in range(24):
        hour_correct = correct[correct["hour_utc"] == hour]
        if len(hour_correct) < 10:
            curve[hour] = round(global_thr, 4)
        else:
            mean_conf = float(hour_correct["confidence"].mean())
            curve[hour] = round(max(0.50, mean_conf * 0.85), 4)

    return curve


def _get_feature_importance(model) -> list[tuple[str, float]]:
    """
    Return (feature_name, importance_gain) pairs sorted DESC by gain,
    using model.feature_importances_ (normalised gain array).
    """
    importances = model.feature_importances_   # shape (n_features,)
    ranked = sorted(
        zip(FEATURE_COLS, importances.tolist()),
        key=lambda x: -x[1],
    )
    return ranked


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_config() -> dict:
    """
    Read all backtest/model artefacts and write DISCOVERED_PARAMS to config.py.

    Steps:
      1. Load/derive winner params per instrument.
      2. Load entry metadata (N, T, AUC per instrument).
      3. Load exit-backtest metrics per instrument.
      4. Filter ACTIVE_INSTRUMENTS by Sharpe/DD/trades thresholds.
      5. Compute CONFIDENCE_CURVE per active instrument.
      6. Compute FEATURE_IMPORTANCE per active instrument.
      7. Write DISCOVERED_PARAMS block to config.py.

    Returns
    -------
    The populated DISCOVERED_PARAMS dict.
    """
    log.info("=== generate_config() starting ===")

    # ------------------------------------------------------------------
    # 1. Entry metadata
    # ------------------------------------------------------------------
    entry_meta_path = MODELS_DIR / "metadata.json"
    if not entry_meta_path.exists():
        raise FileNotFoundError(
            f"{entry_meta_path} not found. Run ml/multi_combo_trainer.py first."
        )
    entry_meta = _load_json(entry_meta_path)

    # ------------------------------------------------------------------
    # 2. Winner params (load or derive)
    # ------------------------------------------------------------------
    winner_params = _ensure_winner_params(entry_meta)

    # ------------------------------------------------------------------
    # 3. Exit-backtest metrics per instrument
    # ------------------------------------------------------------------
    metrics_by_inst: dict[str, dict] = {}
    for instrument in ALL_INSTRUMENTS:
        metrics_path = RESULTS_EXIT_DIR / f"{instrument}_metrics.json"
        if metrics_path.exists():
            metrics_by_inst[instrument] = _load_json(metrics_path)

    # ------------------------------------------------------------------
    # 4. Determine ACTIVE_INSTRUMENTS
    # ------------------------------------------------------------------
    active: list[str] = []
    excluded: list[str] = []

    for instrument in ALL_INSTRUMENTS:
        if instrument not in metrics_by_inst:
            log.info("[%s] EXCLUDED — no metrics file.", instrument)
            excluded.append(instrument)
            continue
        if instrument not in winner_params:
            log.info("[%s] EXCLUDED — no winner params.", instrument)
            excluded.append(instrument)
            continue

        m = metrics_by_inst[instrument]
        sharpe  = float(m.get("sharpe_ratio",  0.0))
        max_dd  = float(m.get("max_drawdown",  1.0))
        n_trades = int(m.get("total_trades",   0))

        reasons = []
        if sharpe  < MIN_SHARPE:   reasons.append(f"Sharpe {sharpe:.2f} < {MIN_SHARPE}")
        if max_dd  > MAX_DRAWDOWN: reasons.append(f"MaxDD {max_dd:.1%} > {MAX_DRAWDOWN:.0%}")
        if n_trades < MIN_TRADES:  reasons.append(f"Trades {n_trades} < {MIN_TRADES}")

        if reasons:
            log.info("[%s] EXCLUDED — %s", instrument, " | ".join(reasons))
            excluded.append(instrument)
        else:
            log.info(
                "[%s] ACTIVE — Sharpe=%.2f MaxDD=%.1f%% Trades=%d",
                instrument, sharpe, max_dd * 100, n_trades,
            )
            active.append(instrument)

    log.info("Active: %s", active)
    if excluded:
        log.info("Excluded: %s", excluded)

    # ------------------------------------------------------------------
    # 5-6. Per-instrument: CONFIDENCE_CURVE + FEATURE_IMPORTANCE
    # ------------------------------------------------------------------
    confidence_curves:  dict[str, dict[int, float]] = {}
    feature_importance: dict[str, list[str]]        = {}
    imp_matrices: dict[str, dict[str, float]]       = {}

    for instrument in active:
        instr_meta  = entry_meta.get(instrument, {})
        N = int(instr_meta.get("N", 10))
        T = float(instr_meta.get("T", 0.002))

        entry_model_path = MODELS_DIR / f"entry_{instrument}.pkl"
        if not entry_model_path.exists():
            log.warning("[%s] Entry model missing — skipping curves.", instrument)
            continue

        with open(entry_model_path, "rb") as fh:
            entry_model = pickle.load(fh)

        # Confidence curve (from training labels)
        log.info("[%s] Computing confidence curve ...", instrument)
        curve = _compute_confidence_curve(instrument, entry_model, N, T)
        confidence_curves[instrument] = curve

        # Feature importance
        ranked = _get_feature_importance(entry_model)
        feature_importance[instrument] = [name for name, _ in ranked]
        imp_matrices[instrument]       = {name: imp for name, imp in ranked}

        log.info(
            "[%s] Top-3 features: %s",
            instrument, [n for n, _ in ranked[:3]],
        )

    # ------------------------------------------------------------------
    # Global feature ranking (mean gain across all active instruments)
    # ------------------------------------------------------------------
    global_ranking: list[str] = []
    if imp_matrices:
        mean_imp: dict[str, float] = {}
        for feat in FEATURE_COLS:
            vals = [d.get(feat, 0.0) for d in imp_matrices.values()]
            mean_imp[feat] = float(np.mean(vals))
        global_ranking = [
            f for f, _ in sorted(mean_imp.items(), key=lambda x: -x[1])
        ]

    # ------------------------------------------------------------------
    # 7. Assemble DISCOVERED_PARAMS
    # ------------------------------------------------------------------
    instrument_params: dict[str, dict] = {}
    for instrument in active:
        m         = metrics_by_inst[instrument]
        instr_meta = entry_meta.get(instrument, {})
        wp        = winner_params[instrument]

        instrument_params[instrument] = {
            "N":               int(instr_meta.get("N", 10)),
            "T":               float(instr_meta.get("T", 0.002)),
            "trail_atr_mult":  float(wp["trail_atr_mult"]),
            "exit_threshold":  float(wp["exit_threshold"]),
            "entry_model":     f"ml/models/entry_{instrument}.pkl",
            "exit_model":      f"ml/models/exit_{instrument}.pkl",
            "holdout_sharpe":  round(float(m.get("sharpe_ratio", 0.0)), 4),
            "holdout_max_dd":  round(float(m.get("max_drawdown", 0.0)), 4),
            "holdout_win_rate": round(float(m.get("win_rate", 0.0)), 4),
            "holdout_pnl_gbp": round(float(m.get("total_pnl_gbp", 0.0)), 2),
        }

    discovered: dict = {
        "ACTIVE_INSTRUMENTS":   active,
        "INSTRUMENT_PARAMS":    instrument_params,
        "CONFIDENCE_CURVE":     confidence_curves,
        "FEATURE_IMPORTANCE":   feature_importance,
        "GLOBAL_FEATURE_RANKING": global_ranking,
        "GENERATED_AT":         datetime.now(tz=timezone.utc).isoformat(),
        "GENERATED_FROM":       "ml/config_generator.py",
    }

    # ------------------------------------------------------------------
    # Write to config.py
    # ------------------------------------------------------------------
    _write_discovered_params(discovered)

    log.info("=== generate_config() complete ===")
    return discovered


# ---------------------------------------------------------------------------
# config.py writer
# ---------------------------------------------------------------------------

def _write_discovered_params(discovered: dict) -> None:
    """
    Replace the DISCOVERED_PARAMS block in config.py with the new dict.

    Finds the comment line '# Populated by ml/config_generator.py' and
    replaces everything from there to end-of-file with the new block.
    Falls back to appending if the marker is not found.
    """
    config_text = CONFIG_PATH.read_text(encoding="utf-8")

    # Pretty-print the dict as a Python literal
    params_repr = _dict_to_python_literal(discovered, indent=0)
    new_block = (
        "# Populated by ml/config_generator.py after training — never hand-edit\n"
        f"DISCOVERED_PARAMS = {params_repr}\n"
    )

    # Match the marker line plus everything after it
    marker_pattern = re.compile(
        r"#\s*Populated by ml/config_generator\.py.*",
        re.DOTALL,
    )
    if marker_pattern.search(config_text):
        new_text = marker_pattern.sub(new_block.rstrip("\n"), config_text)
    else:
        # Fallback: replace bare DISCOVERED_PARAMS = {} anywhere
        bare_pattern = re.compile(r"^DISCOVERED_PARAMS\s*=\s*\{\}", re.MULTILINE)
        if bare_pattern.search(config_text):
            new_text = bare_pattern.sub(
                new_block.rstrip("\n"), config_text
            )
        else:
            # Append at end
            new_text = config_text.rstrip() + "\n\n" + new_block

    CONFIG_PATH.write_text(new_text, encoding="utf-8")
    log.info("config.py updated — DISCOVERED_PARAMS written.")


def _dict_to_python_literal(obj, indent: int = 0) -> str:
    """
    Recursively render a Python object as a formatted literal string.
    Handles dict, list, str, int, float, bool, None.
    """
    pad  = "    " * indent
    pad1 = "    " * (indent + 1)

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in obj.items():
            key_s = repr(k)
            val_s = _dict_to_python_literal(v, indent + 1)
            items.append(f"{pad1}{key_s}: {val_s}")
        return "{\n" + ",\n".join(items) + ",\n" + pad + "}"

    elif isinstance(obj, list):
        if not obj:
            return "[]"
        # Short lists of scalars on one line
        if all(isinstance(x, (str, int, float, bool)) for x in obj) and len(obj) <= 6:
            return "[" + ", ".join(repr(x) for x in obj) + "]"
        # Longer lists — one item per line
        items = [f"{pad1}{_dict_to_python_literal(x, indent + 1)}" for x in obj]
        return "[\n" + ",\n".join(items) + ",\n" + pad + "]"

    elif isinstance(obj, bool):
        return repr(obj)

    elif isinstance(obj, float):
        # Avoid scientific notation for small floats
        if abs(obj) < 1e-4 and obj != 0.0:
            return f"{obj:.8f}"
        return repr(round(obj, 8))

    else:
        return repr(obj)


# ---------------------------------------------------------------------------
# Feature pruning report
# ---------------------------------------------------------------------------

def print_pruning_report() -> None:
    """
    Print a table of feature importance rankings across all active instruments.
    Shows which features are in the top 40 globally (KEEP) vs below (DROP).
    Informational only — no features are actually pruned in v1.
    """
    try:
        import importlib
        import config as cfg
        importlib.reload(cfg)
        discovered = cfg.DISCOVERED_PARAMS
    except Exception:
        log.error("config.DISCOVERED_PARAMS not available — run generate_config() first.")
        return

    fi = discovered.get("FEATURE_IMPORTANCE", {})
    if not fi:
        print("No feature importance data available.")
        return

    instruments = list(fi.keys())

    # Rank of each feature per instrument (1-indexed)
    rank_map: dict[str, list[int]] = {feat: [] for feat in FEATURE_COLS}
    for inst in instruments:
        ordered = fi[inst]
        for rank, feat in enumerate(ordered, start=1):
            if feat in rank_map:
                rank_map[feat].append(rank)

    # Mean rank + top-20 count
    rows = []
    for feat in FEATURE_COLS:
        ranks = rank_map[feat]
        mean_rank  = float(np.mean(ranks)) if ranks else 999.0
        top20_count = sum(1 for r in ranks if r <= 20)
        rows.append({
            "feature":     feat,
            "mean_rank":   round(mean_rank, 1),
            "top20_count": top20_count,
        })

    rows.sort(key=lambda x: x["mean_rank"])

    # Global ranking from DISCOVERED_PARAMS
    global_ranking = discovered.get("GLOBAL_FEATURE_RANKING", [feat for feat in FEATURE_COLS])
    keep_set = set(global_ranking[:40])

    print(f"\n{'Feature':<35} {'MeanRank':>9} {'Top20/'+str(len(instruments)):>10} {'Decision':>8}")
    print("-" * 65)
    for row in rows:
        decision = "KEEP" if row["feature"] in keep_set else "DROP"
        print(
            f"{row['feature']:<35} {row['mean_rank']:>9.1f}"
            f" {row['top20_count']:>10d}"
            f" {decision:>8}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_config()
    print_pruning_report()
