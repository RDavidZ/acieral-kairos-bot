"""
ml/exit_trainer.py — Walk-forward XGBoost exit model for Acieral Kairos Bot

Trains a binary exit model per instrument from oracle peak-exit labels.
The model predicts P(exit) at each 1H bar during an open trade.

Feature set
-----------
EXIT_FEATURE_COLS = FEATURE_COLS (81 market features) +
                    7 trade-context features (tc_*)  = 88 total

Walk-forward scheme
-------------------
Data spans the holdout period (~27 months for a 2024–2026 backtest).
Folds are generated from the actual data range: 18-month training,
3-month validation, 3-month step — yields ~3 usable folds.
Fold membership is determined by trade entry time (all bars of a single
trade stay in the same fold), using min(bar_time) per trade_id as proxy.

Class imbalance
---------------
Binary classification: scale_pos_weight = n_stay / n_exit (~13×).
No SMOTE needed — XGBoost handles it natively for binary tasks.
"""

import json
import logging
from datetime import timezone
from pathlib import Path
from statistics import mode

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
from xgboost import XGBClassifier

from config import HARD_CONSTRAINTS
from ml.labeler import FEATURE_COLS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TC_FEATURES = [
    "tc_bars_held",
    "tc_unrealised_atr",
    "tc_mfe_atr",
    "tc_mae_atr",
    "tc_pct_mfe_given_back",
    "tc_direction",
    "tc_dist_to_sl_atr",
]

EXIT_FEATURE_COLS: list[str] = FEATURE_COLS + TC_FEATURES   # 88 features

assert len(EXIT_FEATURE_COLS) == 88, (
    f"Expected 88 EXIT_FEATURE_COLS, got {len(EXIT_FEATURE_COLS)}"
)

MODELS_DIR      = Path(__file__).parent / "models"
EXIT_LABELS_DIR = Path(__file__).parent / "exit_labels"
BACKTEST_DIR    = Path(__file__).parent.parent / "backtest"

MODELS_DIR.mkdir(exist_ok=True)

ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]

# XGBoost fixed settings
XGB_FIXED = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    early_stopping_rounds=20,
)

# Hyperparameter grid (12 combos)
PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in [200, 400]
    for d in [3, 4, 5]
    for lr in [0.05, 0.1]
]

# Minimum fold sizes
MIN_TRAIN_EXIT = 100   # EXIT=1 rows in training fold
MIN_VAL_EXIT   = 20    # EXIT=1 rows in validation fold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_exit_labels(instrument: str) -> pd.DataFrame:
    path = EXIT_LABELS_DIR / f"{instrument}_exit_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Exit labels not found: {path}\n"
            f"Run ml.exit_labeler.label_exits('{instrument}') first."
        )
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    missing = [c for c in EXIT_FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{instrument}] Missing EXIT_FEATURE_COLS in exit labels: {missing}"
        )
    return df


def _add_fold_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'fold_anchor' column: the earliest bar_time per trade_id.
    This is a proxy for trade entry_time (≈ entry_time + 1H), used
    to keep all bars of a trade in the same fold.
    """
    anchors = df.groupby("trade_id")["time"].min().rename("fold_anchor")
    return df.merge(anchors, on="trade_id", how="left")


def _generate_folds(df: pd.DataFrame) -> list[dict]:
    """
    Generate walk-forward folds over the exit label data.
    18-month training, 3-month validation, 3-month step.
    Uses 'fold_anchor' column for boundary assignment.
    """
    anchors  = df["fold_anchor"]
    data_start = anchors.min().normalize()
    data_end   = anchors.max()

    train_window = pd.DateOffset(months=18)
    val_window   = pd.DateOffset(months=3)
    step         = pd.DateOffset(months=3)

    folds = []
    fold_start = data_start

    while True:
        train_start = fold_start
        train_end   = fold_start + train_window
        val_start   = train_end
        val_end     = val_start + val_window

        if val_start >= data_end:
            break

        folds.append({
            "train_start": train_start,
            "train_end":   train_end,
            "val_start":   val_start,
            "val_end":     val_end,
        })

        fold_start = fold_start + step

    return folds


def _load_exit_metadata() -> dict:
    path = MODELS_DIR / "exit_metadata.json"
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def _save_exit_metadata(metadata: dict) -> None:
    with open(MODELS_DIR / "exit_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_exit_instrument(instrument: str) -> dict:
    """
    Train the walk-forward exit model for *instrument*.

    Parameters
    ----------
    instrument : OANDA instrument name, e.g. 'EUR_USD'

    Returns
    -------
    dict with keys: n_folds, mean_val_auc, std_val_auc, best_params,
                    n_trades, fold_results
    """
    log.info("=== train_exit_instrument(%s) ===", instrument)

    df = _load_exit_labels(instrument)
    df = _add_fold_anchor(df)

    n_trades = df["trade_id"].nunique()
    n_exit   = int((df["label"] == 1).sum())
    n_stay   = int((df["label"] == 0).sum())
    spw      = n_stay / max(n_exit, 1)

    log.info(
        "[%s] %d rows | EXIT=%d (%.1f%%) | STAY=%d | scale_pos_weight=%.2f",
        instrument, len(df), n_exit, n_exit / len(df) * 100, n_stay, spw,
    )

    folds = _generate_folds(df)
    log.info("[%s] %d walk-forward folds generated.", instrument, len(folds))

    fold_results: list[dict] = []

    for fold_idx, fold in enumerate(folds):
        train_mask = (
            (df["fold_anchor"] >= fold["train_start"]) &
            (df["fold_anchor"] <  fold["train_end"])
        )
        val_mask = (
            (df["fold_anchor"] >= fold["val_start"]) &
            (df["fold_anchor"] <  fold["val_end"])
        )

        df_tr  = df[train_mask]
        df_val = df[val_mask]

        n_tr_exit  = int((df_tr["label"]  == 1).sum())
        n_val_exit = int((df_val["label"] == 1).sum())

        if n_tr_exit < MIN_TRAIN_EXIT:
            log.info(
                "  Fold %2d | SKIP — train EXIT=%d < %d",
                fold_idx, n_tr_exit, MIN_TRAIN_EXIT,
            )
            continue

        if n_val_exit < MIN_VAL_EXIT:
            log.info(
                "  Fold %2d | SKIP — val EXIT=%d < %d",
                fold_idx, n_val_exit, MIN_VAL_EXIT,
            )
            continue

        X_tr_full = df_tr[EXIT_FEATURE_COLS].to_numpy(dtype=np.float32)
        y_tr_full = df_tr["label"].to_numpy(dtype=np.int32)
        X_val     = df_val[EXIT_FEATURE_COLS].to_numpy(dtype=np.float32)
        y_val     = df_val["label"].to_numpy(dtype=np.int32)

        # 10% eval split from training fold for early stopping
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X_tr_full, y_tr_full, test_size=0.10,
            random_state=42, stratify=y_tr_full,
        )

        best_auc    = -np.inf
        best_params = PARAM_GRID[0]

        for params in PARAM_GRID:
            model = XGBClassifier(
                **XGB_FIXED,
                **params,
                scale_pos_weight=spw,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_ev, y_ev)],
                verbose=False,
            )

            prob_val = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, prob_val)

            if auc > best_auc:
                best_auc    = auc
                best_params = params

        tr_start_str = fold["train_start"].strftime("%Y-%m")
        tr_end_str   = fold["train_end"].strftime("%Y-%m")
        val_start_str = fold["val_start"].strftime("%Y-%m")
        val_end_str   = fold["val_end"].strftime("%Y-%m")

        log.info(
            "  Fold %2d | train %s -> %s | val %s -> %s "
            "| AUC=%.4f | params=%s",
            fold_idx,
            tr_start_str, tr_end_str,
            val_start_str, val_end_str,
            best_auc, best_params,
        )

        fold_results.append({
            "fold":         fold_idx,
            "val_auc":      best_auc,
            "best_params":  best_params,
            "n_tr_exit":    n_tr_exit,
            "n_val_exit":   n_val_exit,
        })

    if not fold_results:
        raise RuntimeError(
            f"[{instrument}] No valid folds completed. "
            f"Check that exit labels exist and have sufficient EXIT=1 rows."
        )

    val_aucs   = [r["val_auc"]     for r in fold_results]
    all_params = [r["best_params"] for r in fold_results]

    mean_auc = float(np.mean(val_aucs))
    std_auc  = float(np.std(val_aucs))

    # Mode best_params across folds (key-by-key)
    mode_params = {
        key: mode([p[key] for p in all_params])
        for key in all_params[0]
    }

    log.info(
        "[%s] Training final exit model | mean_auc=%.4f +/-%.4f | mode_params=%s",
        instrument, mean_auc, std_auc, mode_params,
    )

    # ------------------------------------------------------------------
    # Final model — retrain on ALL exit label data, no early stopping
    # ------------------------------------------------------------------
    X_all = df[EXIT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy(dtype=np.int32)

    final_params = {k: v for k, v in XGB_FIXED.items()
                    if k != "early_stopping_rounds"}
    final_model = XGBClassifier(
        **final_params,
        **mode_params,
        scale_pos_weight=spw,
    )
    final_model.fit(X_all, y_all, verbose=False)

    model_path = MODELS_DIR / f"exit_{instrument}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(final_model, fh)
    log.info("[%s] Exit model saved -> %s", instrument, model_path.name)

    # ------------------------------------------------------------------
    # Update exit_metadata.json
    # ------------------------------------------------------------------
    from datetime import datetime
    metadata = _load_exit_metadata()
    metadata[instrument] = {
        "mean_val_auc": mean_auc,
        "std_val_auc":  std_auc,
        "best_params":  mode_params,
        "n_folds":      len(fold_results),
        "n_trades":     n_trades,
        "trained_at":   datetime.now(tz=timezone.utc).isoformat(),
    }
    _save_exit_metadata(metadata)
    log.info("[%s] exit_metadata.json updated.", instrument)

    return {
        "instrument":   instrument,
        "n_folds":      len(fold_results),
        "mean_val_auc": mean_auc,
        "std_val_auc":  std_auc,
        "best_params":  mode_params,
        "n_trades":     n_trades,
        "fold_results": fold_results,
    }


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_exit_threshold(
    instrument: str,
    model_path: str,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Sweep exit probability thresholds across the full exit label dataset.
    Returns a DataFrame summarising precision/recall/F1 and oracle capture
    rate at each threshold.

    Parameters
    ----------
    instrument  : OANDA instrument name
    model_path  : Path to the saved exit model .pkl
    thresholds  : Probability thresholds to evaluate
                  (default: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    df = _load_exit_labels(instrument)

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    X    = df[EXIT_FEATURE_COLS].to_numpy(dtype=np.float32)
    y    = df["label"].to_numpy(dtype=np.int32)
    prob = model.predict_proba(X)[:, 1]

    n_oracle_peaks = int(y.sum())  # total EXIT=1 labels

    rows = []
    for thr in thresholds:
        pred = (prob >= thr).astype(np.int32)

        n_predicted = int(pred.sum())
        if n_predicted == 0:
            prec = rec = f1 = 0.0
        else:
            prec = float(precision_score(y, pred, zero_division=0))
            rec  = float(recall_score(y, pred,    zero_division=0))
            f1   = float(f1_score(y, pred,        zero_division=0))

        # Oracle peaks captured: pred=1 AND label=1
        captured = int(((pred == 1) & (y == 1)).sum())
        pct_captured = captured / n_oracle_peaks * 100 if n_oracle_peaks else 0.0

        # Mean tc_unrealised_atr at predicted exit bars
        pred_exit_mask = pred == 1
        if pred_exit_mask.sum() > 0:
            mean_unreal = float(
                df.loc[pred_exit_mask, "tc_unrealised_atr"].mean()
            )
        else:
            mean_unreal = 0.0

        rows.append({
            "threshold":        thr,
            "n_predicted":      n_predicted,
            "precision":        round(prec, 4),
            "recall":           round(rec, 4),
            "f1":               round(f1, 4),
            "oracle_captured":  captured,
            "pct_captured":     round(pct_captured, 1),
            "mean_unrealised_atr": round(mean_unreal, 3),
        })

    result = pd.DataFrame(rows)
    log.info(
        "[%s] Threshold sweep complete. n_oracle_peaks=%d",
        instrument, n_oracle_peaks,
    )
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def train_all_exits() -> pd.DataFrame:
    """
    Run the full exit pipeline for all 8 instruments:
      1. Refresh entry backtest trade log at confidence=0.5
      2. Re-generate exit labels
      3. Train walk-forward exit model

    Saves exit models and updates exit_metadata.json.
    Returns summary DataFrame.
    """
    from backtest.engine import run_backtest
    from ml.exit_labeler import label_exits

    log.info("=== train_all_exits() starting ===")

    # Load entry metadata to get winner N/T per instrument
    entry_meta_path = MODELS_DIR / "metadata.json"
    if not entry_meta_path.exists():
        raise FileNotFoundError(
            f"Entry metadata not found: {entry_meta_path}\n"
            "Run ml.multi_combo_trainer.run_all_multi_combo() first."
        )
    with open(entry_meta_path) as fh:
        entry_meta = json.load(fh)

    summary_rows = []

    for instrument in ALL_INSTRUMENTS:
        if instrument not in entry_meta:
            log.warning("[%s] No entry metadata — skipping.", instrument)
            continue

        instr_meta = entry_meta[instrument]
        N = instr_meta["N"]
        T = instr_meta["T"]
        entry_model = MODELS_DIR / f"entry_{instrument}.pkl"

        if not entry_model.exists():
            log.warning("[%s] Entry model not found — skipping.", instrument)
            continue

        log.info("--- %s (N=%d, T=%.4f) ---", instrument, N, T)

        try:
            # Step 1: Refresh trade log at confidence=0.5
            log.info("[%s] Refreshing backtest trade log ...", instrument)
            run_backtest(instrument, str(entry_model), N, T, split="holdout")

            # Step 2: Re-generate exit labels
            log.info("[%s] Generating exit labels ...", instrument)
            df_labels = label_exits(instrument)
            out_path  = EXIT_LABELS_DIR / f"{instrument}_exit_labels.parquet"
            df_labels.to_parquet(out_path, index=False)
            log.info("[%s] Exit labels saved -> %s", instrument, out_path.name)

            # Step 3: Train exit model
            res = train_exit_instrument(instrument)

            summary_rows.append({
                "instrument":   instrument,
                "n_folds":      res["n_folds"],
                "mean_val_auc": res["mean_val_auc"],
                "std_val_auc":  res["std_val_auc"],
                "n_trades":     res["n_trades"],
            })

        except Exception as exc:
            log.error(
                "[%s] train_all_exits FAILED: %s", instrument, exc, exc_info=True
            )

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        header = (
            f"\n{'Instrument':<15} {'Folds':>6} {'Mean AUC':>9}"
            f" {'Std AUC':>8} {'Trades':>7}"
        )
        print(header)
        print("-" * len(header.strip()))
        for _, row in summary_df.iterrows():
            print(
                f"{row['instrument']:<15} {int(row['n_folds']):>6}"
                f" {row['mean_val_auc']:>9.4f}"
                f" {row['std_val_auc']:>8.4f}"
                f" {int(row['n_trades']):>7}"
            )

    log.info("=== train_all_exits() complete ===")
    return summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_all_exits()
