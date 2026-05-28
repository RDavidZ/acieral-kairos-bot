"""
ml/trainer.py — Walk-forward XGBoost entry model trainer for Acieral Kairos Bot

Two-pass approach per fold:
  Pass 1 (grid search): SMOTE-balanced train data → 12 hyperparameter combos
                        → pick best val AUC
  Pass 2 (fold retrain): raw train data + sample_weight → best params

Final model: all training data, mode hyperparams, sample_weight (no SMOTE).
"""

import json
import logging
import pickle
import threading
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import mode as stat_mode

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import HARD_CONSTRAINTS
from ml.labeler import FEATURE_COLS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LABELS_DIR = Path(__file__).parent / "labels_train"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

ALL_INSTRUMENTS = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
TRAINING_START  = HARD_CONSTRAINTS["TRAINING_START"]   # "2017-01-01"
HOLDOUT_START   = HARD_CONSTRAINTS["HOLDOUT_START"]    # "2024-01-01"
TRAIN_MONTHS    = HARD_CONSTRAINTS["TRAINING_WINDOW_MONTHS"]  # 18
VAL_MONTHS      = HARD_CONSTRAINTS["VALIDATION_MONTHS"]       # 3

# Shared lock for concurrent metadata.json writes (instrument-level parallelism)
_metadata_lock = threading.Lock()

# XGBoost fixed parameters
XGB_FIXED = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=4,   # 4 threads per model; 3 parallel instruments → 12 cores total
)

# Hyperparameter grid (12 combos per fold)
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth":    [3, 4, 5],
    "learning_rate": [0.05, 0.1],
}

# Default N/T per instrument (pre-multi-combo selection)
DEFAULT_NT = {inst: (10, 0.002) for inst in ALL_INSTRUMENTS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t_to_bps_str(T: float) -> str:
    """Convert threshold to filename suffix: 0.002 → 'T20bps'."""
    bps = int(round(T * 10_000))
    return f"T{bps}bps"


def get_scale_pos_weight(y: np.ndarray) -> dict:
    """
    Compute per-class inverse-frequency weights.

    weight_c = (total_samples - count_c) / count_c

    Returns a dict {class_label: weight}.  Majority class (0 = NO_TRADE)
    will have the lowest weight; minority classes 1 and 2 will have higher
    weights proportional to their under-representation.
    """
    total = len(y)
    unique, counts = np.unique(y, return_counts=True)
    return {int(c): float((total - n) / n) for c, n in zip(unique, counts)}


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Convert per-class inverse-frequency weights into a per-sample array
    suitable for XGBoost's sample_weight parameter.

    Uses sklearn's compute_sample_weight('balanced') which implements the
    same formula as get_scale_pos_weight but returns per-sample values.
    """
    return compute_sample_weight("balanced", y)


# ---------------------------------------------------------------------------
# SMOTE balancing
# ---------------------------------------------------------------------------

def _smote_balance(
    X: np.ndarray,
    y: np.ndarray,
    no_trade_cap_mult: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance training data using SMOTE.

    Steps
    -----
    1. Find minority_count = min(n_short, n_long).
    2. Cap NO_TRADE at no_trade_cap_mult × minority_count.
    3. Subsample NO_TRADE rows to the cap.
    4. Combine subsampled NO_TRADE + all SHORT + all LONG.
    5. Apply SMOTE to bring SHORT and LONG up to no_trade_target.

    Returns X_balanced, y_balanced.
    """
    idx_no    = np.where(y == 0)[0]
    idx_short = np.where(y == 1)[0]
    idx_long  = np.where(y == 2)[0]

    minority_count = min(len(idx_short), len(idx_long))
    if minority_count < 4:
        # Too few minority samples to apply SMOTE — return as-is
        return X, y

    no_trade_target = min(len(idx_no), no_trade_cap_mult * minority_count)

    # Subsample NO_TRADE
    rng = np.random.default_rng(42)
    if len(idx_no) > no_trade_target:
        idx_no_sampled = rng.choice(idx_no, size=no_trade_target, replace=False)
    else:
        idx_no_sampled = idx_no

    idx_combined = np.concatenate([idx_no_sampled, idx_short, idx_long])
    X_combined = X[idx_combined]
    y_combined = y[idx_combined]

    # SMOTE: oversample minority classes to match no_trade_target
    smote = SMOTE(
        random_state=42,
        k_neighbors=3,
        sampling_strategy={1: no_trade_target, 2: no_trade_target},
    )
    try:
        X_bal, y_bal = smote.fit_resample(X_combined, y_combined)
    except Exception as exc:
        log.warning("SMOTE failed (%s) — using unbalanced data.", exc)
        return X_combined, y_combined

    return X_bal, y_bal


# ---------------------------------------------------------------------------
# Walk-forward fold generation
# ---------------------------------------------------------------------------

def _generate_folds(df: pd.DataFrame) -> list[dict]:
    """
    Generate walk-forward fold definitions.

    Each fold is a dict with keys:
      train_start, train_end, val_start, val_end  (pd.Timestamp, UTC)

    Fold cadence
    ------------
    Fold 0: train [TRAINING_START, +18m)  val [+18m, +21m)
    Fold 1: train [+3m, +21m)             val [+21m, +24m)
    ...
    Last fold: val_end ≤ HOLDOUT_START
    """
    holdout = pd.Timestamp(HOLDOUT_START, tz="UTC")
    folds   = []

    step = 0
    while True:
        train_start = pd.Timestamp(TRAINING_START, tz="UTC") + pd.DateOffset(months=3 * step)
        val_start   = train_start + pd.DateOffset(months=TRAIN_MONTHS)
        val_end     = val_start   + pd.DateOffset(months=VAL_MONTHS)

        if val_end > holdout:
            break

        folds.append({
            "train_start": train_start,
            "train_end":   val_start,
            "val_start":   val_start,
            "val_end":     val_end,
        })
        step += 1

    return folds


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_instrument(
    instrument: str,
    N: int,
    T: float,
) -> dict:
    """
    Walk-forward XGBoost training for one instrument.

    Parameters
    ----------
    instrument : OANDA instrument name, e.g. 'EUR_USD'
    N          : Labelling lookahead (bars), must match an existing label file
    T          : Labelling threshold (fraction), must match an existing label file

    Returns
    -------
    dict with keys:
      n_folds, mean_val_auc, std_val_auc, best_params,
      fold_results, model_path, metadata_path
    """
    # ------------------------------------------------------------------
    # 1. Load label file
    # ------------------------------------------------------------------
    t_str  = _t_to_bps_str(T)
    fname  = f"{instrument}_N{N}_{t_str}_labels.parquet"
    fpath  = LABELS_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(
            f"Label file not found: {fpath}\n"
            f"Run ml.labeler.label_instrument('{instrument}', N={N}, T={T}) first."
        )

    df = pd.read_parquet(fpath)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    log.info(
        "[%s] Loaded %d rows (N=%d, T=%.4f). Label dist: %s",
        instrument, len(df), N, T,
        df["label"].value_counts().to_dict(),
    )

    # ------------------------------------------------------------------
    # 2. Generate walk-forward folds
    # ------------------------------------------------------------------
    folds = _generate_folds(df)
    log.info("[%s] %d walk-forward folds generated.", instrument, len(folds))

    # ------------------------------------------------------------------
    # 3. Walk-forward loop
    # ------------------------------------------------------------------
    fold_results: list[dict] = []
    skipped = 0

    for fold_idx, fold in enumerate(folds):
        train_df = df[
            (df["time"] >= fold["train_start"]) &
            (df["time"] <  fold["train_end"])
        ]
        val_df = df[
            (df["time"] >= fold["val_start"]) &
            (df["time"] <  fold["val_end"])
        ]

        n_labeled_train = int((train_df["label"] != 0).sum())
        n_labeled_val   = int((val_df["label"] != 0).sum())

        if n_labeled_train < 200:
            log.debug(
                "  Fold %d skipped — too few labeled training rows (%d < 200).",
                fold_idx, n_labeled_train,
            )
            skipped += 1
            continue

        if n_labeled_val < 50:
            log.debug(
                "  Fold %d skipped — too few labeled val rows (%d < 50).",
                fold_idx, n_labeled_val,
            )
            skipped += 1
            continue

        X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=np.int32)
        X_val   = val_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_val   = val_df["label"].to_numpy(dtype=np.int32)

        # Replace NaN with 0 (XGBoost can handle NaN but SMOTE cannot)
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val   = np.nan_to_num(X_val,   nan=0.0)

        # (a-c) SMOTE balancing
        X_smote, y_smote = _smote_balance(X_train, y_train)

        # (d-e) Hyperparameter grid search
        best_fold_auc    = -1.0
        best_fold_params: dict = {}

        # 10% of SMOTE data for early-stopping eval set
        X_tr, X_es, y_tr, y_es = train_test_split(
            X_smote, y_smote,
            test_size=0.1,
            random_state=42,
            stratify=y_smote,
        )

        for n_est, md, lr in product(
            PARAM_GRID["n_estimators"],
            PARAM_GRID["max_depth"],
            PARAM_GRID["learning_rate"],
        ):
            model = XGBClassifier(
                n_estimators=n_est,
                max_depth=md,
                learning_rate=lr,
                early_stopping_rounds=20,
                **XGB_FIXED,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                verbose=False,
            )

            y_proba = model.predict_proba(X_val)
            try:
                auc = roc_auc_score(
                    y_val, y_proba,
                    multi_class="ovr",
                    average="macro",
                )
            except ValueError:
                # Validation set has only one class — skip
                auc = 0.5

            if auc > best_fold_auc:
                best_fold_auc    = auc
                best_fold_params = {
                    "n_estimators":  n_est,
                    "max_depth":     md,
                    "learning_rate": lr,
                }

        # (g) Retrain on full fold training data with sample_weight
        sw = _sample_weights(y_train)
        retrain_model = XGBClassifier(
            n_estimators=best_fold_params["n_estimators"],
            max_depth=best_fold_params["max_depth"],
            learning_rate=best_fold_params["learning_rate"],
            **XGB_FIXED,
        )
        retrain_model.fit(X_train, y_train, sample_weight=sw, verbose=False)

        fold_results.append({
            "fold_idx":    fold_idx,
            "train_start": fold["train_start"].isoformat(),
            "train_end":   fold["train_end"].isoformat(),
            "val_start":   fold["val_start"].isoformat(),
            "val_end":     fold["val_end"].isoformat(),
            "n_train":     len(train_df),
            "n_labeled_train": n_labeled_train,
            "n_val":       len(val_df),
            "n_labeled_val": n_labeled_val,
            "best_params": best_fold_params,
            "val_auc":     round(best_fold_auc, 6),
        })

        log.info(
            "  Fold %2d | train %s – %s | val %s – %s | "
            "AUC=%.4f | params=%s",
            fold_idx,
            fold["train_start"].strftime("%Y-%m"),
            fold["train_end"].strftime("%Y-%m"),
            fold["val_start"].strftime("%Y-%m"),
            fold["val_end"].strftime("%Y-%m"),
            best_fold_auc,
            best_fold_params,
        )

    if not fold_results:
        raise RuntimeError(
            f"[{instrument}] No folds completed — check data range and label counts."
        )

    # ------------------------------------------------------------------
    # 4. Final model on ALL training data
    # ------------------------------------------------------------------
    aucs = [f["val_auc"] for f in fold_results]
    mean_auc = float(np.mean(aucs))
    std_auc  = float(np.std(aucs))

    # Mode of each hyperparameter across folds
    all_n_est = [f["best_params"]["n_estimators"]  for f in fold_results]
    all_md    = [f["best_params"]["max_depth"]      for f in fold_results]
    all_lr    = [f["best_params"]["learning_rate"]  for f in fold_results]

    mode_params = {
        "n_estimators":  stat_mode(all_n_est),
        "max_depth":     stat_mode(all_md),
        "learning_rate": stat_mode(all_lr),
    }

    log.info(
        "[%s] Training final model | mean_auc=%.4f ± %.4f | mode_params=%s",
        instrument, mean_auc, std_auc, mode_params,
    )

    X_all = np.nan_to_num(
        df[FEATURE_COLS].to_numpy(dtype=np.float32), nan=0.0
    )
    y_all = df["label"].to_numpy(dtype=np.int32)
    sw_all = _sample_weights(y_all)

    final_model = XGBClassifier(
        n_estimators=mode_params["n_estimators"],
        max_depth=mode_params["max_depth"],
        learning_rate=mode_params["learning_rate"],
        **XGB_FIXED,
    )
    final_model.fit(X_all, y_all, sample_weight=sw_all, verbose=False)

    # Save model
    model_path = MODELS_DIR / f"entry_{instrument}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    log.info("[%s] Model saved → %s", instrument, model_path.name)

    # Save / update metadata (locked — multiple instruments may write concurrently)
    metadata_path = MODELS_DIR / "metadata.json"
    with _metadata_lock:
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata[instrument] = {
            "N":             N,
            "T":             T,
            "mean_val_auc":  round(mean_auc, 6),
            "std_val_auc":   round(std_auc, 6),
            "best_params":   mode_params,
            "n_folds":       len(fold_results),
            "skipped_folds": skipped,
            "trained_at":    datetime.now(timezone.utc).isoformat(),
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    log.info("[%s] Metadata saved → %s", instrument, metadata_path.name)

    return {
        "instrument":    instrument,
        "N":             N,
        "T":             T,
        "n_folds":       len(fold_results),
        "mean_val_auc":  mean_auc,
        "std_val_auc":   std_auc,
        "best_params":   mode_params,
        "fold_results":  fold_results,
        "model_path":    str(model_path),
        "metadata_path": str(metadata_path),
    }


# ---------------------------------------------------------------------------
# Full training run
# ---------------------------------------------------------------------------

def train_all_default() -> None:
    """
    Train all 8 instruments using default N/T values.

    These defaults are used before multi-combo selection.
    After multi_combo_trainer.py identifies the winning N/T per instrument,
    models are retrained on the optimal combo.
    """
    log.info("=== train_all_default() starting ===")
    summaries = []

    for instrument in ALL_INSTRUMENTS:
        N, T = DEFAULT_NT[instrument]
        log.info("--- Training %s (N=%d, T=%.4f) ---", instrument, N, T)
        try:
            results = train_instrument(instrument, N, T)
            summaries.append(results)
            log.info(
                "  [%s] DONE  folds=%d  mean_AUC=%.4f ± %.4f  params=%s",
                instrument,
                results["n_folds"],
                results["mean_val_auc"],
                results["std_val_auc"],
                results["best_params"],
            )
        except Exception as exc:
            log.error("  [%s] FAILED: %s", instrument, exc)

    log.info("=== train_all_default() complete ===")
    log.info("%-12s  %6s  %10s  %10s  %s", "Instrument", "Folds", "Mean AUC", "Std AUC", "Mode params")
    for s in summaries:
        log.info(
            "%-12s  %6d  %10.4f  %10.4f  %s",
            s["instrument"], s["n_folds"],
            s["mean_val_auc"], s["std_val_auc"],
            s["best_params"],
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_all_default()
