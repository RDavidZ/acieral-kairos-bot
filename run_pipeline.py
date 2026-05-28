"""
run_pipeline.py — Full rebuild pipeline (synthetic HTF, 2017-2024 training).

Training  : 2017-01-01 → 2024-12-31
Holdout 1 : 2025-01-01 → 2025-12-31  (model selection)
Holdout 2 : 2026-01-01 → present      (SEALED — do not open)

Run from repo root:
  venv/Scripts/python run_pipeline.py
"""

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import HARD_CONSTRAINTS

CACHE_DIR = REPO_ROOT / "data" / "cache"
instruments = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]

log.info("=== Pipeline start ===")
log.info("Training  : %s -> %s", HARD_CONSTRAINTS["TRAINING_START"], HARD_CONSTRAINTS["TRAINING_END"])
log.info("Holdout 1 : %s -> %s", HARD_CONSTRAINTS["HOLDOUT_1_START"], HARD_CONSTRAINTS["HOLDOUT_1_END"])
log.info("Holdout 2 : %s -> present (SEALED)", HARD_CONSTRAINTS["HOLDOUT_2_START"])

t0 = time.time()


def elapsed():
    return time.time() - t0


# ---------------------------------------------------------------------------
# Step 1 — Preprocess + feature caches (all 8 instruments)
# ---------------------------------------------------------------------------
log.info("--- STEP 1: Preprocess + feature caches ---")
from data.preprocessor import preprocess_instrument
from strategy.feature_builder import build_features

for inst in instruments:
    log.info("[%s] preprocessing...", inst)
    df = preprocess_instrument(inst)
    log.info("[%s] building features...", inst)
    features = build_features(inst, df)
    out = CACHE_DIR / f"{inst}_H1_features.parquet"
    features.to_parquet(out)
    log.info("[%s] %d rows x %d cols -> %s", inst, len(features), len(features.columns), out.name)

log.info("Step 1 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 2 — Labeler grid search (training: 2017-2024)
# ---------------------------------------------------------------------------
log.info("--- STEP 2: Labeler grid search (training 2017-2024) ---")
from ml.labeler import label_all_grid
label_all_grid()
log.info("Step 2 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 3 — Walk-forward trainer (training: 2017-2024)
# ---------------------------------------------------------------------------
log.info("--- STEP 3: Walk-forward trainer ---")
from ml.trainer import train_all_default
train_all_default()
log.info("Step 3 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 4 — Entry-only backtest on training data (2017-2024)
# ---------------------------------------------------------------------------
log.info("--- STEP 4: Entry-only backtest (training data, split=train) ---")
from backtest.engine import run_all_backtests
run_all_backtests(split="train")
log.info("Step 4 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 5 — Multi-combo trainer (holdout 1: 2025)
# ---------------------------------------------------------------------------
log.info("--- STEP 5: Multi-combo trainer (holdout 1: 2025) ---")
from ml.multi_combo_trainer import run_all_multi_combo
run_all_multi_combo()
log.info("Step 5 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 6 — Exit labeler
# ---------------------------------------------------------------------------
log.info("--- STEP 6: Exit labeler ---")
from ml.exit_labeler import label_all_exits
label_all_exits()
log.info("Step 6 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 7 — Exit trainer
# ---------------------------------------------------------------------------
log.info("--- STEP 7: Exit trainer ---")
from ml.exit_trainer import train_all_exits
train_all_exits()
log.info("Step 7 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 8 — Backtest with exit model (holdout 1: 2025)
# ---------------------------------------------------------------------------
log.info("--- STEP 8: Backtest with exit model (holdout 1: 2025, split=holdout) ---")
from backtest.engine_with_exit import run_all_with_exit
run_all_with_exit(split="holdout")
log.info("Step 8 complete (%.0fs)", elapsed())

# ---------------------------------------------------------------------------
# Step 9 — Config generator → update DISCOVERED_PARAMS
# ---------------------------------------------------------------------------
log.info("--- STEP 9: Config generator ---")
from ml.config_generator import generate_config, print_pruning_report
generate_config()
print_pruning_report()
log.info("Step 9 complete (%.0fs)", elapsed())

log.info("=== Pipeline complete in %.0fs (%.1fh) ===", elapsed(), elapsed() / 3600)
