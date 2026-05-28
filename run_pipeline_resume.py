"""
run_pipeline_resume.py — Resume from Step 5 after crash.

Steps 1-4 already complete. Picks up from multi-combo trainer.

Run from repo root:
  venv/Scripts/python run_pipeline_resume.py
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

log.info("=== Pipeline RESUME (Steps 5-9) ===")
log.info("Training  : %s -> %s", HARD_CONSTRAINTS["TRAINING_START"], HARD_CONSTRAINTS["TRAINING_END"])
log.info("Holdout 1 : %s -> %s", HARD_CONSTRAINTS["HOLDOUT_1_START"], HARD_CONSTRAINTS["HOLDOUT_1_END"])
log.info("Holdout 2 : %s -> present (SEALED)", HARD_CONSTRAINTS["HOLDOUT_2_START"])

t0 = time.time()


def elapsed():
    return time.time() - t0


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

log.info("=== Resume complete in %.0fs (%.1fh) ===", elapsed(), elapsed() / 3600)
