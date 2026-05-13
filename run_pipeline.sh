#!/usr/bin/env bash
# run_pipeline.sh — Full ML pipeline for all 8 instruments
# Run from acieral-kairos-bot root:  bash run_pipeline.sh 2>&1 | tee pipeline.log
set -euo pipefail

VENV="venv/Scripts/activate"
source "$VENV"

echo "================================================================"
echo "ACIERAL KAIROS BOT — FULL ML PIPELINE"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""

# ------------------------------------------------------------------ #
# STEP 1 — Label all instruments across full N/T grid                #
# ------------------------------------------------------------------ #
echo "================================================================"
echo "STEP 1: Label all instruments (N/T grid)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python -c "
from ml.labeler import label_all_grid
label_all_grid()
print('LABELLING COMPLETE')
"

echo ""
echo "Step 1 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ------------------------------------------------------------------ #
# STEP 2 — Multi-combo training and winner selection                  #
# ------------------------------------------------------------------ #
echo "================================================================"
echo "STEP 2: Multi-combo training (top-3 N/T per instrument)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python -c "
from ml.multi_combo_trainer import run_all_multi_combo
import pandas as pd
summary = run_all_multi_combo()
print(summary.to_string())
print('MULTI-COMBO COMPLETE')
"

echo ""
echo "Step 2 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ------------------------------------------------------------------ #
# STEP 3 — Oracle peak-exit labels                                    #
# ------------------------------------------------------------------ #
echo "================================================================"
echo "STEP 3: Exit labelling (oracle peak per trade)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python -c "
from ml.exit_labeler import label_all_exits
label_all_exits()
print('EXIT LABELLING COMPLETE')
"

echo ""
echo "Step 3 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ------------------------------------------------------------------ #
# STEP 4 — Train exit models                                          #
# ------------------------------------------------------------------ #
echo "================================================================"
echo "STEP 4: Train exit models (walk-forward binary XGBoost)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python -c "
from ml.exit_trainer import train_all_exits
train_all_exits()
print('EXIT TRAINING COMPLETE')
"

echo ""
echo "Step 4 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ------------------------------------------------------------------ #
# STEP 5 — Parameter sweep (trail × threshold, all instruments)       #
# ------------------------------------------------------------------ #
echo "================================================================"
echo "STEP 5: Parameter sweep (42 combos × 8 instruments)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python -c "
from backtest.engine_with_exit import run_all_with_exit
run_all_with_exit(split='holdout')
print('BACKTEST SWEEP COMPLETE')
"

echo ""
echo "Step 5 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ------------------------------------------------------------------ #
# STEP 6 — Generate final config                                      #
# ------------------------------------------------------------------ #
echo "================================================================"
echo "STEP 6: Generate final config (DISCOVERED_PARAMS)"
echo "Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"

python -c "
from ml.config_generator import generate_config, print_pruning_report
params = generate_config()
print()
print('=== ACTIVE INSTRUMENTS ===')
print(f'Active: {params[\"ACTIVE_INSTRUMENTS\"]}')
print()
print('=== PER-INSTRUMENT PARAMS ===')
for inst, p in params['INSTRUMENT_PARAMS'].items():
    print(f'{inst}: Sharpe={p[\"holdout_sharpe\"]:.2f}  trail={p[\"trail_atr_mult\"]}  thr={p[\"exit_threshold\"]}  win={p[\"holdout_win_rate\"]:.1%}  P&L=£{p[\"holdout_pnl_gbp\"]:.0f}  DD={p[\"holdout_max_dd\"]:.1%}')
print()
print('=== TOP 20 GLOBAL FEATURES ===')
for i, f in enumerate(params['GLOBAL_FEATURE_RANKING'][:20], 1):
    print(f'  {i:2d}. {f}')
print()
print_pruning_report()
print('CONFIG GENERATION COMPLETE')
"

echo ""
echo "Step 6 finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "================================================================"
echo "PIPELINE COMPLETE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
