# Pipeline Rebuild Status — 2026-05-24

## What we're doing
Full retrain from scratch: synthetic HTF feature builder, training 2017-2024, holdout 1 = 2025, holdout 2 = 2026 (sealed).

## Steps completed ✅
- Step 1 — Feature caches: all 8 instruments (`data/cache/*_H1_features.parquet`)
- Step 2 — Labeler grid search: all 8 instruments (`ml/labels_train/`)
- Step 3 — Walk-forward trainer: all 8 instruments (`ml/models/entry_{instrument}.pkl`)
- Step 4 — Training backtest: all 8 instruments (`backtest/results_train/`)
- Step 5 — Multi-combo trainer: EUR_USD ✅, GBP_USD ✅, AUD_USD ✅, USD_JPY ✅
  - combo_results.csv saved for all 4 above
  - USD_CHF, SPX500_USD, NAS100_USD, DE30_EUR: NOT YET DONE

## Steps remaining ❌
- Step 5 (resume) — Multi-combo trainer: USD_CHF, SPX500_USD, NAS100_USD, DE30_EUR
- Step 6 — Exit labeler
- Step 7 — Exit trainer
- Step 8 — Holdout 1 backtest with exit model
- Step 9 — Config generator

## How to resume
Run from repo root (with venv activated):
```
venv/Scripts/python run_pipeline_resume.py
```

This script starts at Step 5. Already-completed combo pkl files are cached and will be skipped automatically. USD_CHF N4 combo pkl may be present (check `ml/models/entry_USD_CHF_N4_T5bps.pkl`) — if so, that combo will skip training.

## Code changes made (do NOT revert)
Both files patched for parallelisation — **do not revert**:

### `ml/trainer.py`
- Added `import threading`
- Added `_metadata_lock = threading.Lock()` (module-level)
- Added `n_jobs=4` to `XGB_FIXED` — limits XGBoost to 4 threads per model
- `metadata.json` read-modify-write wrapped with `_metadata_lock`

### `ml/multi_combo_trainer.py`
- Added `import threading`, `from concurrent.futures import ThreadPoolExecutor, as_completed`
- Imports `_metadata_lock` from `ml.trainer`
- `run_all_multi_combo()` now runs 3 instruments simultaneously via `ThreadPoolExecutor(max_workers=3)`
- `metadata.json` write in `run_multi_combo()` wrapped with `_metadata_lock`
- `summary_rows.append()` protected by a local `summary_lock`

**Effect:** 3 instruments train in parallel, each using 4 XGBoost threads → 12 cores fully utilised.
Previously sequential, ~16h remaining. With parallelisation: ~5-6h remaining.

## Script locations
- `run_pipeline_resume.py` — resume script (Steps 5-9), use this
- `run_pipeline.py` — full pipeline from scratch (Steps 1-9), do NOT use
