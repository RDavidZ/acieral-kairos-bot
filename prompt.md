In `acieral-kairos-bot/`, delete the following files and directories from the working tree on the main branch. These are all safely archived on `archive/pre-synthetic-htf-rebuild-2026-05-23`.

Delete:
```bash
# Trained models
rm -rf ml/models/entry_*.pkl ml/models/exit_*.pkl ml/models/metadata.json ml/models/exit_metadata.json

# Labels
rm -rf ml/labels/ ml/labels_train/ ml/exit_labels/

# Multi-combo results
rm -rf ml/multi_combo_results/

# Backtest results
rm -rf backtest/results/ backtest/results_train/ backtest/results_exit/

# HTF raw parquets (H4, D1, W1) — H1 stays
find data/cache -name "*_H4*.parquet" -delete
find data/cache -name "*_D1*.parquet" -delete
find data/cache -name "*_W1*.parquet" -delete
find data/cache -name "*_D.parquet" -delete
find data/cache -name "*_W.parquet" -delete

# Feature cache parquets — will be rebuilt
find data/cache -name "*_features.parquet" -delete

# Archive folder on main (not needed here, lives on archive branch)
rm -rf archive/
```

Then commit the deletions to main:
```bash
git add -A
git commit -m "chore: remove pre-rebuild artifacts from main

Models, labels, backtest results, HTF parquets and feature caches
deleted. All preserved on archive/pre-synthetic-htf-rebuild-2026-05-23.
H1 processed parquets kept — foundation for synthetic HTF rebuild.
"
```

Verify what remains in key directories:
```bash
ls data/cache/
ls ml/models/ 2>/dev/null || echo "ml/models empty or gone"
ls ml/ 
```

Report what was deleted and what remains. Do not touch H1 parquets or any Python source files.