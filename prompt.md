Fix the feature_builder.py divergence between the FTMO bot and OANDA bot.

SSH access:
- Oracle VPS: ubuntu@132.145.33.221
- IONOS VPS: Administrator@212.227.210.56

Step 1: Read both feature_builder.py files in full:
- Oracle: /home/ubuntu/acieral-kairos-bot/strategy/feature_builder.py
- IONOS: C:\projects\kairos-ftmo\strategy\feature_builder.py

Step 2: Read the FTMO preprocessor to understand data structures it produces:
- IONOS: C:\projects\kairos-ftmo\data\preprocessor.py

Step 3: Replace C:\projects\kairos-ftmo\strategy\feature_builder.py with the Oracle version, making only the minimal changes needed to make it compatible with the FTMO preprocessor's data structures. Specifically:
- Keep Oracle's NaN handling for FVG columns (do not fill with 0.0)
- Keep Oracle's HTF loading from parquet files (not _htf_raw_cache)
- Ensure file paths and cache directory references match the FTMO preprocessor's conventions

Step 4: On the IONOS VPS, run this smoke test:
cd C:\projects\kairos-ftmo
$env:PYTHONPATH='C:\projects\kairos-ftmo'
.\venv\Scripts\python.exe -c "
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from data.preprocessor import load_cached_features, get_cache_dir

cache_dir = get_cache_dir(1)
feat = load_cached_features('EUR_USD', cache_dir=cache_dir)
print(f'Features: {len(feat)}')
print(f'NaNs: {feat.isna().sum()}')
fvg_cols = [c for c in feat.index if 'fvg' in c.lower()]
print(f'FVG cols: {fvg_cols}')
print(f'FVG values: {feat[fvg_cols].to_dict()}')
print('OK')
"

Step 5: Verify MD5 of feature_builder.py matches Oracle:
- Oracle MD5: dab66b1a2f13607965af265e5f9104df
- Check IONOS MD5 after replacement

Show full output. Stop and wait for instructions.