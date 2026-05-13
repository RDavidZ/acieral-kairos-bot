"""
smoke_test_features.py — Verify feature cache and candle fetch pipeline for EUR_USD.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

SEP = "=" * 60

# ---------------------------------------------------------------------------
# Test 1: preprocess_instrument
# ---------------------------------------------------------------------------
print(SEP)
print("TEST 1: preprocess_instrument('EUR_USD')")
print(SEP)

from data.preprocessor import preprocess_instrument, preprocess

df_proc = preprocess("EUR_USD")
print(f"  Rows returned  : {len(df_proc)}")
print(f"  Has atr_14     : {'atr_14' in df_proc.columns}")
print(f"  Date range     : {df_proc['time'].min().date()} to{df_proc['time'].max().date()}")

assert len(df_proc) > 100, f"FAIL: only {len(df_proc)} rows returned"
assert "atr_14" in df_proc.columns, "FAIL: atr_14 column missing"
print("  PASS\n")

# ---------------------------------------------------------------------------
# Test 2: feature cache parquet
# ---------------------------------------------------------------------------
print(SEP)
print("TEST 2: feature cache parquet")
print(SEP)

from ml.labeler import FEATURE_COLS

cache_path = Path(__file__).parent / "data" / "cache" / "EUR_USD_H1_features.parquet"
assert cache_path.exists(), f"FAIL: cache not found at {cache_path}"

feat_df = pd.read_parquet(cache_path)
feat_df["time"] = pd.to_datetime(feat_df["time"], utc=True)

print(f"  Rows in cache  : {len(feat_df)}")
print(f"  Columns        : {len(feat_df.columns)}")
print(f"  Date range     : {feat_df['time'].min().date()} to{feat_df['time'].max().date()}")

missing_cols = [c for c in FEATURE_COLS if c not in feat_df.columns]
print(f"  Missing FEATURE_COLS: {missing_cols if missing_cols else 'none'}")

assert len(feat_df) > 100, f"FAIL: only {len(feat_df)} rows in cache"
assert not missing_cols, f"FAIL: missing columns: {missing_cols}"
print("  PASS\n")

# ---------------------------------------------------------------------------
# Test 3: build_live_features_v2
# ---------------------------------------------------------------------------
print(SEP)
print("TEST 3: build_live_features_v2('EUR_USD', h1_df=None, now=None)")
print(SEP)

from strategy.feature_builder import build_live_features_v2

# build_live_features_v2 ignores h1_df when cache exists; pass a minimal placeholder
row = build_live_features_v2("EUR_USD", live_h1_df=pd.DataFrame(), now=None)

print(f"  Type returned  : {type(row).__name__}")

critical_cols = ["atr_14", "close", "rsi_14", "hour_utc"]
for col in critical_cols:
    val = row.get(col) if hasattr(row, "get") else row[col]
    is_nan = pd.isna(val)
    status = "NaN — FAIL" if is_nan else f"{val}"
    print(f"  {col:<12} : {status}")

for col in critical_cols:
    val = row.get(col) if hasattr(row, "get") else row[col]
    assert not pd.isna(val), f"FAIL: {col} is NaN in latest feature row"

print("  PASS\n")

# ---------------------------------------------------------------------------
# Test 4: summary of latest feature row
# ---------------------------------------------------------------------------
print(SEP)
print("TEST 4: Latest feature row summary")
print(SEP)

ts = row.get("time", "unknown") if hasattr(row, "get") else row["time"]
close   = row["close"]
atr_14  = row["atr_14"]
rsi_14  = row["rsi_14"]
hour    = row["hour_utc"]

print(f"  timestamp : {ts}")
print(f"  close     : {close}")
print(f"  atr_14    : {atr_14:.6f}")
print(f"  rsi_14    : {rsi_14:.2f}")
print(f"  hour_utc  : {int(hour)}")
print()
print("All tests PASSED.")
