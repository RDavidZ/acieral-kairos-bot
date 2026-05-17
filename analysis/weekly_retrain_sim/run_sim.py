"""
analysis/weekly_retrain_sim/run_sim.py

Weekly Retrain Validation — May 12-16, 2026
============================================
Answers the question: would retraining entry models (training cutoff May 9)
produce the same or better performance on last week's 5 trading days?

Phases
------
1.  Setup sandbox cache (copy data/cache/ → sim_cache/)
2.  Fetch OANDA candles up to May 16 (incremental)
3.  Preprocess + build features in sandbox
4.  Retrain entry models — HOLDOUT_START patched to "2026-05-12" so models
    train on all data through May 9 (31 walk-forward folds)
5.  Backtest May 12-16 with NEW retrained models
6.  Backtest May 12-16 with EXISTING April 13 models (comparison baseline)
7.  Side-by-side report

No look-ahead bias:
  - Training cutoff = May 9 (end of week before test week)
  - Test window = May 12-16 (last week, fully out-of-sample)
  - Same exit models used for both runs (exit models not retrained)
  - All feature building sandboxed — zero writes to data/cache/ or ml/models/

Estimated runtime: 8-10 hours (31 CV folds × 8 instruments)

Run from repo root:
  python analysis/weekly_retrain_sim/run_sim.py
"""

import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).parent.parent.parent
SIM_ROOT    = Path(__file__).parent
SIM_CACHE   = SIM_ROOT / "sim_cache"
SIM_MODELS  = SIM_ROOT / "sim_models"
SIM_LABELS  = SIM_ROOT / "sim_labels"
RESULTS_NEW = SIM_ROOT / "results_new"
RESULTS_OLD = SIM_ROOT / "results_old"
DATA_CACHE  = REPO_ROOT / "data" / "cache"
PROD_MODELS = REPO_ROOT / "ml" / "models"

sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_CUTOFF  = "2026-05-12"   # new HOLDOUT_START — train on data before this
SIM_START     = "2026-05-12"   # backtest start (Mon May 12)
SIM_END       = "2026-05-17"   # backtest end exclusive (Sat May 17)
FETCH_THROUGH = "2026-05-17"   # inclusive fetch bound

INSTRUMENTS = [
    "EUR_USD",    # liquid G7 forex
    "USD_JPY",    # yen dynamics, different volatility profile
    "DE30_EUR",   # index — tests whether improvement holds outside forex
]
TIMEFRAMES = ["H1", "H4", "D", "W"]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

for d in (SIM_CACHE, SIM_MODELS, SIM_LABELS, RESULTS_NEW, RESULTS_OLD):
    d.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ---------------------------------------------------------------------------
# PHASE 1 — Copy existing cache into sandbox
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 1 — Seeding sim_cache from data/cache/")
print("=" * 70)

copied = 0
for src in DATA_CACHE.glob("*.parquet"):
    dst = SIM_CACHE / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        copied += 1
        print(f"  copied  {src.name}")

already = sum(1 for _ in SIM_CACHE.glob("*.parquet"))
print(f"  {copied} files copied ({already} total in sim_cache)")

# ---------------------------------------------------------------------------
# Monkey-patch CACHE_DIR in all modules BEFORE importing their functions
# ---------------------------------------------------------------------------

import data.preprocessor       as _prep
import data.fetcher             as _fetcher
import strategy.feature_builder as _fb
import backtest.engine_with_exit as _bwe

for _mod in (_prep, _fetcher, _fb, _bwe):
    _mod.CACHE_DIR = SIM_CACHE

# ---------------------------------------------------------------------------
# PHASE 2 — Fetch OANDA candles up to May 16
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 2 — Fetching OANDA candles (incremental to May 16)")
print("=" * 70)

from data.fetcher import _get_client, _fetch_candles_paginated

fetch_ok = True
try:
    client = _get_client()
    for inst in INSTRUMENTS:
        for tf in TIMEFRAMES:
            path = SIM_CACHE / f"{inst}_{tf}.parquet"
            if not path.exists():
                print(f"  [{inst} {tf}] WARNING: cache missing — skipping")
                continue

            cached = pd.read_parquet(path)
            cached["time"] = pd.to_datetime(cached["time"], utc=True)
            last_ts  = cached["time"].max()
            from_str = (last_ts + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

            new_df = _fetch_candles_paginated(client, inst, tf, from_str)
            if new_df.empty:
                print(f"  [{inst} {tf}] already current (last: {last_ts.date()})")
                continue

            combined = (
                pd.concat([cached, new_df], ignore_index=True)
                  .drop_duplicates(subset=["time"])
                  .sort_values("time")
                  .reset_index(drop=True)
            )
            combined.to_parquet(path, index=False)
            print(
                f"  [{inst} {tf}] +{len(new_df)} bars "
                f"({last_ts.date()} → {combined['time'].max().date()})"
            )
except Exception as exc:
    fetch_ok = False
    print(f"\n  OANDA fetch failed: {exc}")
    print("  Continuing with existing cached data.\n")

# ---------------------------------------------------------------------------
# PHASE 3 — Preprocess + build features (sandboxed)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 3 — Preprocessing + feature build")
print("=" * 70)

from data.preprocessor        import preprocess_instrument
from strategy.feature_builder import build_features

for inst in INSTRUMENTS:
    try:
        preprocess_instrument(inst)
        df_feat = build_features(inst)
        feat_path = SIM_CACHE / f"{inst}_H1_features.parquet"
        df_feat.to_parquet(feat_path, index=False)
        df_feat["time"] = pd.to_datetime(df_feat["time"], utc=True)
        print(
            f"  [{inst}] features: {df_feat['time'].min().date()} → "
            f"{df_feat['time'].max().date()} ({len(df_feat):,} rows)"
        )
    except Exception as exc:
        print(f"  [{inst}] FAILED: {exc}")

# ---------------------------------------------------------------------------
# PHASE 4 — Retrain entry models (HOLDOUT_START = 2026-05-12)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print(f"PHASE 4 — Retraining entry models (cutoff {TRAIN_CUTOFF})")
print(f"          31 walk-forward folds × {len(INSTRUMENTS)} instruments")
print(f"          Estimated runtime: 8-10 hours")
print("=" * 70)

import ml.labeler as _labeler
import ml.trainer as _trainer

# Patch holdout boundary and paths in labeler + trainer
_labeler.HOLDOUT_START = TRAIN_CUTOFF
_labeler.CACHE_DIR     = SIM_CACHE
_labeler.LABELS_DIR    = SIM_LABELS
SIM_LABELS.mkdir(exist_ok=True)

_trainer.HOLDOUT_START = TRAIN_CUTOFF
_trainer.LABELS_DIR    = SIM_LABELS
_trainer.MODELS_DIR    = SIM_MODELS
SIM_MODELS.mkdir(exist_ok=True)

from ml.labeler import label_instrument
from ml.trainer import train_instrument
from config     import DISCOVERED_PARAMS

retrain_summary = {}
total_start = time.time()

for inst in INSTRUMENTS:
    inst_start = time.time()
    print(f"\n  [{inst}] starting...")

    try:
        params = DISCOVERED_PARAMS["INSTRUMENT_PARAMS"][inst]
        N = int(params["N"])
        T = float(params["T"])

        print(f"  [{inst}] labelling N={N} T={T:.4f}...")
        label_instrument(inst, N, T)

        print(f"  [{inst}] training...")
        result = train_instrument(inst, N, T)

        elapsed = (time.time() - inst_start) / 60
        print(
            f"  [{inst}] DONE — AUC={result['mean_auc']:.4f} "
            f"folds={result['n_folds']} ({elapsed:.1f} min)"
        )
        retrain_summary[inst] = result

    except Exception as exc:
        elapsed = (time.time() - inst_start) / 60
        print(f"  [{inst}] FAILED after {elapsed:.1f} min: {exc}")
        retrain_summary[inst] = {"error": str(exc)}

total_elapsed = (time.time() - total_start) / 60
print(f"\n  Total retrain time: {total_elapsed:.1f} min")

# ---------------------------------------------------------------------------
# PHASE 5 — Backtest May 12-16 with NEW retrained models
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 5 — Backtest May 12-16 with NEW models")
print("=" * 70)

# Patch backtest engine to use sim_cache features and new models
_bwe.HOLDOUT_START = TRAIN_CUTOFF   # so split='holdout' gives >= May 12

from backtest.engine_with_exit import run_backtest_with_exit
from config import DISCOVERED_PARAMS as DP

results_new = {}
for inst in INSTRUMENTS:
    entry_path = SIM_MODELS / f"entry_{inst}.pkl"
    exit_path  = PROD_MODELS / f"exit_{inst}.pkl"

    if not entry_path.exists():
        print(f"  [{inst}] SKIP — no new model (retrain failed)")
        continue
    if not exit_path.exists():
        print(f"  [{inst}] SKIP — no exit model found")
        continue

    try:
        params = DP["INSTRUMENT_PARAMS"][inst]
        m = run_backtest_with_exit(
            instrument=inst,
            entry_model_path=str(entry_path),
            exit_model_path=str(exit_path),
            trail_atr_mult=float(params["trail_atr_mult"]),
            exit_threshold=float(params["exit_threshold"]),
            split="holdout",
            end_date=SIM_END,
            _save=True,
            out_dir=RESULTS_NEW,
        )
        results_new[inst] = m
        print(
            f"  [{inst}] trades={m['total_trades']} "
            f"win={m['win_rate']:.1%} pnl=£{m['total_pnl_gbp']:.2f} "
            f"sharpe={m['sharpe_ratio']:.2f}"
        )
    except Exception as exc:
        print(f"  [{inst}] FAILED: {exc}")

# ---------------------------------------------------------------------------
# PHASE 6 — Backtest May 12-16 with EXISTING April 13 models (baseline)
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 6 — Backtest May 12-16 with EXISTING April 13 models (baseline)")
print("=" * 70)

results_old = {}
for inst in INSTRUMENTS:
    entry_path = PROD_MODELS / f"entry_{inst}.pkl"
    exit_path  = PROD_MODELS / f"exit_{inst}.pkl"

    if not entry_path.exists():
        print(f"  [{inst}] SKIP — no existing entry model")
        continue
    if not exit_path.exists():
        print(f"  [{inst}] SKIP — no existing exit model")
        continue

    try:
        params = DP["INSTRUMENT_PARAMS"][inst]
        m = run_backtest_with_exit(
            instrument=inst,
            entry_model_path=str(entry_path),
            exit_model_path=str(exit_path),
            trail_atr_mult=float(params["trail_atr_mult"]),
            exit_threshold=float(params["exit_threshold"]),
            split="holdout",
            end_date=SIM_END,
            _save=True,
            out_dir=RESULTS_OLD,
        )
        results_old[inst] = m
        print(
            f"  [{inst}] trades={m['total_trades']} "
            f"win={m['win_rate']:.1%} pnl=£{m['total_pnl_gbp']:.2f} "
            f"sharpe={m['sharpe_ratio']:.2f}"
        )
    except Exception as exc:
        print(f"  [{inst}] FAILED: {exc}")

# ---------------------------------------------------------------------------
# PHASE 7 — Side-by-side comparison
# ---------------------------------------------------------------------------

print()
print("=" * 70)
print("PHASE 7 — Side-by-side comparison: NEW vs OLD models (May 12-16)")
print("=" * 70)
print()

all_insts = sorted(set(list(results_new.keys()) + list(results_old.keys())))

header = f"{'Instrument':<14} {'Trades':>6}  {'WinRate':>7}  {'P&L £':>8}  {'Sharpe':>7}  |  {'Trades':>6}  {'WinRate':>7}  {'P&L £':>8}  {'Sharpe':>7}"
divider = "-" * len(header)
print(f"{'':14} {'--- NEW RETRAINED ---':^35}  |  {'--- EXISTING (Apr 13) ---':^35}")
print(header)
print(divider)

total_pnl_new = 0.0
total_pnl_old = 0.0
total_trades_new = 0
total_trades_old = 0

for inst in all_insts:
    n = results_new.get(inst)
    o = results_old.get(inst)

    def _fmt(m):
        if m is None:
            return f"{'—':>6}  {'—':>7}  {'—':>8}  {'—':>7}"
        return (
            f"{m['total_trades']:>6}  "
            f"{m['win_rate']:>7.1%}  "
            f"£{m['total_pnl_gbp']:>7.2f}  "
            f"{m['sharpe_ratio']:>7.2f}"
        )

    print(f"{inst:<14} {_fmt(n)}  |  {_fmt(o)}")

    if n:
        total_pnl_new    += n["total_pnl_gbp"]
        total_trades_new += n["total_trades"]
    if o:
        total_pnl_old    += o["total_pnl_gbp"]
        total_trades_old += o["total_trades"]

print(divider)
print(
    f"{'TOTAL':<14} "
    f"{total_trades_new:>6}  {'':>7}  £{total_pnl_new:>7.2f}  {'':>7}  |  "
    f"{total_trades_old:>6}  {'':>7}  £{total_pnl_old:>7.2f}"
)
print()

# Verdict
delta_pnl = total_pnl_new - total_pnl_old
verdict = "NEW BETTER" if delta_pnl > 0 else ("EXISTING BETTER" if delta_pnl < 0 else "EQUAL")
print(f"P&L delta (new - old): £{delta_pnl:+.2f}  →  {verdict}")
print()
print("Retrain AUC summary:")
for inst, r in retrain_summary.items():
    if "error" in r:
        print(f"  {inst:<14} FAILED: {r['error']}")
    else:
        old_auc = DP["INSTRUMENT_PARAMS"].get(inst, {}).get("mean_auc", float("nan"))
        new_auc = r.get("mean_auc", float("nan"))
        delta   = new_auc - old_auc if old_auc == old_auc else float("nan")
        arrow   = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(
            f"  {inst:<14} AUC {old_auc:.4f} → {new_auc:.4f} "
            f"({delta:+.4f} {arrow})"
        )

print()
print("Done. Results written to:")
print(f"  New models:  {RESULTS_NEW}")
print(f"  Old models:  {RESULTS_OLD}")
print(f"  New models saved to: {SIM_MODELS}")
