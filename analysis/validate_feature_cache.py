"""
analysis/validate_feature_cache.py

Validate that build_live_features_v2 (cache-based) produces the same
trade signals as the full backtest engine.

Compares:
- Existing results: backtest/results_2026/{instrument}_trades.csv
- Cache-based:      scoring each backtest entry bar using the H1 features
                    parquet (same data source the live bot will use via v2)

For each backtest trade in 2026, scores the entry bar from the features
cache and checks: same direction? same confidence? same best bar that day?
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from ml.labeler import FEATURE_COLS
import pickle

RESULTS_DIR = Path(__file__).parent.parent / "backtest" / "results_2026"
CACHE_DIR   = Path(__file__).parent.parent / "data" / "cache"
MODELS_DIR  = Path(__file__).parent.parent / "ml" / "models"

INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF",
    "SPX500_USD", "NAS100_USD", "DE30_EUR",
]

SEP = "=" * 80


def score_row(model, row: pd.Series) -> tuple[int, float]:
    """Score one feature row. Returns (predicted_class, confidence)."""
    X = np.nan_to_num(row[FEATURE_COLS].values.reshape(1, -1).astype(np.float32), nan=0.0)
    probas = model.predict_proba(X)[0]
    pred   = int(np.argmax(probas))
    conf   = float(probas[pred])
    return pred, conf


def class_to_dir(c: int) -> str:
    return "LONG" if c == 2 else "SHORT" if c == 1 else "NO_TRADE"


print(SEP)
print("VALIDATION: build_live_features_v2 vs full backtest engine (2026 OOS)")
print(SEP)
print()

summary_rows = []

for instrument in INSTRUMENTS:
    bt_path    = RESULTS_DIR / f"{instrument}_trades.csv"
    cache_path = CACHE_DIR   / f"{instrument}_H1_features.parquet"
    model_path = MODELS_DIR  / f"entry_{instrument}.pkl"

    if not bt_path.exists():
        print(f"[{instrument}] SKIP — no backtest trades file")
        continue
    if not cache_path.exists():
        print(f"[{instrument}] SKIP — no feature cache")
        continue

    # Load backtest trades (2026 only)
    bt = pd.read_csv(bt_path, parse_dates=["entry_time"])
    bt["entry_time"] = pd.to_datetime(bt["entry_time"], utc=True)
    bt["entry_date"] = bt["entry_time"].dt.date

    # Load feature cache
    cache = pd.read_parquet(cache_path)
    cache["time"] = pd.to_datetime(cache["time"], utc=True)
    cache["date"] = cache["time"].dt.date

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    n_total           = len(bt)
    n_same_bar        = 0
    n_dir_match       = 0
    n_best_bar_match  = 0
    conf_diffs        = []
    mismatches        = []

    for _, trade in bt.iterrows():
        bt_date   = trade["entry_date"]
        bt_hour   = trade["entry_time"].hour
        bt_dir    = trade["direction"]          # "LONG" or "SHORT"
        bt_conf   = float(trade["confidence"])

        # Get all cache rows for this date
        day_rows = cache[cache["date"] == bt_date]
        if day_rows.empty:
            mismatches.append({
                "date": bt_date, "bt_dir": bt_dir, "bt_hour": bt_hour,
                "issue": "no cache rows for this date",
            })
            continue

        # Score the exact bar the backtest used (matched by hour)
        exact = day_rows[day_rows["time"].dt.hour == bt_hour]
        if not exact.empty:
            pred_class, pred_conf = score_row(model, exact.iloc[0])
            pred_dir = class_to_dir(pred_class)
            conf_diff = abs(pred_conf - bt_conf)
            conf_diffs.append(conf_diff)

            if pred_dir == bt_dir:
                n_same_bar += 1
                n_dir_match += 1
            else:
                mismatches.append({
                    "date":     bt_date,
                    "bt_dir":   bt_dir,
                    "bt_hour":  bt_hour,
                    "bt_conf":  bt_conf,
                    "v2_dir":   pred_dir,
                    "v2_conf":  pred_conf,
                    "issue":    "direction mismatch on exact bar",
                })
        else:
            mismatches.append({
                "date": bt_date, "bt_dir": bt_dir, "bt_hour": bt_hour,
                "issue": f"exact hour {bt_hour} not in cache for this date",
            })

        # Find best signal across the full day (what would the live bot pick?)
        best_class, best_conf, best_hour = 0, 0.0, -1
        for _, row in day_rows.iterrows():
            pc, conf = score_row(model, row)
            if pc != 0 and conf > best_conf:
                best_class, best_conf, best_hour = pc, conf, row["time"].hour

        if best_class != 0 and class_to_dir(best_class) == bt_dir:
            n_best_bar_match += 1

    avg_conf_diff = float(np.mean(conf_diffs)) if conf_diffs else float("nan")

    print(f"[{instrument}]")
    print(f"  Backtest trades (2026)       : {n_total}")
    print(f"  Same bar + direction match   : {n_same_bar} / {n_total} "
          f"({n_same_bar/n_total*100:.1f}%)")
    print(f"  Best-bar direction match     : {n_best_bar_match} / {n_total} "
          f"({n_best_bar_match/n_total*100:.1f}%)")
    print(f"  Avg confidence difference    : {avg_conf_diff:.4f}")

    if mismatches:
        print(f"  Mismatches ({len(mismatches)}):")
        for m in mismatches[:5]:   # show first 5
            issue  = m.get("issue", "")
            v2_dir = m.get("v2_dir", "-")
            v2_c   = m.get("v2_conf", 0.0)
            bt_c   = m.get("bt_conf", 0.0)
            print(f"    {m['date']} h{m['bt_hour']:02d}: bt={m['bt_dir']}({bt_c:.3f}) "
                  f"v2={v2_dir}({v2_c:.3f}) [{issue}]")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches)-5} more")
    print()

    summary_rows.append({
        "instrument":        instrument,
        "n_trades":          n_total,
        "same_bar_pct":      n_same_bar / n_total * 100 if n_total else 0,
        "best_bar_pct":      n_best_bar_match / n_total * 100 if n_total else 0,
        "avg_conf_diff":     avg_conf_diff,
        "n_mismatches":      len(mismatches),
    })

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print(SEP)
print("SUMMARY TABLE")
print(SEP)

hdr = (f"{'Instrument':<14} {'Trades':>6} {'SameBar%':>9} {'BestBar%':>9} "
       f"{'AvgConfDiff':>12} {'Mismatches':>11}")
print(hdr)
print("-" * 65)

for r in summary_rows:
    print(f"{r['instrument']:<14} {r['n_trades']:>6} {r['same_bar_pct']:>9.1f} "
          f"{r['best_bar_pct']:>9.1f} {r['avg_conf_diff']:>12.4f} "
          f"{r['n_mismatches']:>11}")

total_trades   = sum(r["n_trades"]   for r in summary_rows)
avg_same_bar   = np.mean([r["same_bar_pct"]  for r in summary_rows])
avg_best_bar   = np.mean([r["best_bar_pct"]  for r in summary_rows])
avg_conf_diff  = np.mean([r["avg_conf_diff"] for r in summary_rows
                           if not np.isnan(r["avg_conf_diff"])])

print("-" * 65)
print(f"{'OVERALL':<14} {total_trades:>6} {avg_same_bar:>9.1f} "
      f"{avg_best_bar:>9.1f} {avg_conf_diff:>12.4f}")
print()

threshold = 90.0
if avg_same_bar >= threshold:
    print(f"PASS — avg same-bar direction match {avg_same_bar:.1f}% >= {threshold:.0f}%")
    print("Cache-based feature approach reproduces backtest signals correctly.")
else:
    print(f"FAIL — avg same-bar direction match {avg_same_bar:.1f}% < {threshold:.0f}%")
    print("Cache-based features diverge from backtest. Investigate mismatches above.")
print()
