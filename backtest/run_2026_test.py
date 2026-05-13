"""
backtest/run_2026_test.py — 2026 out-of-sample backtest for Acieral Kairos Bot

Executes all 6 steps when run with:
    python -m backtest.run_2026_test

Steps
-----
  1. Fetch 2026 OANDA candles + supplementary (VIX/SPY) → data/cache_2026/
  2. Preprocess H1 data for each instrument             → data/cache_2026/*_H1_processed.parquet
  3. Build all ~80 features                             → data/cache_2026/*_H1_features.parquet
  4. Run backtest with winner params (models unchanged) → backtest/results_2026/
  5. Compare with live Supabase practice trades (Apr 7-20) → results_2026/live_vs_backtest_comparison.csv
  6. Print summary report

IMPORTANT: This script does NOT modify any existing files, does NOT push to VPS,
           and does NOT write to Supabase (step 5 is read-only).
"""

import csv
import json
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Ensure project root is on sys.path when run as a module ──────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import HARD_CONSTRAINTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────────────────
CACHE_LIVE_DIR   = ROOT / "data" / "cache"
CACHE_2026_DIR   = ROOT / "data" / "cache_2026"
RESULTS_2026_DIR = ROOT / "backtest" / "results_2026"
MODELS_DIR       = ROOT / "ml" / "models"
WINNER_PARAMS_PATH = ROOT / "backtest" / "results_exit" / "winner_params.json"

CACHE_2026_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_2026_DIR.mkdir(parents=True, exist_ok=True)

# ── Date range ────────────────────────────────────────────────────────────────
FETCH_FROM_2026 = "2026-01-01T00:00:00Z"
TODAY_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TODAY_TS  = pd.Timestamp(TODAY_STR, tz="UTC")

# ── Instrument lists ──────────────────────────────────────────────────────────
ALL_INSTRUMENTS   = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
FOREX_PAIRS       = HARD_CONSTRAINTS["FOREX_PAIRS"]
INDEX_INSTRUMENTS = HARD_CONSTRAINTS["INDEX_INSTRUMENTS"]
TIMEFRAMES        = ["H1", "H4", "D", "W"]

# ── Live-trade comparison window ──────────────────────────────────────────────
BOT_ID          = "acieral_kairos_v1"
LIVE_WINDOW_DAYS = 14   # last 14 days (Apr 7-20 2026)
MATCH_HOURS      = 2    # ±2 hours to count as a signal match


# =============================================================================
# STEP 1 — Fetch 2026 data
# =============================================================================

def _fetch_supplementary_to_2026() -> None:
    """Fetch full-history VIX close and SPY volume to cache_2026/.

    We fetch the full historical range (2017 → today) so the rolling-20-day
    average in the feature builder has enough warm-up context.
    """
    import yfinance as yf

    log.info("[Step 1] Fetching VIX and SPY supplementary data …")

    def _flatten(raw: pd.DataFrame) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix_raw = yf.download("^VIX", start="2017-01-01", end=TODAY_STR,
                          progress=False, auto_adjust=True)
    vix_raw = _flatten(vix_raw)
    vix_df = vix_raw[["Close"]].reset_index()
    vix_df.columns = ["date", "vix_close"]
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
    vix_df = vix_df.dropna(subset=["vix_close"]).reset_index(drop=True)
    vix_df["vix_close"] = vix_df["vix_close"].astype(float).round(2)
    vix_path = CACHE_2026_DIR / "vix_daily.parquet"
    vix_df.to_parquet(vix_path, index=False)
    log.info("  VIX: %d rows → %s", len(vix_df), vix_path.name)

    # ── SPY volume ────────────────────────────────────────────────────────────
    spy_raw = yf.download("SPY", start="2017-01-01", end=TODAY_STR,
                          progress=False, auto_adjust=True)
    spy_raw = _flatten(spy_raw)
    spy_df = spy_raw[["Volume"]].reset_index()
    spy_df.columns = ["date", "spy_volume"]
    spy_df["date"] = pd.to_datetime(spy_df["date"]).dt.date
    spy_df = spy_df.dropna(subset=["spy_volume"]).reset_index(drop=True)
    spy_df["spy_volume"] = spy_df["spy_volume"].astype("int64")
    spy_path = CACHE_2026_DIR / "spy_volume_daily.parquet"
    spy_df.to_parquet(spy_path, index=False)
    log.info("  SPY volume: %d rows → %s", len(spy_df), spy_path.name)


def step1_fetch() -> None:
    """Fetch 2026 OANDA candles for all 8 instruments across H1/H4/D/W."""
    log.info("=" * 60)
    log.info("STEP 1 — Fetching 2026 OANDA data")
    log.info("=" * 60)

    import data.fetcher as _fetcher

    # Monkey-patch to redirect saves to cache_2026/ and start from 2026-01-01.
    # Python functions read module globals at call time, so patching the module
    # attribute is sufficient.
    _orig_cache     = _fetcher.CACHE_DIR
    _orig_fetch_from = _fetcher.FETCH_FROM

    _fetcher.CACHE_DIR   = CACHE_2026_DIR
    _fetcher.FETCH_FROM  = FETCH_FROM_2026

    try:
        _fetcher.fetch_all()
    finally:
        _fetcher.CACHE_DIR   = _orig_cache
        _fetcher.FETCH_FROM  = _orig_fetch_from

    # Supplementary (VIX/SPY) — uses yfinance, not OANDA
    _fetch_supplementary_to_2026()

    log.info("[Step 1] Done.\n")


# =============================================================================
# STEP 2 — Preprocess 2026 data
# =============================================================================

def step2_preprocess() -> None:
    """Run the preprocessor on 2026 H1 data for each instrument."""
    log.info("=" * 60)
    log.info("STEP 2 — Preprocessing 2026 data")
    log.info("=" * 60)

    import data.preprocessor as _prep

    _orig_cache = _prep.CACHE_DIR
    _prep.CACHE_DIR = CACHE_2026_DIR

    try:
        for instrument in ALL_INSTRUMENTS:
            log.info("  Preprocessing %s …", instrument)
            df = _prep.preprocess(instrument)
            out = CACHE_2026_DIR / f"{instrument}_H1_processed.parquet"
            df.to_parquet(out, index=False)
            log.info("  [%s] %d processed rows → %s", instrument, len(df), out.name)
    finally:
        _prep.CACHE_DIR = _orig_cache

    log.info("[Step 2] Done.\n")


# =============================================================================
# STEP 3 — Build features for 2026
# =============================================================================

def step3_features() -> None:
    """Build the full ~80-feature set for each instrument using 2026 data."""
    log.info("=" * 60)
    log.info("STEP 3 — Building features for 2026")
    log.info("=" * 60)

    import strategy.feature_builder as _fb

    # feature_builder.CACHE_DIR governs all raw parquet reads (H1/H4/D/W,
    # H1_processed, supplementary) and is accessed at call time as a global.
    _orig_cache = _fb.CACHE_DIR
    _fb.CACHE_DIR = CACHE_2026_DIR

    try:
        for instrument in ALL_INSTRUMENTS:
            log.info("  Building features for %s …", instrument)
            df = _fb.build_features(instrument)
            out = CACHE_2026_DIR / f"{instrument}_H1_features.parquet"
            df.to_parquet(out, index=False)
            log.info(
                "  [%s] %d rows × %d cols → %s",
                instrument, len(df), len(df.columns), out.name,
            )
    finally:
        _fb.CACHE_DIR = _orig_cache

    log.info("[Step 3] Done.\n")


# =============================================================================
# STEP 4 — Run backtest with winner params
# =============================================================================

def step4_backtest() -> dict:
    """
    Run engine_with_exit for each instrument using:
      - 2026 features from data/cache_2026/
      - existing trained models from ml/models/
      - winner params from backtest/results_exit/winner_params.json

    Saves per-instrument trades CSV + metrics JSON to backtest/results_2026/.
    Returns {instrument: metrics_dict} for all 8 instruments.
    """
    log.info("=" * 60)
    log.info("STEP 4 — Running 2026 backtest")
    log.info("=" * 60)

    if not WINNER_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"winner_params.json not found at {WINNER_PARAMS_PATH}. "
            "Run backtest/engine_with_exit.py first."
        )
    with open(WINNER_PARAMS_PATH) as fh:
        winner_params: dict = json.load(fh)

    import backtest.engine as _engine
    import backtest.engine_with_exit as _engine_exit

    # Patch all three module-level path constants that the engine uses.
    # engine_with_exit.CACHE_DIR governs feat_path loading.
    # engine.CACHE_DIR governs _merge_rate (GBP_USD / EUR_USD rate files).
    # engine_with_exit.RESULTS_EXIT_DIR governs where trades/metrics are saved.
    _orig_engine_cache       = _engine.CACHE_DIR
    _orig_exit_cache         = _engine_exit.CACHE_DIR
    _orig_exit_results       = _engine_exit.RESULTS_EXIT_DIR

    _engine.CACHE_DIR              = CACHE_2026_DIR
    _engine_exit.CACHE_DIR         = CACHE_2026_DIR
    _engine_exit.RESULTS_EXIT_DIR  = RESULTS_2026_DIR

    all_metrics: dict = {}

    try:
        for instrument in ALL_INSTRUMENTS:
            entry_model = MODELS_DIR / f"entry_{instrument}.pkl"
            exit_model  = MODELS_DIR / f"exit_{instrument}.pkl"

            if not entry_model.exists():
                log.warning("[%s] Entry model missing — skipping.", instrument)
                continue
            if not exit_model.exists():
                log.warning("[%s] Exit model missing — skipping.", instrument)
                continue

            params     = winner_params.get(instrument, {})
            trail_mult = float(params.get("trail_atr_mult", 2.0))
            exit_thr   = float(params.get("exit_threshold", 0.35))

            log.info(
                "  [%s] trail=%.1f thr=%.2f …", instrument, trail_mult, exit_thr
            )

            try:
                # split='holdout' filters df["time"] >= 2024-01-01.
                # Since cache_2026/ only holds 2026 data, all rows pass the filter.
                m = _engine_exit.run_backtest_with_exit(
                    instrument,
                    str(entry_model),
                    str(exit_model),
                    trail_atr_mult=trail_mult,
                    exit_threshold=exit_thr,
                    split="holdout",
                    _save=True,
                )
                all_metrics[instrument] = m
            except Exception as exc:
                log.error("[%s] Backtest failed: %s", instrument, exc, exc_info=True)

    finally:
        _engine.CACHE_DIR              = _orig_engine_cache
        _engine_exit.CACHE_DIR         = _orig_exit_cache
        _engine_exit.RESULTS_EXIT_DIR  = _orig_exit_results

    log.info("[Step 4] Done.\n")
    return all_metrics


# =============================================================================
# STEP 5 — Compare with live Supabase practice trades
# =============================================================================

def _load_backtest_trades_2026() -> dict[str, list[dict]]:
    """Load all per-instrument backtest trades from backtest/results_2026/."""
    trades: dict[str, list[dict]] = {}
    for instrument in ALL_INSTRUMENTS:
        path = RESULTS_2026_DIR / f"{instrument}_trades.csv"
        if not path.exists():
            trades[instrument] = []
            continue
        df = pd.read_csv(path, parse_dates=["entry_time"])
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        trades[instrument] = df.to_dict("records")
    return trades


def step5_compare() -> list[dict]:
    """
    Load live practice trades from Supabase (last LIVE_WINDOW_DAYS days) and
    compare each with 2026 backtest signals.

    Match rule: same instrument, backtest signal within ±MATCH_HOURS of live entry.
    Returns list of comparison rows.
    """
    log.info("=" * 60)
    log.info("STEP 5 — Comparing live trades with backtest signals")
    log.info("=" * 60)

    # ── Load live trades (read-only Supabase call) ────────────────────────────
    live_trades: list[dict] = []
    cutoff_ts = datetime.now(timezone.utc) - timedelta(days=LIVE_WINDOW_DAYS)
    cutoff_str = cutoff_ts.strftime("%Y-%m-%d")

    try:
        from dashboard.db import get_trades as _db_get_trades
        log.info("  Querying Supabase for practice trades since %s …", cutoff_str)
        raw_live = _db_get_trades(
            bot_id=BOT_ID,
            limit=500,
            status="closed",
            trade_type="practice",
        )
        # Filter to within the comparison window
        for t in raw_live:
            entry_raw = t.get("entry_time")
            if entry_raw is None:
                continue
            if isinstance(entry_raw, str):
                entry_ts = pd.Timestamp(entry_raw, tz="UTC")
            else:
                entry_ts = pd.Timestamp(entry_raw).tz_localize("UTC") \
                    if getattr(entry_raw, "tzinfo", None) is None \
                    else pd.Timestamp(entry_raw)
            if entry_ts >= pd.Timestamp(cutoff_ts):
                live_trades.append({**t, "_entry_ts": entry_ts})
        log.info("  Found %d live practice trades in window.", len(live_trades))
    except Exception as exc:
        log.warning("  Supabase query failed (%s) — comparison will be empty.", exc)

    # ── Load 2026 backtest trades ─────────────────────────────────────────────
    bt_trades = _load_backtest_trades_2026()
    delta = timedelta(hours=MATCH_HOURS)

    # ── Build comparison rows ─────────────────────────────────────────────────
    comparison: list[dict] = []

    for lt in live_trades:
        instrument  = lt.get("pair", "")
        live_dir    = lt.get("direction", "")
        live_entry  = lt["_entry_ts"]
        live_price  = lt.get("entry_price")

        bt_for_inst = bt_trades.get(instrument, [])
        best_match: dict | None = None
        best_diff   = timedelta.max
        exact_match = False

        for bt in bt_for_inst:
            bt_ts = bt.get("entry_time")
            if bt_ts is None:
                continue
            if hasattr(bt_ts, "tzinfo") and bt_ts.tzinfo is not None:
                bt_ts = bt_ts.tz_convert("UTC")
            else:
                bt_ts = pd.Timestamp(bt_ts, tz="UTC")
            diff  = abs(live_entry - bt_ts)
            if diff <= delta and diff < best_diff:
                best_diff   = diff
                best_match  = bt
                exact_match = (diff <= timedelta(hours=1))

        if best_match:
            bt_dir   = best_match.get("direction", "")
            bt_price = best_match.get("entry_price")
            bt_time  = best_match.get("entry_time")
            dir_ok   = (live_dir == bt_dir)

            if exact_match and dir_ok:
                match_status = "MATCH"
            elif exact_match and not dir_ok:
                match_status = "DIRECTION_MISMATCH"
            else:
                match_status = "APPROXIMATE"

            notes = ""
            if not dir_ok:
                notes = f"Live={live_dir}, BT={bt_dir}"
            elif abs((live_price or 0) - (bt_price or 0)) > 0.002 * (live_price or 1):
                notes = f"Price diff: live={live_price}, bt={bt_price}"
        else:
            bt_dir    = ""
            bt_price  = None
            bt_time   = None
            match_status = "NO_MATCH"
            notes = "No backtest signal within ±2h"

        comparison.append({
            "live_entry_time":       str(live_entry),
            "instrument":            instrument,
            "live_direction":        live_dir,
            "live_entry_price":      live_price,
            "backtest_signal_time":  str(bt_time) if bt_time else "",
            "backtest_direction":    bt_dir,
            "backtest_entry_price":  bt_price if bt_price is not None else "",
            "match_status":          match_status,
            "notes":                 notes,
        })

    # ── Save comparison CSV ───────────────────────────────────────────────────
    comp_path = RESULTS_2026_DIR / "live_vs_backtest_comparison.csv"
    _COMP_COLS = [
        "live_entry_time", "instrument", "live_direction", "live_entry_price",
        "backtest_signal_time", "backtest_direction", "backtest_entry_price",
        "match_status", "notes",
    ]
    with open(comp_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COMP_COLS)
        writer.writeheader()
        writer.writerows(comparison)

    log.info("  Comparison saved → %s", comp_path.name)
    log.info("[Step 5] Done.\n")
    return comparison


# =============================================================================
# STEP 6 — Print summary report
# =============================================================================

def step6_print_summary(
    all_metrics: dict,
    comparison: list[dict],
) -> None:
    """Print a clean summary report to stdout."""

    print("\n" + "=" * 70)
    print("  2026 OUT-OF-SAMPLE BACKTEST RESULTS")
    print(f"  Period: 2026-01-01 to {TODAY_STR}")
    print("=" * 70)

    # ── Per-instrument table ──────────────────────────────────────────────────
    header = (
        f"  {'Instrument':<13} {'Trades':>7} {'WR%':>6} {'Sharpe':>8}"
        f" {'P&L (£)':>10} {'MaxDD':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_trades     = 0
    total_pnl        = 0.0
    all_daily_pnl: list[float] = []

    for instrument in ALL_INSTRUMENTS:
        m = all_metrics.get(instrument)
        if m is None:
            print(f"  {instrument:<13} {'—':>7} {'—':>6} {'—':>8} {'—':>10} {'—':>7}")
            continue

        trades = m.get("total_trades", 0)
        wr     = m.get("win_rate", 0.0) * 100
        sharpe = m.get("sharpe_ratio", 0.0)
        pnl    = m.get("total_pnl_gbp", 0.0)
        dd     = m.get("max_drawdown", 0.0) * 100

        total_trades += trades
        total_pnl    += pnl

        print(
            f"  {instrument:<13} {trades:>7} {wr:>5.1f}% {sharpe:>8.2f}"
            f" £{pnl:>9.2f} {dd:>6.1f}%"
        )

    # ── Combined row ─────────────────────────────────────────────────────────
    # Combined Sharpe: compute from the per-instrument trade files
    combined_daily: dict = {}
    for instrument in ALL_INSTRUMENTS:
        path = RESULTS_2026_DIR / f"{instrument}_trades.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["exit_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
        df["date"] = df["exit_time"].dt.date
        for d, grp in df.groupby("date"):
            combined_daily[d] = combined_daily.get(d, 0.0) + float(grp["pnl_gbp"].sum())

    combined_sharpe = 0.0
    if len(combined_daily) > 1:
        eq = pd.Series(combined_daily).sort_index()
        std = eq.std()
        if std > 1e-9:
            combined_sharpe = float((eq.mean() / std) * math.sqrt(252))

    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'COMBINED':<13} {total_trades:>7} {'':>6} {combined_sharpe:>8.2f}"
        f" £{total_pnl:>9.2f}"
    )

    print()
    print(f"  Total trades:    {total_trades}")
    print(f"  Combined P&L:    £{total_pnl:.2f}")
    print(f"  Combined Sharpe: {combined_sharpe:.2f}")

    # ── Live trade comparison ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  LIVE TRADE COMPARISON  (last 14 days)")
    print("=" * 70)

    n_live      = len(comparison)
    n_match     = sum(1 for r in comparison if r["match_status"] == "MATCH")
    n_approx    = sum(1 for r in comparison if r["match_status"] == "APPROXIMATE")
    n_dir_mis   = sum(1 for r in comparison if r["match_status"] == "DIRECTION_MISMATCH")
    n_no_match  = sum(1 for r in comparison if r["match_status"] == "NO_MATCH")
    n_dir_match = sum(
        1 for r in comparison
        if r["match_status"] in ("MATCH", "APPROXIMATE")
        and r["live_direction"] == r["backtest_direction"]
    )
    matched_total = n_match + n_approx + n_dir_mis

    match_pct = matched_total / n_live * 100 if n_live else 0.0
    dir_pct   = n_dir_match / matched_total * 100 if matched_total else 0.0

    print(f"  Live trades in last 2 weeks: {n_live}")
    print(f"  Matched in backtest:         {matched_total} ({match_pct:.0f}%)")
    print(f"    - Exact match (<=1h):      {n_match}")
    print(f"    - Approximate (1-2h):      {n_approx}")
    print(f"    - Direction mismatch:      {n_dir_mis}")
    print(f"  Direction matches:           {n_dir_match} ({dir_pct:.0f}% of matched)")
    print(f"  Unmatched:                   {n_no_match}")

    if comparison:
        print()
        print("  Discrepancies:")
        disc = [r for r in comparison if r["notes"]]
        if disc:
            for r in disc[:10]:
                ts = r["live_entry_time"][:16]
                print(f"    {ts}  {r['instrument']:<13} {r['match_status']:<22} {r['notes']}")
        else:
            print("    None — all matched trades have consistent direction and price.")

    if n_live == 0:
        print()
        print("  (No live practice trades found — Supabase may be unavailable,")
        print("   or no trades were taken in the comparison window.)")

    print()
    print(f"  Full comparison CSV: backtest/results_2026/live_vs_backtest_comparison.csv")
    print("=" * 70)
    print()


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    log.info("Acieral Kairos — 2026 Out-of-Sample Backtest")
    log.info("Period: 2026-01-01 to %s", TODAY_STR)
    log.info("Cache:  %s", CACHE_2026_DIR)
    log.info("Output: %s", RESULTS_2026_DIR)
    log.info("")

    step1_fetch()
    step2_preprocess()
    step3_features()
    all_metrics  = step4_backtest()
    comparison   = step5_compare()
    step6_print_summary(all_metrics, comparison)


if __name__ == "__main__":
    main()
