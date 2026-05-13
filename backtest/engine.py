"""
backtest/engine.py — Entry-only backtest engine for Acieral Kairos Bot

Simulates live trading using trained entry models with per-instrument
real GBP P&L accounting.  Trades exit at fixed ATR-based SL or TP
(3:1 risk-reward placeholder) — no exit model yet.

Key design choices
------------------
- Features loaded from *_H1_features.parquet (NOT the label file).
- Model predictions computed upfront as a vectorised batch.
- Inner loop uses raw numpy arrays for speed (~14k holdout bars in <5s).
- SL checked before TP on every bar (conservative, slightly pessimistic P&L).
- All rate conversions use close of the bar on which the event occurs.
"""

import csv
import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import HARD_CONSTRAINTS
from ml.labeler import FEATURE_COLS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CACHE_DIR         = Path(__file__).parent.parent / "data" / "cache"
RESULTS_DIR       = Path(__file__).parent / "results"
RESULTS_TRAIN_DIR = Path(__file__).parent / "results_train"
MODELS_DIR        = Path(__file__).parent.parent / "ml" / "models"

for _d in (RESULTS_DIR, RESULTS_TRAIN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Hard constraints
FOREX_PAIRS         = HARD_CONSTRAINTS["FOREX_PAIRS"]
INDEX_INSTRUMENTS   = HARD_CONSTRAINTS["INDEX_INSTRUMENTS"]
ALL_INSTRUMENTS     = HARD_CONSTRAINTS["ALL_INSTRUMENTS"]
QUOTE_TYPE          = HARD_CONSTRAINTS["INSTRUMENT_QUOTE_TYPE"]
INDEX_SESSION_CLOSE = HARD_CONSTRAINTS["INDEX_SESSION_CLOSE_UTC"]
SL_ATR_MULT         = float(HARD_CONSTRAINTS["BACKTEST_SL_ATR_MULT"])   # 1.5
HOLDOUT_START       = HARD_CONSTRAINTS["HOLDOUT_START"]                  # "2024-01-01"
DAILY_DD_KILL_PCT   = float(HARD_CONSTRAINTS["DAILY_DRAWDOWN_KILL_PCT"]) # 0.03

# Last valid H1 open hour for index session close
# e.g. "21:00" → last open hour 20; "16:30" → last open hour 15
SESSION_LAST_OPEN_HOUR: dict[str, int] = {
    inst: int(cs.split(":")[0]) - 1
    for inst, cs in INDEX_SESSION_CLOSE.items()
}

TP_RR            = 3.0       # 3:1 RR placeholder until exit model
MIN_CONFIDENCE   = 0.5       # loose gate — multi-combo will tighten
STARTING_EQUITY  = 10_000.0
MAX_BARS_HELD    = 24        # hard close after 24 H1 bars (~1 trading day)
FALLBACK_GBPUSD  = 1.25      # used when rate data unavailable


# ---------------------------------------------------------------------------
# Position sizing helpers
# ---------------------------------------------------------------------------

def _compute_units(
    instrument: str,
    entry_price: float,
    sl_distance: float,
    equity: float,
    gbpusd_entry: float,
    gbpeur_entry: float | None = None,
) -> float:
    """Return position size in OANDA units."""
    risk_gbp = min(equity * 0.005, 50.0)
    qt = QUOTE_TYPE[instrument]

    if qt == "usd_quote":
        # risk_gbp treated as USD-equivalent for simplicity (CLAUDE.md spec)
        return risk_gbp / sl_distance

    elif qt == "usd_base":
        # sl_distance is in quote-currency terms; convert to USD fraction
        sl_usd = sl_distance / entry_price
        return (risk_gbp * gbpusd_entry) / sl_usd

    elif qt == "usd_index":
        return (risk_gbp * gbpusd_entry) / sl_distance

    elif qt == "eur_index":
        if gbpeur_entry is None or not math.isfinite(gbpeur_entry) or gbpeur_entry <= 0:
            gbpeur_entry = 1.17
        return (risk_gbp * gbpeur_entry) / sl_distance

    raise ValueError(f"Unknown quote type: {qt}")


def _compute_pnl_gbp(
    instrument: str,
    direction: int,
    entry_price: float,
    exit_price: float,
    units: float,
    gbpusd_exit: float,
    gbpeur_exit: float | None = None,
) -> float:
    """Compute realised GBP P&L for a closed trade."""
    dir_mult = 1 if direction == 2 else -1   # 2 = LONG, 1 = SHORT
    dp = dir_mult * (exit_price - entry_price)
    qt = QUOTE_TYPE[instrument]

    if qt == "usd_quote":
        return dp * units / gbpusd_exit

    elif qt == "usd_base":
        # dp is in quote currency; divide by exit to get USD equivalent
        return (dp / exit_price * units) / gbpusd_exit

    elif qt == "usd_index":
        return dp * units / gbpusd_exit

    elif qt == "eur_index":
        if gbpeur_exit is None or not math.isfinite(gbpeur_exit) or gbpeur_exit <= 0:
            gbpeur_exit = 1.17
        return dp * units / gbpeur_exit

    raise ValueError(f"Unknown quote type: {qt}")


# ---------------------------------------------------------------------------
# Rate loading
# ---------------------------------------------------------------------------

def _merge_rate(df: pd.DataFrame, pair: str, col: str,
                fallback: float = FALLBACK_GBPUSD) -> pd.DataFrame:
    """
    Merge the close price of *pair* onto *df* as column *col*
    using merge_asof (most-recently-closed rate bar).
    Falls back to a scalar if the parquet is missing.
    """
    path = CACHE_DIR / f"{pair}_H1.parquet"
    if not path.exists():
        log.warning("Rate file %s not found — using fallback %.4f", path.name, fallback)
        df[col] = fallback
        return df

    rate = pd.read_parquet(path, columns=["time", "close"])
    rate["time"] = pd.to_datetime(rate["time"], utc=True)
    rate = rate.sort_values("time").rename(columns={"close": col})
    return pd.merge_asof(df, rate, on="time", direction="backward")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    trades: list[dict],
    equity_by_date: dict,
) -> dict:
    """Derive summary statistics from completed trade list and daily equity."""
    if not trades:
        return {
            "total_trades":    0,
            "win_rate":        0.0,
            "profit_factor":   0.0,
            "sharpe_ratio":    0.0,
            "max_drawdown":    0.0,
            "total_pnl_gbp":   0.0,
            "avg_hold_bars":   0.0,
            "trades_per_week": 0.0,
            "skip_rate":       1.0,
        }

    pnls     = [t["pnl_gbp"] for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    n        = len(trades)
    win_rate = len(wins) / n

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses)) if losses else 0.0
    pf = (gross_profit / gross_loss) if gross_loss > 1e-9 else (999.99 if wins else 0.0)

    # Sharpe (annualised, daily P&L basis)
    sharpe = 0.0
    if equity_by_date and len(equity_by_date) > 1:
        eq_series  = pd.Series(equity_by_date).sort_index()
        daily_pnl  = eq_series.diff().dropna()
        std_pnl    = daily_pnl.std()
        if std_pnl > 1e-9:
            sharpe = float((daily_pnl.mean() / std_pnl) * math.sqrt(252))

    # Max drawdown from equity curve
    max_dd = 0.0
    if equity_by_date:
        eq_arr = np.array([v for _, v in sorted(equity_by_date.items())],
                          dtype=np.float64)
        peak   = np.maximum.accumulate(eq_arr)
        dd     = np.where(peak > 0, (peak - eq_arr) / peak, 0.0)
        max_dd = float(dd.max())

    avg_hold = float(np.mean([t["bars_held"] for t in trades]))
    total_pnl = sum(pnls)

    # Trades per week
    if equity_by_date and len(equity_by_date) > 1:
        all_dates = sorted(equity_by_date.keys())
        n_days    = (all_dates[-1] - all_dates[0]).days + 1
        n_weeks   = max(n_days / 7.0, 1.0)
        tpw       = n / n_weeks
    else:
        tpw = 0.0

    # Skip rate (fraction of trading days with no trade entry)
    if equity_by_date:
        traded_dates = {pd.Timestamp(t["entry_time"]).date() for t in trades}
        skip_rate    = max(0.0, 1.0 - len(traded_dates) / len(equity_by_date))
    else:
        skip_rate = 1.0

    return {
        "total_trades":    n,
        "win_rate":        round(win_rate, 4),
        "profit_factor":   round(min(pf, 999.99), 4),
        "sharpe_ratio":    round(sharpe, 4),
        "max_drawdown":    round(max_dd, 4),
        "total_pnl_gbp":   round(total_pnl, 2),
        "avg_hold_bars":   round(avg_hold, 2),
        "trades_per_week": round(tpw, 2),
        "skip_rate":       round(skip_rate, 4),
    }


# ---------------------------------------------------------------------------
# Core backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    instrument: str,
    model_path: str,
    N: int,
    T: float,
    split: str = "holdout",
) -> dict:
    """
    Run the entry-only backtest for *instrument*.

    Parameters
    ----------
    instrument : OANDA instrument name, e.g. 'EUR_USD'
    model_path : Path to trained entry model .pkl file
    N, T       : Labelling parameters (used for logging only — not to load labels)
    split      : 'holdout' uses data from HOLDOUT_START onward;
                 'train'   uses data before HOLDOUT_START

    Returns
    -------
    dict of summary metrics (same keys as _compute_metrics output)
    """
    # ------------------------------------------------------------------
    # 1. Load and filter features
    # ------------------------------------------------------------------
    feat_path = CACHE_DIR / f"{instrument}_H1_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {feat_path}\n"
            f"Run strategy.feature_builder.build_all() first."
        )

    df = pd.read_parquet(feat_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    holdout_ts = pd.Timestamp(HOLDOUT_START, tz="UTC")
    if split == "holdout":
        df = df[df["time"] >= holdout_ts].reset_index(drop=True)
    else:
        df = df[df["time"] <  holdout_ts].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(
            f"No data for {instrument} with split='{split}'.  "
            f"Holdout starts {HOLDOUT_START}."
        )

    log.info(
        "[%s] Backtest split='%s': %d bars (%s → %s)",
        instrument, split, len(df),
        df["time"].iloc[0].date(), df["time"].iloc[-1].date(),
    )

    # ------------------------------------------------------------------
    # 2. Load entry model
    # ------------------------------------------------------------------
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    # ------------------------------------------------------------------
    # 3. Merge rate data
    # ------------------------------------------------------------------
    df = _merge_rate(df, "GBP_USD", "_gbpusd", fallback=FALLBACK_GBPUSD)

    is_de30 = (instrument == "DE30_EUR")
    if is_de30:
        df = _merge_rate(df, "EUR_USD", "_eurusd", fallback=0.92)
        df["_gbpeur"] = (
            df["_gbpusd"] / df["_eurusd"].replace(0.0, np.nan)
        ).ffill().fillna(1.17)

    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Batch predict all bars
    # ------------------------------------------------------------------
    X = np.nan_to_num(
        df[FEATURE_COLS].to_numpy(dtype=np.float32), nan=0.0
    )
    probas       = model.predict_proba(X)
    pred_classes = np.argmax(probas, axis=1).astype(np.int8)
    confidences  = np.max(probas, axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Extract price / rate arrays for fast loop access
    # ------------------------------------------------------------------
    n        = len(df)
    closes   = df["close"].to_numpy(dtype=np.float64)
    highs    = df["high"].to_numpy(dtype=np.float64)
    lows     = df["low"].to_numpy(dtype=np.float64)
    atrs     = df["atr_14"].to_numpy(dtype=np.float64)
    gbpusd   = df["_gbpusd"].to_numpy(dtype=np.float64)
    gbpeur   = df["_gbpeur"].to_numpy(dtype=np.float64) if is_de30 else None

    dates_arr = df["time"].dt.date.to_numpy()
    hours_arr = df["time"].dt.hour.to_numpy(dtype=np.int32)
    dow_arr   = df["time"].dt.dayofweek.to_numpy(dtype=np.int32)  # 0=Mon, 4=Fri
    ts_vals   = df["time"].to_numpy()  # numpy datetime64 / pandas Timestamp

    is_index       = instrument in INDEX_INSTRUMENTS
    last_sess_hour = SESSION_LAST_OPEN_HOUR.get(instrument)  # None for forex

    # ------------------------------------------------------------------
    # 6. Simulation state
    # ------------------------------------------------------------------
    equity           = STARTING_EQUITY
    day_start_equity = STARTING_EQUITY
    current_date     = None
    traded_today     = False
    kill_switch      = False
    open_trade: dict | None = None

    trades:         list[dict] = []
    equity_by_date: dict       = {}

    # ------------------------------------------------------------------
    # 7. Main simulation loop
    # ------------------------------------------------------------------
    for i in range(n):
        bar_date  = dates_arr[i]
        bar_hour  = int(hours_arr[i])
        bar_dow   = int(dow_arr[i])
        bar_high  = highs[i]
        bar_low   = lows[i]
        bar_close = closes[i]

        # ---- Day boundary ----------------------------------------
        if bar_date != current_date:
            if current_date is not None:
                equity_by_date[current_date] = equity
            current_date     = bar_date
            day_start_equity = equity
            traded_today     = False
            kill_switch      = False

        # ---- Exit check for open trade ---------------------------
        if open_trade is not None:
            bars_held = i - open_trade["entry_bar"]
            direction = open_trade["direction"]   # 1=SHORT, 2=LONG
            sl_price  = open_trade["sl_price"]
            tp_price  = open_trade["tp_price"]
            ep        = open_trade["entry_price"]

            # Update MFE / MAE
            if direction == 2:  # LONG
                open_trade["mfe_price"] = max(open_trade["mfe_price"], bar_high)
                open_trade["mae_price"] = min(open_trade["mae_price"], bar_low)
            else:               # SHORT
                open_trade["mfe_price"] = min(open_trade["mfe_price"], bar_low)
                open_trade["mae_price"] = max(open_trade["mae_price"], bar_high)

            exit_price: float | None = None
            exit_reason: str  | None = None

            # SL — checked before TP (conservative)
            if direction == 2 and bar_low  <= sl_price:
                exit_price, exit_reason = sl_price, "SL"
            elif direction == 1 and bar_high >= sl_price:
                exit_price, exit_reason = sl_price, "SL"

            # TP
            if exit_reason is None:
                if direction == 2 and bar_high >= tp_price:
                    exit_price, exit_reason = tp_price, "TP"
                elif direction == 1 and bar_low  <= tp_price:
                    exit_price, exit_reason = tp_price, "TP"

            # EOD / session close
            if exit_reason is None:
                if is_index and last_sess_hour is not None and bar_hour >= last_sess_hour:
                    exit_price, exit_reason = bar_close, "EOD"
                elif not is_index and bar_dow == 4 and bar_hour >= 20:
                    exit_price, exit_reason = bar_close, "FRIDAY_CLOSE"

            # Hard max-bars cap
            if exit_reason is None and bars_held >= MAX_BARS_HELD:
                exit_price, exit_reason = bar_close, "MAX_BARS"

            # ---- Close trade -------------------------------------
            if exit_price is not None:
                gbu = float(gbpusd[i])
                gbe = float(gbpeur[i]) if is_de30 else None
                if not math.isfinite(gbu) or gbu <= 0:
                    gbu = FALLBACK_GBPUSD

                pnl = _compute_pnl_gbp(
                    instrument, direction, ep, exit_price,
                    open_trade["units"], gbu, gbpeur_exit=gbe,
                )
                equity += pnl

                trades.append({
                    "entry_time":   open_trade["entry_time"],
                    "exit_time":    str(pd.Timestamp(ts_vals[i])),
                    "direction":    "LONG" if direction == 2 else "SHORT",
                    "entry_price":  round(ep, 5),
                    "exit_price":   round(exit_price, 5),
                    "sl_price":     round(sl_price, 5),
                    "tp_price":     round(tp_price, 5),
                    "units":        round(open_trade["units"], 2),
                    "pnl_gbp":      round(pnl, 4),
                    "exit_reason":  exit_reason,
                    "bars_held":    bars_held,
                    "mfe_price":    round(open_trade["mfe_price"], 5),
                    "mae_price":    round(open_trade["mae_price"], 5),
                    "confidence":   round(float(open_trade["confidence"]), 4),
                    "atr_at_entry": round(open_trade["atr_at_entry"], 6),
                })
                open_trade = None

                # Kill switch
                if (day_start_equity - equity) / max(day_start_equity, 1.0) \
                        >= DAILY_DD_KILL_PCT:
                    kill_switch = True

        # ---- Entry check -----------------------------------------
        if open_trade is None and not traded_today and not kill_switch:
            pc   = int(pred_classes[i])
            conf = float(confidences[i])

            if pc in {1, 2} and conf >= MIN_CONFIDENCE:
                atr = float(atrs[i])
                if not math.isfinite(atr) or atr <= 0:
                    continue

                ep_price = closes[i]
                sl_dist  = SL_ATR_MULT * atr

                if pc == 2:  # LONG
                    sl_price = ep_price - sl_dist
                    tp_price = ep_price + TP_RR * sl_dist
                else:        # SHORT
                    sl_price = ep_price + sl_dist
                    tp_price = ep_price - TP_RR * sl_dist

                gbu = float(gbpusd[i])
                gbe = float(gbpeur[i]) if is_de30 else None
                if not math.isfinite(gbu) or gbu <= 0:
                    gbu = FALLBACK_GBPUSD

                units = _compute_units(
                    instrument, ep_price, sl_dist, equity,
                    gbu, gbpeur_entry=gbe,
                )

                open_trade = {
                    "entry_bar":    i,
                    "entry_time":   str(pd.Timestamp(ts_vals[i])),
                    "direction":    pc,
                    "entry_price":  ep_price,
                    "sl_price":     sl_price,
                    "tp_price":     tp_price,
                    "sl_distance":  sl_dist,
                    "units":        units,
                    "confidence":   conf,
                    "atr_at_entry": atr,
                    "mfe_price":    ep_price,
                    "mae_price":    ep_price,
                }
                traded_today = True

    # Record equity for the final (incomplete) day
    if current_date is not None:
        equity_by_date[current_date] = equity

    # Force-close any trade still open at data end
    if open_trade is not None:
        ep = open_trade["entry_price"]
        exit_price = closes[-1]
        gbu = float(gbpusd[-1])
        gbe = float(gbpeur[-1]) if is_de30 else None
        if not math.isfinite(gbu) or gbu <= 0:
            gbu = FALLBACK_GBPUSD

        pnl = _compute_pnl_gbp(
            instrument, open_trade["direction"], ep, exit_price,
            open_trade["units"], gbu, gbpeur_exit=gbe,
        )
        equity += pnl

        bars_held = n - 1 - open_trade["entry_bar"]
        trades.append({
            "entry_time":   open_trade["entry_time"],
            "exit_time":    str(pd.Timestamp(ts_vals[-1])),
            "direction":    "LONG" if open_trade["direction"] == 2 else "SHORT",
            "entry_price":  round(ep, 5),
            "exit_price":   round(exit_price, 5),
            "sl_price":     round(open_trade["sl_price"], 5),
            "tp_price":     round(open_trade["tp_price"], 5),
            "units":        round(open_trade["units"], 2),
            "pnl_gbp":      round(pnl, 4),
            "exit_reason":  "DATA_END",
            "bars_held":    bars_held,
            "mfe_price":    round(open_trade["mfe_price"], 5),
            "mae_price":    round(open_trade["mae_price"], 5),
            "confidence":   round(float(open_trade["confidence"]), 4),
            "atr_at_entry": round(open_trade["atr_at_entry"], 6),
        })

    # ------------------------------------------------------------------
    # 8. Compute and save metrics
    # ------------------------------------------------------------------
    metrics = _compute_metrics(trades, equity_by_date)
    metrics["instrument"] = instrument
    metrics["split"]      = split
    metrics["n_bars"]     = n

    out_dir = RESULTS_DIR if split == "holdout" else RESULTS_TRAIN_DIR

    # Trades CSV
    _TRADE_COLS = [
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "sl_price", "tp_price", "units", "pnl_gbp", "exit_reason",
        "bars_held", "mfe_price", "mae_price", "confidence", "atr_at_entry",
    ]
    trades_path = out_dir / f"{instrument}_trades.csv"
    with open(trades_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_TRADE_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)

    # Equity curve CSV
    equity_path = out_dir / f"{instrument}_equity.csv"
    with open(equity_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "equity"])
        for d, eq in sorted(equity_by_date.items()):
            writer.writerow([str(d), round(eq, 4)])

    # Metrics JSON (exclude non-serialisable entries)
    metrics_for_json = {k: v for k, v in metrics.items()}
    metrics_path = out_dir / f"{instrument}_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics_for_json, fh, indent=2)

    log.info(
        "[%s] %d trades | win=%.1f%% | PF=%.2f | Sharpe=%.2f | DD=%.1f%% | P&L=£%.2f",
        instrument,
        metrics["total_trades"],
        metrics["win_rate"] * 100,
        metrics["profit_factor"],
        metrics["sharpe_ratio"],
        metrics["max_drawdown"] * 100,
        metrics["total_pnl_gbp"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Diagnostic: signal rate at various confidence thresholds
# ---------------------------------------------------------------------------

def check_signal_rate(
    instrument: str,
    model_path: str,
    N: int,
    T: float,
) -> None:
    """
    Print how often the entry model fires a directional signal on holdout data
    at several confidence thresholds.

    Useful for calibrating MIN_CONFIDENCE before the multi-combo selection step.
    Compare the entry rate against the labeller's ~2.3% trade rate to understand
    how selective the model is at each threshold.
    """
    from ml.labeler import FEATURE_COLS as _FC

    feat_path = CACHE_DIR / f"{instrument}_H1_features.parquet"
    df = pd.read_parquet(feat_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    holdout_ts = pd.Timestamp(HOLDOUT_START, tz="UTC")
    df = df[df["time"] >= holdout_ts].reset_index(drop=True)

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    X       = np.nan_to_num(df[_FC].to_numpy(dtype=np.float32), nan=0.0)
    probas  = model.predict_proba(X)
    classes = np.argmax(probas, axis=1)
    confs   = np.max(probas, axis=1)

    n_total     = len(df)
    directional = classes != 0                     # predicted SHORT or LONG

    # labeller trade rate reference: ~2.3% long + 2.2% short = ~4.5% directional
    # but only one label per day, so per-bar rate is lower
    label_ref_pct = (2.26 + 2.16)   # from EUR_USD N=10 T=20bps grid

    thresholds = [0.5, 0.6, 0.7, 0.8]

    print(f"\n{'='*60}")
    print(f"Signal rate diagnostic - {instrument}  (N={N}, T={T})")
    print(f"Holdout bars: {n_total:,}  "
          f"({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
    print(f"Labeller directional trade rate reference: ~{label_ref_pct:.2f}%")
    print(f"{'-'*60}")
    print(f"  {'Condition':<35} {'Count':>7}  {'%':>7}")
    print(f"  {'-'*35} {'-'*7}  {'-'*7}")

    n_dir = int(directional.sum())
    print(f"  {'class != 0  (any signal)':<35} {n_dir:>7,}  {n_dir/n_total*100:>6.2f}%")

    for thr in thresholds:
        mask  = directional & (confs >= thr)
        count = int(mask.sum())
        print(f"  {'conf >= ' + str(thr) + ' AND class != 0':<35} {count:>7,}  {count/n_total*100:>6.2f}%")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_backtests(split: str = "holdout") -> None:
    """
    Run backtest for all 8 instruments using models in ml/models/.
    Reads N/T per instrument from ml/models/metadata.json.
    Saves per-instrument results and a summary CSV.
    """
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "ml/models/metadata.json not found — run ml/trainer.py first."
        )
    with open(meta_path) as fh:
        metadata = json.load(fh)

    summary_rows = []
    for instrument in ALL_INSTRUMENTS:
        if instrument not in metadata:
            log.warning("[%s] No metadata — skipping.", instrument)
            continue

        meta       = metadata[instrument]
        N, T       = int(meta["N"]), float(meta["T"])
        model_path = MODELS_DIR / f"entry_{instrument}.pkl"

        if not model_path.exists():
            log.warning("[%s] Model file missing — skipping.", instrument)
            continue

        try:
            m = run_backtest(instrument, str(model_path), N, T, split)
            summary_rows.append({
                "instrument":      instrument,
                "total_trades":    m["total_trades"],
                "win_rate":        m["win_rate"],
                "profit_factor":   m["profit_factor"],
                "sharpe_ratio":    m["sharpe_ratio"],
                "max_drawdown":    m["max_drawdown"],
                "total_pnl_gbp":   m["total_pnl_gbp"],
                "avg_hold_bars":   m["avg_hold_bars"],
                "trades_per_week": m["trades_per_week"],
            })
        except Exception as exc:
            log.error("[%s] FAILED: %s", instrument, exc, exc_info=True)

    if not summary_rows:
        log.warning("No results to summarise.")
        return

    out_dir = RESULTS_DIR if split == "holdout" else RESULTS_TRAIN_DIR
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    log.info("Summary → %s", out_dir / "summary.csv")

    # Print table
    header = (f"{'Instrument':<15} {'Trades':>7} {'WinRate':>8} {'PF':>7}"
              f" {'Sharpe':>8} {'MaxDD':>7} {'P&L GBP':>10}")
    print("\n" + header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['instrument']:<15} {row['total_trades']:>7}"
            f" {row['win_rate']:>7.1%} {row['profit_factor']:>7.2f}"
            f" {row['sharpe_ratio']:>8.2f} {row['max_drawdown']:>6.1%}"
            f" £{row['total_pnl_gbp']:>9.2f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_backtests(split="holdout")
