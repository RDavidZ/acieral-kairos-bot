# CLAUDE-REFERENCE.md — Acieral Kairos Bot

Deep reference for strategy, feature engineering, ML pipeline, architecture rules,
and build status. Read when working on `ml/`, `strategy/`, `backtest/`, or `execution/`.

For operational state, deploy workflow, and known issues → see **CLAUDE.md**.

---

## Strategy Logic

### Core Concept

Fair Value Gaps (FVGs) are price imbalances where no two-sided trading occurred.
They act as structural reference points that price tends to revisit. The ML entry
model learns — from data — which combinations of FVG context, higher-timeframe
structure, session timing, volatility regime, and momentum make a 1H FVG setup
likely to produce a profitable trade.

Strategy note: model is primarily driven by session timing, HTF distance features,
and daily structure — not FVG structure directly. FVG features present but rank
outside top 20 globally. Name reflects the temporal edge (Kairos = the opportune
moment) rather than the entry signal type.

### What the bot does each bar (1H candle close)

1. **Feature cache update (HH:00):** Fetch latest H1 bars → preprocess → update parquet.

2. **Feature computation (HH:01):** FVG detection, swing structure, higher-TF context
   (4H and Daily candles forward-filled onto 1H rows), session flags, momentum.

3. **Entry evaluation:** Entry model scores every 1H candle close.
   - At most one trade per instrument per day.
   - Confidence must exceed CONFIDENCE_CURVE threshold for that instrument.
   - If no candle reaches threshold: skip the day (no trade).

4. **Exit evaluation:** Once a trade is open, exit model evaluates every 1H close.
   - Trailing SL is the hard floor (activates at N × ATR profit, per-instrument).
   - Exit model fires when P(exit) ≥ threshold AND price is above trail floor.
   - Hard EOD close for indices (session end). Hard Friday 21:00 UTC close for forex.

5. **Risk manager** enforces all hard constraints regardless of model output.

### Two-Model Architecture

**Model 1 — Entry Model (XGBoost multiclass, per instrument)**
- Output: 0 = No trade, 1 = Short, 2 = Long
- Trained per instrument — no cross-instrument weight sharing
- Class imbalance: SMOTE + compute_sample_weight balanced (XGBoost 3.x multiclass fix)

**Model 2 — Exit Model (XGBoost binary, per instrument)**
- Output: 0 = Stay, 1 = Exit
- Trained on oracle peak-exit labels from entry-only backtest trade log
- Trailing SL floor beneath exit model — exit model cannot fire below floor

---

## Feature Engineering (~80 features, instrument-agnostic computation)

**Rules (never violate):**
- No lookahead — all features use only data available at candle close time t
- No raw pip/point values — all price distances are ATR-normalised
- No hardcoded time gates — time-of-day is a numeric feature, not a filter
- Higher-TF features use the most recently *closed* higher-TF candle only

### Feature Groups

**1. FVG Features (primary signal)**
fvg_bull_exists, fvg_bear_exists, fvg_dist_atr, fvg_size_atr, fvg_age_bars,
fvg_fill_pct, fvg_count_direction, fvg_stacked

**2. Swing Structure**
swing_high_dist_atr, swing_low_dist_atr, swing_swept, sweep_magnitude_atr,
bars_since_sweep, structure_bias, swing_test_count
Pivot lookback N ∈ {5, 10, 15, 20} — grid searched per instrument (default 10)

**3. Session and Time**
minutes_since_midnight_utc, hour_utc, day_of_week,
session_asian, session_london, session_ny, session_overlap,
hours_since_session_open, is_london_first_3h, is_ny_first_3h
(Indices: anchored to own session open — SPX/NAS to NY open, DE30 to EU open)

**4. Higher-Timeframe Context (top-down, encoded as features)**
Daily: d1_ema20_slope, d1_close_vs_ema20, d1_high_dist_atr, d1_low_dist_atr,
       d1_fvg_bull_exists, d1_fvg_bear_exists, d1_fvg_dist_atr, d1_swing_bias
4H:   h4_ema20_slope, h4_close_vs_ema20, h4_fvg_bull_exists, h4_fvg_bear_exists,
       h4_fvg_dist_atr, h4_swing_bias
Both: htf_alignment (Daily and 4H bias agree — binary confluence flag)
Weekly: w1_high_dist_atr, w1_low_dist_atr

**5. Volatility and Regime**
atr_14, atr_ratio, atr_ratio_delta, candle_body_ratio,
wick_upper_ratio, wick_lower_ratio, range_vs_atr_5, adx_14, adx_delta

**6. Momentum (rate-of-change preferred over absolute level)**
rsi_14, rsi_14_delta_3, rsi_divergence_bull, rsi_divergence_bear,
macd_hist, macd_hist_delta, roc_4, roc_10

**7. Candle Context**
close_lag_1..4, body_ratio_lag_1..3, wick_upper_lag_1, wick_lower_lag_1,
day_high_dist_atr, day_low_dist_atr, day_displacement_atr,
bull_candle_count, bear_candle_count

**Indices only (supplementary):**
vix_close, vix_delta_1, spy_volume_ratio (yfinance, merged by trading date)

> 81 features in FEATURE_COLS (75 specified + 6 additional from feature_builder).
> Label distribution: ~95.6% NO_TRADE, ~2.2% SHORT, ~2.3% LONG on EUR_USD.
> T encoded as basis points in filenames.

---

## Labelling

### Entry Labels
- Triple-barrier approach: best-scoring candle per day wins the label
- Grid search: N ∈ {4,6,8,10,12,16,20}, T ∈ {0.05%,0.10%,0.15%,0.20%}
- Top 3 N/T combos per instrument → retrain → holdout Sharpe → winner selected
- For indices: lookahead capped at end of same trading session (no overnight labels)

### Exit Labels
- Oracle peak-profit candle per trade → EXIT=1, all others STAY=0
- 7 trade-context features added: tc_bars_held, tc_unrealised_pnl_atr,
  tc_mfe_atr, tc_mae_atr, tc_pct_mfe_given_back, tc_direction, tc_dist_to_sl_atr

---

## Walk-Forward Training

- Rolling 18-month training window, 3-month validation
- ~20-22 folds per instrument across 2017–2023
- Hyperparameter grid per fold: n_estimators ∈ {200,400}, max_depth ∈ {3,4,5}, lr ∈ {0.05,0.1}
- Holdout: 2024-01-01 onwards — sealed, opened once for final evaluation only
- Live retrain: every 20 trading days at 00:05 UTC

---

## Architecture Rules (Never Violate)

1. `config.py` has two sections: `HARD_CONSTRAINTS` (human-defined) and `DISCOVERED_PARAMS` (generated by config_generator.py — never hand-edit)
2. Dependency order: `data/ → strategy/ → ml/ → backtest/ → execution/`
3. `backtest/` never imports from `execution/`
4. No time gates in feature engineering — time is a numeric feature input
5. No lookahead — features at candle t use only data available at close of t
6. Higher-TF features use only the most recently *closed* candle — no partial candles
7. Models never retrained during live trading — only at midnight between sessions
8. All time handling uses `zoneinfo` — `Europe/London` for forex, instrument-native session for indices
9. All candle slicing uses close time
10. All price distances are ATR-normalised — no raw pip or point values in features
11. OANDA instrument names use underscores: `EUR_USD`, `SPX500_USD`
12. One trade per instrument per day — enforced in both backtest engine and risk manager
13. Per-instrument models — never share weights across instruments
14. Index positions must close at session end — hard constraint, not model decision
15. Feature builder is instrument-aware: supplementary VIX/volume features computed only for indices; session anchoring differs per instrument class
16. Two threads share mutable trade state: the APScheduler thread (HH:01 candle close) and the PriceStream daemon thread (tick-level SL/EOD/news checks). All mutations to `_open_trades` must hold `RiskManager._lock` (a `threading.RLock`). OANDA API calls and Telegram sends must happen **outside** the lock to avoid blocking the stream parser.
17. GBP/USD and GBP/EUR rates are cached on the scheduler thread (`refresh_gbpusd_rate()` called in `on_candle_close`). The streaming thread uses cached rates — it must never make live HTTP calls during `manage_on_tick`.
18. Dashboard writes are always best-effort: wrap in `try/except`, log warnings, never block a trade operation. `dashboard.db` may be unavailable (ImportError) on local dev — all imports inside functions.

---

## File Structure

```
acieral-kairos-bot/
├── CLAUDE.md                            ← operational reference
├── CLAUDE-REFERENCE.md                  ← this file (strategy/architecture/build)
├── DASHBOARD.md                         ← full dashboard system reference
├── config.py                            ← HARD_CONSTRAINTS + DISCOVERED_PARAMS
├── retrain_state.json                   ← persists trading_days_since_retrain across restarts
├── requirements.txt
├── run_pipeline.sh                      ← local ML pipeline runner
├── analysis/                            ← ad-hoc analysis scripts (not deployed)
│   ├── de30_day_analysis.py
│   ├── de30_day_analysis_holdout.py
│   ├── live_vs_backtest_check.py
│   ├── scenario_analysis_2026.py
│   ├── scenario_day_of_week.py
│   └── validate_feature_cache.py
├── data/
│   ├── fetcher.py                       ← fetch 1H, 4H, D1, W1 via oandapyV20
│   ├── fetcher_supplementary.py         ← VIX, SPY volume via yfinance (indices)
│   ├── preprocessor.py                  ← ATR, indicators, forward-fill HTF onto 1H
│   ├── news_fetcher.py                  ← ForexFactory calendar, 12h cache
│   └── cache/
│       ├── news_cache.json              ← cached news events (auto-refreshed)
│       └── {instrument}_h1.parquet      ← feature cache (updated HH:00 by scheduler)
├── strategy/
│   ├── fvg_detector.py                  ← FVG detection algorithm (deterministic)
│   ├── swing_detector.py                ← pivot-point swing high/low detection
│   └── feature_builder.py              ← all ~80 features, per-instrument aware
├── ml/
│   ├── labeler.py                       ← triple-barrier entry labels, N/T grid search
│   ├── trainer.py                       ← walk-forward XGBoost entry model
│   ├── multi_combo_trainer.py           ← top-3 N/T → holdout winner per instrument
│   ├── exit_labeler.py                  ← oracle peak-exit labels from backtest log
│   ├── exit_trainer.py                  ← walk-forward XGBoost exit model
│   ├── config_generator.py              ← derives DISCOVERED_PARAMS → config.py
│   ├── models/
│   │   ├── entry_{instrument}.pkl       ← per-instrument entry models
│   │   ├── exit_{instrument}.pkl        ← per-instrument exit models
│   │   ├── metadata.json
│   │   └── exit_metadata.json
│   ├── labels/                          ← holdout entry labels per instrument
│   ├── labels_train/                    ← training entry labels
│   ├── exit_labels/                     ← exit labels per instrument
│   └── multi_combo_results/             ← N/T combo results per instrument
├── backtest/
│   ├── engine.py                        ← entry-only backtest, real GBP P&L
│   ├── engine_with_exit.py              ← trail SL + exit model, per-instrument sweep
│   ├── run_2026_test.py                 ← 2026 OOS test runner
│   ├── run_holdout_curve.py             ← holdout equity curve builder
│   ├── results/                         ← per-instrument: trades, metrics, equity CSVs
│   ├── results_train/
│   └── results_exit/
├── risk/
│   ├── manager.py                       ← stateful risk enforcement, RLock, tiered sizing
│   └── news_filter.py                   ← entry block / pre-close / size reduction logic
├── execution/
│   ├── oanda_client.py                  ← oandapyV20 wrapper, retry logic, JPY precision
│   ├── order_manager.py                 ← full trade lifecycle, DB writes, tick management
│   └── price_stream.py                  ← OANDA PriceStream daemon (real-time tick feed)
├── notifications/
│   └── telegram_bot.py                  ← all alert formats
├── deploy/
│   ├── acieral-kairos.service           ← systemd unit (Restart=on-failure)
│   ├── DEPLOY.md                        ← full deployment guide
│   ├── package.ps1                      ← local backup + bundle script
│   ├── setup_vps.sh                     ← VPS install + data fetch
│   └── seed_dashboard.py               ← seeds backtest trades + equity curve to Supabase
├── logs/
│   └── acieral_kairos.log              ← rotating log (10MB × 5 files)
└── bot/
    └── main.py                          ← APScheduler: 7 jobs, PriceStream daemon, retrain
```

---

## Build Status

### Phase 1 — Data and Feature Pipeline
| Step | Module | Status | Notes |
|---|---|---|---|
| 1 | `config.py` | ✅ | Hard constraints only |
| 2 | `data/fetcher.py` | ✅ | 1H, 4H, D1, W1 — forex + indices |
| 3 | `data/fetcher_supplementary.py` | ✅ | VIX, SPY volume via yfinance |
| 4 | `data/preprocessor.py` | ✅ | ATR, indicators, HTF forward-fill |
| 5 | `strategy/fvg_detector.py` | ✅ | Deterministic FVG detection |
| 6 | `strategy/swing_detector.py` | ✅ | Pivot-point swing high/low |
| 7 | `strategy/feature_builder.py` | ✅ | All ~80 features |

> pandas-ta unavailable on PyPI — replaced with ta==0.11.0 throughout.
> All feature builder code uses ta.trend, ta.momentum etc.

### Phase 2 — ML Pipeline
| Step | Module | Status | Notes |
|---|---|---|---|
| 8 | `ml/labeler.py` | ✅ | Triple-barrier, N/T grid |
| 9 | `ml/trainer.py` | ✅ | Walk-forward XGBoost entry. EUR_USD baseline: mean val AUC 0.9531 ± 0.0099, 22 folds. |
| 10 | `backtest/engine.py` | ✅ | Entry-only, real GBP P&L |
| 11 | `ml/multi_combo_trainer.py` | ✅ | Full 8-instrument run. All converged on T=0.0005 (5bps). Forex N=6–8, Indices N=4. Winners: EUR_USD N=6 Sharpe=11.69, GBP_USD N=8 Sharpe=11.97, AUD_USD N=8 Sharpe=11.22, USD_JPY N=8 Sharpe=10.60, USD_CHF N=8 Sharpe=12.02, SPX500 N=4 Sharpe=13.78, NAS100 N=4 Sharpe=11.02, DE30 N=4 Sharpe=15.68. WF AUC range 0.91–0.94. Note: NAS100 and DE30 top-3 combos produced identical results — label degeneracy at N=4 (index volatility means T5bps trades always exceed T10/T15bps within 4 bars). Not a bug — N=4 T5bps winner is valid. |
| 12 | `ml/exit_labeler.py` | ✅ | Full 8-instrument run. Forex: 413–473 trades, 5,857–7,883 bars, EXIT%=5.9–7.1%. Indices: 440–485 trades, 3,321–4,609 bars, EXIT%=10.5–13.9%. Oracle sanity confirmed: EXIT bars MFE-given-back=0.000. |
| 13 | `ml/exit_trainer.py` | ✅ | Full 8-instrument run. AUC range 0.966–0.984: EUR_USD 0.9753, GBP_USD 0.9776, AUD_USD 0.9774, USD_JPY 0.9773, USD_CHF 0.9772, SPX500 0.9660, NAS100 0.9714, DE30 0.9835. AUC inflated by tc_ features — true OOS eval in engine_with_exit. |
| 14 | `backtest/engine_with_exit.py` | ✅ | Full 8-instrument 42-combo sweep. Winners: EUR_USD trail=3.0 thr=0.35 Sharpe=11.74, GBP_USD trail=2.5 thr=0.30 Sharpe=12.60, AUD_USD trail=2.0 thr=0.30 Sharpe=13.05, USD_JPY trail=2.0 thr=0.30 Sharpe=12.46, USD_CHF trail=3.0 thr=0.30 Sharpe=13.11, SPX500 trail=3.0 thr=0.30 Sharpe=12.87, NAS100 trail=3.0 thr=0.60 Sharpe=9.92, DE30 trail=2.5 thr=0.30 Sharpe=15.91. |
| 15 | `ml/config_generator.py` | ✅ | Full 8-instrument run. All 8 ACTIVE (Sharpe≥1.5, DD≤15%, trades≥50). Top global features: d1_low_dist_atr, d1_high_dist_atr, minutes_since_midnight, hour_utc, close_lag_1, day_high_dist_atr, bear_candle_count, session_ny, day_low_dist_atr, session_london. KEEP/DROP boundary at global rank 40. |

### Phase 3 — Execution Layer
| Step | Module | Status | Notes |
|---|---|---|---|
| 16 | `risk/manager.py` | ✅ | RiskManager: stateful, 10 can_trade checks in order, separate forex/index counters, daily reset preserves equity and open trades, kill switch clears on daily reset. `threading.RLock` protects trade dict. `on_kill_switch` callback fires once on False→True transition. Tiered risk via `RISK_TIERS`. `get_all_open_trades()` returns snapshot safe for iteration outside lock. |
| 17 | `execution/oanda_client.py` | ✅ | OandaClient: retry wrapper (3×, exponential backoff). JPY 3dp precision on market_order and update_stop_loss. is_market_open handles forex weekend and index session hours. |
| 18 | `execution/order_manager.py` | ✅ | Full trade lifecycle. `attempt_entry` → entry model → risk gate → OANDA order → DB write → Telegram. `manage_open_trade` (candle close): MFE/MAE update → trail SL → SL hit → EOD/Friday/max-bars hard closes → exit model. `manage_on_tick` (streaming thread): SL hit detection + trail SL updates (throttled 1/min OANDA call; in-memory updated every tick). Real-time index session EOD and pre-news close on every tick. `force_close` for news pre-closes and EOD forced exits. Units: float rounded to 2dp. Equity snapshot written after every trade close (all 5 paths). |
| 19 | `notifications/telegram_bot.py` | ✅ | HTML parse mode, retry once on failure, no-op if not configured. 6 alert types: trade opened, trade closed, skipped, trail activated, unrealised milestone (£50/£100), daily summary. |
| 20 | `execution/price_stream.py` | ✅ | OANDA PriceStream as daemon thread. Calls `_on_price_tick(instrument, bid, ask, now)` on every tick. Errors logged, never crash the stream. |
| 21 | `bot/main.py` | ✅ | **7 scheduler jobs:** (1) HH:00 — feature cache update; (2) HH:01 — candle close: GBP rate refresh → all instruments → news/entry/exit logic; (3) 00:05 — daily reset; (4) 21:05 — daily summary Telegram; (5) every 30 min — equity snapshot; (6) every 30 min — heartbeat; (7) every 12h — news DB sync. Weekend handling: force-closes index positions, logs forex (protected by OANDA hard SL). Retrain counter persisted to `retrain_state.json`. |

### Phase 4 — Deployment
| Step | Task | Status |
|---|---|---|
| 22 | VPS setup, venv, systemd service | ✅ | `deploy/package.ps1` (local bundle), `deploy/setup_vps.sh` (VPS install), `deploy/acieral-kairos.service` (systemd). Full guide: `deploy/DEPLOY.md` |
| 23 | Supabase bot registration | ✅ | `register_bot()` called in `AcieralKairosBot.__init__` |
| 24 | Seed backtest history to dashboard | ✅ | `deploy/seed_dashboard.py` seeds all 8 instruments' backtest trades + equity curve from `backtest/results_exit/` using `bulk_write_trades()` (500-row batches — never use individual `write_trade()` for seeds) |

> **Phase 1 ✅ Phase 2 ✅ Phase 3 ✅ Phase 4 ✅ — Live, running on VPS**

---

## Live Operations

### Two-Thread Model

**Scheduler thread (APScheduler):**
- Runs all 7 jobs
- Owns the candle-level trade logic (entry, exit model, trail SL, hard closes)
- Refreshes GBP/USD + GBP/EUR rate cache before each candle
- Holds `RiskManager._lock` only for brief trade dict mutations

**Streaming thread (PriceStream daemon):**
- Receives OANDA tick feed continuously
- Calls `manage_on_tick(instrument, bid, ask, now)` per tick
- Handles: tick-level SL hit, real-time trail SL (throttled 1/min OANDA call), real-time index EOD close, real-time pre-news close
- Uses cached GBP rates — never makes live HTTP calls
- Trail SL mutations protected by `RiskManager._lock`; OANDA calls and Telegram sends happen outside the lock

### Trade Close Paths (all 5 write equity snapshots)

1. `manage_open_trade` — normal candle-close path (SL hit, EOD, max bars, exit model)
2. `manage_open_trade` — OANDA 404 recovery (trade already closed externally)
3. `manage_on_tick` — tick-level SL hit
4. `force_close` — news pre-close, weekend EOD cleanup
5. `force_close` — OANDA 404 recovery in force_close

### Startup Sequence

1. Reload config (picks up latest DISCOVERED_PARAMS)
2. Load all entry + exit models from `ml/models/`
3. Fetch OANDA account balance → initialise RiskManager
4. Initialise OrderManager (GBP rate cache = 1.0 until first candle)
5. Send Telegram startup alert
6. `register_bot()` + initial `write_equity_snapshot()` to dashboard
7. `reconcile_open_trades()` — restore any trades open on OANDA
8. `reconcile_db_trades()` — close phantom DB records
9. Start PriceStream daemon thread
10. Register 7 scheduler jobs → `scheduler.start()`

---

## News Filter

**Modules:** `data/news_fetcher.py` and `risk/news_filter.py`

### Data source
ForexFactory calendar via `nfs.faireconomy.media` JSON feed.
Fetches current week + next week. Cached to `data/cache/news_cache.json`, refreshed
every 12 hours. Falls back to stale cache on network failure (log warning only).
News events also synced to dashboard DB every 12h via `_news_db_sync_job`.

### Rules
| Impact | Rule |
|---|---|
| **High** | Block new entries on affected instruments within **±30 min** of event |
| **High** | Close any open trade on affected instruments **≤15 min** before event (`NEWS_CLOSE` exit reason) |
| **Medium** | Reduce position size to **75%** within **±15 min** of event |

### Integration points
- `bot/main.py` `_process_instrument` — `should_close_pre_news` checked first; calls `order_manager.force_close("NEWS_CLOSE")`
- `bot/main.py` `_process_instrument` — `should_block_entry` checked before `attempt_entry`
- `bot/main.py` `_process_instrument` — `get_size_multiplier` passed to `attempt_entry`
- `execution/order_manager.py` `manage_on_tick` — real-time pre-news close check on every tick (reads cached news — no network call)
- `execution/order_manager.py` `manage_on_tick` — real-time index session EOD close check on every tick

### Currency → instrument mapping
USD → EUR_USD, GBP_USD, AUD_USD, USD_JPY, USD_CHF, SPX500_USD, NAS100_USD
EUR → EUR_USD, DE30_EUR | GBP → GBP_USD | AUD → AUD_USD | JPY → USD_JPY | CHF → USD_CHF

---

## Hard Constraints Reference (for risk/manager.py)

```python
HARD_CONSTRAINTS = {
    "FOREX_PAIRS": ["EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF"],
    "INDEX_INSTRUMENTS": ["SPX500_USD", "NAS100_USD", "DE30_EUR"],
    "ALL_INSTRUMENTS": ["EUR_USD", "GBP_USD", "AUD_USD", "USD_JPY", "USD_CHF",
                        "SPX500_USD", "NAS100_USD", "DE30_EUR"],
    "ACCOUNT_CURRENCY": "GBP",
    "OANDA_ENVIRONMENT": "practice",
    "RISK_PER_TRADE": 0.005,
    "RISK_TIERS": [
        {"max_equity": 5000,  "risk_pct": 0.003},
        {"max_equity": 10000, "risk_pct": 0.005},
        {"max_equity": 999999,"risk_pct": 0.007},
    ],
    "MAX_RISK_PER_TRADE_GBP": {
        "EUR_USD": 50.0, "GBP_USD": 50.0, "AUD_USD": 50.0,
        "USD_JPY": 50.0, "USD_CHF": 50.0,
        "SPX500_USD": 50.0, "NAS100_USD": 50.0, "DE30_EUR": 50.0,
    },
    "MAX_UNITS_PER_TRADE": {
        "EUR_USD": 200000, "GBP_USD": 200000, "AUD_USD": 200000,
        "USD_JPY": 200000, "USD_CHF": 200000,
        "SPX500_USD": 10, "NAS100_USD": 10, "DE30_EUR": 10,
    },
    "MAX_TRADES_PER_INSTRUMENT_PER_DAY": 1,
    "MAX_OPEN_TRADES_FOREX": 3,
    "MAX_OPEN_TRADES_INDICES": 2,
    "MAX_OPEN_TRADES_TOTAL": 4,
    "DAILY_DRAWDOWN_KILL_PCT": 0.03,
    "REENTRY_COOLDOWN_HOURS": 2,
    "REENTRY_CONFIDENCE_FLOOR": 0.70,
    "FOREX_WEEKEND_CLOSE_UTC": "21:00",
    "INDEX_SESSION_CLOSE_UTC": {
        "SPX500_USD": "21:00",
        "NAS100_USD": "21:00",
        "DE30_EUR":   "16:30",
    },
    "TRAINING_START": "2017-01-01",
    "HOLDOUT_START": "2024-01-01",
    "TRAINING_WINDOW_MONTHS": 18,
    "VALIDATION_MONTHS": 3,
    "WALKFORWARD_RETRAIN_DAYS": 20,
    "BACKTEST_SL_ATR_MULT": 1.5,
    "INSTRUMENT_QUOTE_TYPE": {
        "EUR_USD": "usd_quote", "GBP_USD": "usd_quote", "AUD_USD": "usd_quote",
        "USD_JPY": "usd_base",  "USD_CHF": "usd_base",
        "SPX500_USD": "usd_index", "NAS100_USD": "usd_index", "DE30_EUR": "eur_index",
    },
}
```

---

## Telegram Alert Format

**Trade opened:**
```
✅ TRADE OPENED | EUR_USD [Kairos]
Direction: LONG | Confidence: 74%
Entry: 1.09142 | SL: ~18 pips (ATR-based)
Risk: 0.5% | Units: 16,800
FVG: 1.09010–1.09090 | Age: 3 bars
Time: 10:00 UTC
```

**Trade closed:**
```
💚 TRADE CLOSED | SPX500_USD [Kairos]
Exit: 5284.50 | Reason: Exit model signal
P&L: +34.2 pts (£41.20) | MFE: 38.0p | MAE: 5.1p
Hold: 4h 00m (4 bars)
Time: 18:00 UTC
```

**Day skipped:**
```
⏭ SKIPPED | GBP_USD
Best confidence: 0.49 | Threshold: 0.62
No valid FVG setup.
```

**Daily summary (21:05 UTC):**
```
📊 DAILY SUMMARY | Mon 14 Apr
Forex:   EUR_USD ✅ +£28.40 | GBP_USD ⏭ | AUD_USD ⏭ | USD_JPY ❌ -£18.20 | USD_CHF ⏭
Indices: SPX500 ✅ +£41.20 | NAS100 ⏭ | DE30 ⏭
P&L: +£51.40 | Equity: £10,051.40 | DD: -0.0% ✅
Next retrain in: 18 trading days
```

---

## Tech Stack

| Component | Library |
|---|---|
| OANDA API | `oandapyV20` |
| Supplementary data | `yfinance` |
| Entry + Exit models | `xgboost` |
| Feature engineering | `ta==0.11.0` (pandas-ta unavailable on PyPI) |
| Data | `pandas`, `numpy` |
| Class imbalance | `imbalanced-learn` (SMOTE) |
| Timezone | `zoneinfo` (Python 3.11 stdlib) |
| Telegram | `requests` (direct HTTP) |
| Scheduling | `APScheduler` |
| Price streaming | `oandapyV20` PriceStream (daemon thread) |
| Thread safety | `threading.RLock` |
| VPS | Oracle Cloud ARM, Ubuntu 22.04, systemd |

---

## Key Differences from Forex Bot v1

| Dimension | Forex Bot v1 (M15) | Acieral Kairos Bot v1 (1H) |
|---|---|---|
| Timeframe | M15 entry | 1H entry |
| Primary signal | ML pattern on 51 features | FVG structural signal + ML context |
| Higher-TF features | 1H context only | 4H and Daily context (true top-down) |
| Instruments | 5 forex pairs | 5 forex + 3 indices |
| Index handling | None | EOD hard close, session-anchored features |
| Supplementary data | None | VIX, SPY volume for index regime |
| Swing structure | Not explicit | Explicit pivot detection, sweep flags |

---

## v2 Roadmap

| Limitation | v2 Plan |
|---|---|
| Fixed ATR stop-loss | Structure-based SL — place stop beyond the swing point that invalidates setup |
| XGBoost exit model | PPO reinforcement learning agent for sequential exit decisions |
| ~~No news filter~~ | **Done** — ForexFactory JSON feed, 12h cache |
| Manual instrument selection | Automated screening — run labelling grid on all OANDA instruments, select Sharpe > threshold |
| No cross-instrument features | USD index (DXY), VIX regime, cross-pair correlation matrix |
| Single entry TF | 15M entry model with 1H confirmation — lower latency entry |
| Python only | MQL5 EA port — after live validation. ONNX export of entry+exit models + MQL5 feature builder rewrite against pruned feature set. |

---

## Brand History

Originally named **Velox**. Renamed to **Acieral Kairos** on 2026-04-14.
Velox name was already in use commercially.
Acieral is an invented word derived from Latin *acies* (sharp edge, keen sight).
Kairos reflects the primary model signal: temporal/session context (Kairos = the opportune moment).
Git repo and local directory: `acieral-kairos-bot`. VPS target: `/home/ubuntu/acieral-kairos-bot`.
