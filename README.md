# Acieral Kairos — Algorithmic Trading Bot

A production algorithmic trading system that trains per-instrument XGBoost models on 7+ years of multi-timeframe forex and equity index data, validates them with walk-forward backtesting, and runs live execution against an OANDA brokerage account. The full pipeline — data collection, feature engineering, model training, backtesting, and live deployment — is automated end-to-end.

**8 instruments** (5 forex pairs + 3 equity indices) | **81 engineered features** | **Walk-forward validation across 26 folds** | **Live on OANDA**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Broker API | oandapyV20 (OANDA REST) |
| ML | XGBoost, scikit-learn, imbalanced-learn |
| Feature engineering | pandas, numpy, ta==0.11.0 |
| Supplementary data | yfinance (VIX, SPY volume) |
| Data storage | Parquet (pyarrow / fastparquet) |
| Scheduling | APScheduler |
| Alerts | Telegram Bot API |
| Deployment | paramiko (SSH), python-dotenv |

---

## How It Works

### Research Pipeline (`run_pipeline.py`)

A 9-step automated pipeline that takes raw candle history and produces deployment-ready model weights and strategy parameters:

```
Raw OANDA candles
    → Preprocessing + 81-feature engineering
    → Entry labeler grid search (N/T parameter sweep)
    → Walk-forward model training — entry models per instrument
    → Entry-only backtest (training windows)
    → Multi-combo evaluation on 2025 holdout
    → Exit labeler generation from backtest trades
    → Exit model walk-forward training
    → Full backtest (entry + exit) on holdout
    → Config generator writes discovered params to config.py
```

### Live Bot

Each hour at candle close, the bot:
1. Fetches multi-timeframe OANDA candles (1H, 4H, D1, W1) and updates the feature cache
2. Runs the entry model per instrument — decides direction and confidence
3. If confidence exceeds the discovered threshold, sizes and submits a market order
4. For open positions, runs the exit model and manages a trailing stop-loss
5. Enforces all risk constraints: daily kill switch, position limits, cooldowns, weekend close

---

## Feature Engineering (81 Features)

Built fresh at each 1H candle close. All price distances are ATR-normalised — no raw price values enter any model.

| Group | Features | Count |
|---|---|---|
| Fair Value Gaps | FVG detection (H1 + H4 + D1), distance to nearest gap, gap size, age, fill %, stacked alignment | 12 |
| Swing Structure | Swing highs/lows, structural breaks (BOS/CHoCH), sweep flags, directional bias score | 9 |
| Session & Time | UTC hour, session flags (Asian/London/NY/overlap), hours since session open, day of week | 10 |
| HTF Context | H4 + D1 + W1 trend, EMA slopes, range position, HTF alignment flag | 17 |
| Volatility & Regime | ATR ratio, candle body/wick ratios, rolling range, ADX | 8 |
| Momentum | RSI + divergence flags, MACD, rate of change | 8 |
| Candle Context | Close lags, intraday displacement, consecutive directional candles | 14 |
| Supplementary | VIX close + delta, SPY volume ratio (index instruments only) | 3 |

**Top features by walk-forward importance:** session timing and daily range position dominate. The model is fundamentally a *time-of-day × daily-range-position × H4-trend classifier*, with H1 FVGs as a trigger filter.

---

## Machine Learning Design

### Entry Model

- **Algorithm:** XGBoost multi-class classifier (NO_TRADE / LONG / SHORT)
- **Labels:** Generated per instrument via a best-opportunity algorithm — for each day, the bar with the highest achievable profit (within N lookahead bars, above threshold T) is labelled as the signal. All others are NO_TRADE.
- **Walk-forward:** 18-month training windows, 3-month validation, 3-month step — ~26 folds per instrument over 2017–2024
- **Per-fold training:** two-pass — hyperparameter search on SMOTE-balanced data, then retrain on raw data with inverse-frequency sample weights
- **Combo selection:** Top-3 (N, T) combinations trained and evaluated on 2025 holdout; winner selected by holdout Sharpe ratio

### Exit Model

- **Algorithm:** XGBoost binary classifier (STAY / EXIT)
- **Features:** 81 market features + 7 trade-context features (bars held, unrealised P&L, MFE, MAE, fraction of peak given back, direction, distance to stop)
- **Labels:** Oracle peak exit — the bar of maximum unrealised P&L for each winning trade
- **Fold assignment:** by trade entry time (all bars of one trade stay in the same fold — no leakage)
- **Parameters:** Trailing stop activation multiplier and exit confidence threshold tuned on holdout backtest

### Data Splits

| Split | Period | Purpose |
|---|---|---|
| Training | 2017–2024 | Walk-forward model training |
| Holdout 1 | 2025 | Model selection (combo, exit params) |
| Holdout 2 | 2026–present | Sealed — untouched |

---

## Risk Management

```python
RISK_TIERS = [
    {"max_equity": 2000,   "risk_pct": 0.010},
    {"max_equity": 10000,  "risk_pct": 0.005},
    {"max_equity": 20000,  "risk_pct": 0.003},
]
MAX_RISK_PER_TRADE_GBP = 50.0      # hard cap regardless of tier

# Kill switch: daily drawdown > -3% halts all new entries
# Max concurrent: 5 forex + 3 indices
# One trade per instrument per day
# Re-entry cooldown: 2 hours after any close
# Forex weekend close: Friday 21:00 UTC
# Index EOD close: session-end per instrument
```

Position sizing uses live OANDA price data — no hardcoded pip values. The discovered strategy parameters (confidence thresholds, ATR multipliers, trailing stop levels) are written to `config.py` by the pipeline's config generator and never hand-edited.

---

## Architecture

```
data/           OANDA candle fetch → parquet cache → incremental updates
strategy/       Feature builder: FVG, swing structure, session, HTF, momentum (81 features)
ml/             Walk-forward training, labeler grid search, entry + exit model pipeline
backtest/       Full simulation: position sizing, SL/trailing stop, exit model, trade logging
risk/           Kill switch, daily drawdown, position limits, cooldown enforcement
execution/      OANDA order placement, price stream daemon, live trade management
notifications/  Telegram alerts (trade open/close/skip, daily summary)
bot/            APScheduler jobs + main loop
deploy/         Remote deployment via SSH/SCP
```

**Dependency order:** `data → strategy → ml → backtest → execution`

No shared state between the research pipeline and the live bot beyond the model `.pkl` files and `config.py`.

---

## Instruments

| Symbol | Class | Walk-Forward Sharpe |
|---|---|---|
| EUR_USD | Forex | — |
| GBP_USD | Forex | — |
| AUD_USD | Forex | — |
| USD_JPY | Forex | — |
| USD_CHF | Forex | — |
| DE30_EUR | Index | 15.91 |
| NAS100_USD | Index | 9.92 |
| UK100_GBP | Index | — |

Per-instrument models — weights are never shared across symbols.

---

## Running

```bash
# Full research pipeline
python run_pipeline.py

# Resume from a specific step if interrupted
python run_pipeline_resume.py --from-step 5

# Live bot
python -m bot.main
```

---

## Related

- **[kairos-ftmo](https://github.com/RDavidZ/kairos-ftmo)** — Port of this bot to MetaTrader 5 for FTMO-funded and FunderPro accounts. Replaces the OANDA execution layer with MT5 lot-based ordering; all models and features are copied directly from this repo. Runs 4 accounts concurrently on a Windows VPS.
- **[Final-Project---Applied-Data-Science-Capstone](https://github.com/RDavidZ/Final-Project---Applied-Data-Science-Capstone)** — End-to-end data science project: SpaceX launch data collection, SQL EDA, classification modelling, and an interactive Plotly Dash dashboard.
