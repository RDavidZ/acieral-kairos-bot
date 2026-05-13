# Acieral Kairos Bot — Self-Review

**Date:** 2026-04-30
**Files reviewed:**
- `bot/main.py`
- `execution/order_manager.py`
- `execution/price_stream.py`
- `risk/manager.py`
- `execution/oanda_client.py`
- `dashboard/db.py`
- `acieral-dashboard/dashboard/api.py` (shared backend — served at `acieral.aureyn.io`)

---

## 1. BUGS

---

### B1 — Double-debit on equity when streaming thread and scheduler both detect SL hit
**Severity:** Critical
**Location:** `execution/order_manager.py` — `manage_on_tick` / `manage_open_trade`

Both threads can independently pass the `get_open_trade(instrument)` check before either has called `register_close_trade`. The streaming thread closes the OANDA trade and calls `register_close_trade` (deleting it from `_open_trades`). The scheduler thread then calls `close_trade` on OANDA, gets a 404, enters the recovery branch, and calls `register_close_trade` a second time. Equity is decremented twice for one trade.

**Why it matters:** On a £10k account, a max-risk losing trade (−£50) records as −£100. The kill-switch threshold is also measured against this corrupted equity figure. Compounds over multiple occurrences.

**Fix:** At the top of the 404 recovery branch in both `manage_open_trade` and `manage_on_tick`, check `self.risk.get_open_trade(instrument) is None` before calling `register_close_trade`. If already None, the streaming thread already registered the close — skip equity mutation, just write DB.

---

### B2 — `MAX_RISK_PER_TRADE_GBP` subscripted as a dict but defined as scalar in CLAUDE.md spec
**Severity:** Critical
**Location:** `execution/order_manager.py:800`, `risk/manager.py:326`

Both files do `_HC["MAX_RISK_PER_TRADE_GBP"][instrument]`. CLAUDE.md defines `"MAX_RISK_PER_TRADE_GBP": 50.0` (scalar). If `config.py` matches the spec, this raises `TypeError: 'float' object is not subscriptable`. In `attempt_entry` this fires *after* `market_order()` succeeds — the trade is live on OANDA but the bot does not call `register_open_trade`. In `compute_position_size` it fires before the order, so no trade is placed and the entry is silently skipped.

**Why it matters:** Order placed on OANDA, not tracked locally. Next reconcile restores it with `atr_at_entry=0.0`, permanently disabling trail SL and exit model for that trade. Verify `config.py` against this immediately.

**Fix:** Either change the config key to a per-instrument dict `{"EUR_USD": 50.0, ...}`, or change the code to `min(equity * risk_pct, _HC["MAX_RISK_PER_TRADE_GBP"])` (scalar).

---

### B3 — `oandapyV20.API` object shared between scheduler thread and streaming thread — not thread-safe
**Severity:** High
**Location:** `execution/oanda_client.py` — `_request_with_retry`, called from both `manage_on_tick` and `on_candle_close`

`self.api.request(request)` sets `request.response` on the endpoint object. If two threads call `self.api.request()` simultaneously, the response of one request can be set on the wrong endpoint object, returning garbage data or raising a key error. The scheduler thread runs `get_latest_candles`, `close_trade`, `update_stop_loss`; the streaming thread runs `close_trade`, `update_stop_loss`, `get_current_price` — all through the same `self.client` instance.

**Why it matters:** Concurrent SL update from the streaming thread and `close_trade` from the scheduler thread can produce incorrect fill prices, failed position closes, or silent data corruption.

**Fix:** Add a `threading.Lock()` in `OandaClient` wrapping every `self.api.request()` call, or instantiate a dedicated `OandaClient` for the streaming thread's emergency closes.

---

### B4 — `mfe_atr` and `mae_atr` stored in DB columns named `mfe_price` / `mae_price`
**Severity:** High
**Location:** `execution/order_manager.py:1054-1055`

```python
"mfe_price": trade.get("mfe_atr"),
"mae_price": trade.get("mae_atr"),
```

The DB columns are named `mfe_price` / `mae_price` but the values stored are ATR-normalised ratios, not prices. A `mfe_price` of `2.3` actually means 2.3× ATR.

**Why it matters:** `get_per_pair_metrics` uses these values. The dashboard's MFE/MAE display is meaningless. Any downstream analytics built on these columns are wrong.

**Fix:** Either rename the DB columns to `mfe_atr` / `mae_atr`, or convert back to price units before writing (`mfe_atr * atr_at_entry + entry_price` for MFE).

---

### B5 — `evaluate_entry` called twice per bar — model runs twice
**Severity:** Medium
**Location:** `bot/main.py:374` and `execution/order_manager.py:721`

`_process_instrument` calls `evaluate_entry` to get confidence for `_best_confidence` tracking, then calls `attempt_entry` which internally calls `evaluate_entry` again. XGBoost inference runs twice per bar per instrument.

**Why it matters:** 16 model inferences instead of 8 at HH:01. On retrain day at 00:05 both jobs overlap. More critically, if confidence crosses the threshold between the two calls due to a model reload mid-retrain, the wrong confidence is used for the entry decision.

**Fix:** Return confidence from `attempt_entry` (it already computes it internally) and use that to update `_best_confidence`.

---

### B6 — `force_close` 404 path does not send Telegram alert
**Severity:** Medium
**Location:** `execution/order_manager.py:670-675`

In `force_close`, when OANDA returns 404 (trade already closed), the code calls `register_close_trade` and `_write_trade_to_db` but not `_send_close_alert`. The trade silently disappears with no Telegram notification.

**Why it matters:** A pre-news close that finds OANDA already executed the SL will go unnoticed. P&L unknown until the next daily summary.

**Fix:** Add `self._send_close_alert(instrument, trade, close_price, pnl_gbp, actual_exit_reason, now)` after the `register_close_trade` call in the `force_close` 404 branch (and verify the same in `manage_open_trade`'s 404 branch — that path does send the alert).

---

### B7 — Kill switch activates with no Telegram alert
**Severity:** Medium
**Location:** `risk/manager.py:222-228`

When `equity < drawdown_floor`, `kill_switch = True` is set and logged. No Telegram notification is sent.

**Why it matters:** The bot silently stops trading for the rest of the day. You won't know until the 21:05 daily summary or by checking logs manually.

**Fix:** Add a callback or pass a `telegram` reference to `RiskManager`. Alternatively, detect the kill-switch transition in `register_close_trade` and fire the alert from `manage_open_trade` / `manage_on_tick` after registering the close.

---

### B8 — `daily_reset` sets stale `day_start_equity` before OANDA sync
**Severity:** Medium
**Location:** `bot/main.py:406-414`

```python
self.risk.daily_reset(now)           # sets day_start_equity = self.risk.equity (stale)
...
self.risk.equity = account["balance"]        # updates equity
self.risk.day_start_equity = account["balance"]   # corrects it
```

If the equity sync fails (caught by `except exc: log.warning`), `day_start_equity` is set from the stale in-memory value rather than the real OANDA balance.

**Why it matters:** Kill-switch drawdown floor is computed from a wrong baseline. Possible false kill-switch activation or failure to trigger it.

**Fix:** Sync equity from OANDA *before* calling `daily_reset`, or restructure `daily_reset` to accept an optional `new_equity` argument.

---

### B9 — `_get_conn` does not validate the replacement connection after a stale close
**Severity:** Low
**Location:** `dashboard/db.py:159-171`

If the test `SELECT 1` fails, the code closes the bad connection then calls `_POOL.getconn()` again and returns it immediately without re-testing. A mass connection expiry (Supabase idle timeout, Pgbouncer recycle) would cause every connection to fail the test and immediately return another bad one, producing errors on every DB write for the current request.

**Fix:** Loop the validity check on the replacement connection, or rely on caller-level exception handling (already in place) and remove the optimistic second `getconn`.

---

### B10 — Weekend early-return blocks index trade management if bot restarts Saturday
**Severity:** Low
**Location:** `bot/main.py:231-232`

`if now.weekday() >= 5: return` exits before checking for open index trades. If the bot was down at 16:30 Friday (DE30 session close) and restarts Saturday, it skips all trade management.

**Why it matters:** OANDA's hard SL still protects the position, but the trade stays unrecorded as open until Monday candle close.

**Fix:** Keep the weekend early-return for entry evaluation only. Move open-trade management before the weekend check.

---

## 2. RACE CONDITIONS

---

### R1 — Trade dict mutated from two threads without lock
**Severity:** Critical
**Location:** `risk/manager.py:267-269` → `execution/order_manager.py` — `manage_on_tick` and `manage_open_trade`

`get_open_trade` acquires `_lock`, returns the *actual dict reference* from `_open_trades` (not a copy), then releases the lock. Both the scheduler thread (`manage_open_trade`) and the streaming thread (`manage_on_tick`) hold this same reference and freely mutate it without a lock.

**Unguarded access points:**

| Location | Field written |
|---|---|
| `manage_on_tick:535` | `trade["mfe_atr"]` |
| `manage_on_tick:541` | `trade["trail_activated"]` |
| `manage_on_tick:548/553/573/578` | `trade["sl_price"]` |
| `manage_on_tick:552/577` | `trade["last_sl_update_time"]` |
| `manage_open_trade:244-246` | `trade["mfe_atr"]`, `trade["mae_atr"]`, `trade["bars_held"]` |
| `manage_open_trade:296` | `trade["trail_activated"]` |
| `manage_open_trade:315` | `trade["milestones_hit"]` |

**Why it matters:** Concurrent writes to `mfe_atr` can leave a lower value, preventing trail activation. Concurrent writes to `sl_price` can reverse the ratchet. Dict mutation in CPython is not atomic for compound operations.

**Fix:** `get_open_trade` should return a copy for reading. All mutations must go through dedicated methods in `RiskManager` that acquire `_lock` (analogous to the existing `update_sl`). Add `update_mfe`, `update_bars_held`, etc.

---

### R2 — `reconcile_open_trades` bypasses lock when mutating `_open_trades`, `_open_forex`, `_open_indices`
**Severity:** High
**Location:** `execution/order_manager.py:879-924`

Directly accesses and deletes from `self.risk._open_trades` and modifies `self.risk._open_forex` / `_open_indices` without acquiring `self.risk._lock`. Called at startup and at 00:05 UTC daily reset while the streaming thread is live.

**Why it matters:** If the streaming thread is mid-tick and calling `register_close_trade` (under lock) simultaneously, the reconcile thread's unguarded `del self.risk._open_trades[instrument]` can corrupt the dict.

**Fix:** Move reconciliation logic into `RiskManager` as a method that acquires `_lock` internally.

---

### R3 — `daily_reset` clears shared state without lock
**Severity:** High
**Location:** `risk/manager.py:241-253`

`_traded_today.clear()` and `_pair_trade_count.clear()` execute without `with self._lock:`. Inconsistent with `register_open_trade` which writes to these under lock.

**Why it matters:** APScheduler timing edge case (concurrent misfire) could cause two daily resets simultaneously, interleaving clears with writes.

**Fix:** Wrap the entire body of `daily_reset` in `with self._lock:`.

---

### R4 — Streaming thread makes OANDA API calls on the shared `OandaClient` instance
**Severity:** Medium
**Location:** `execution/order_manager.py` — `manage_on_tick`

After a tick-triggered SL close, `manage_on_tick` calls `close_trade`, `_compute_pnl_gbp` → `get_current_price`, and `update_stop_loss` — all through `self.client`. Simultaneously, the scheduler may be mid-request (e.g., `get_latest_candles` for 500 candles). See also B3.

**Fix:** Same as B3 — lock all `self.api.request()` calls or use a dedicated client for the streaming thread.

---

### R5 — Trail SL throttle time-check race
**Severity:** Medium
**Location:** `execution/order_manager.py:546-565`

Two consecutive ticks arriving close together (both outside the 60s window) can both read `last_sl_update_time` before either sets it, causing two API SL update calls for the same interval.

**Fix:** Set `trade["last_sl_update_time"] = now` before the API call (optimistic update), so the second tick immediately sees a recent timestamp even if the first API call is still in-flight.

---

## 3. GAPS

---

### G1 — No equity snapshot on candle-close trade exits
**Severity:** High
**Location:** `execution/order_manager.py` — `manage_open_trade`, `force_close`

`manage_on_tick` writes an equity snapshot after SL close (lines 513–517). `manage_open_trade` and `force_close` do not. The dashboard equity curve only updates at midnight.

**Why it matters:** An exit-model close at 14:00 doesn't appear on the equity chart until midnight. A day with multiple trades shows a flat equity line.

**Fix:** Call `write_equity_snapshot("acieral_kairos_v1", self.risk.equity)` inside `manage_open_trade` and `force_close` after `register_close_trade`, mirroring the `manage_on_tick` pattern.

---

### G2 — Index EOD close has up to 59-minute gap if candle-close job misfires
**Severity:** High
**Location:** `execution/order_manager.py` — `manage_on_tick`

DE30 session close is 16:30 UTC. The next candle close fires at 17:01. If the 17:01 job misfires beyond `misfire_grace_time=60`, the position holds until 18:01. The streaming thread has no EOD check.

**Why it matters:** Holding a DE30 position post-16:30 violates the hard constraint. OANDA may widen spreads significantly on a closed market.

**Fix:** Add an EOD check in `manage_on_tick`:
```python
if instrument in INDEX_INSTRUMENTS:
    close_hm = _SESSION_CLOSE_UTC[instrument]
    if (now.hour, now.minute) >= close_hm:
        # trigger force_close("EOD")
```
This fires within the next tick after session close.

---

### G3 — Restored trades lose trail/exit model permanently after restart
**Severity:** High
**Location:** `execution/order_manager.py:909`, `bot/main.py:331-339`

`reconcile_open_trades` restores trades with `atr_at_entry=0.0`. The ATR patch in `_process_instrument` fixes this on the first candle close, but the restored trade also has `mfe_atr=0.0` and `bars_held=0`. If the trade was near trail activation before restart, that state is lost and the initial hard SL becomes the only protection.

**Fix:** Persist the full trade state dict to a JSON file on every update. On restore, load from file rather than constructing a bare dict.

---

### G4 — `on_retrain` does not reload `exit_models`
**Severity:** Medium
**Location:** `bot/main.py:562-564`

Only entry models are reloaded into `self.order_manager.entry_models`. Exit models are not updated. Manually retrained exit models are not picked up until bot restart.

**Fix:** Add the same reload pattern for exit models inside `on_retrain`.

---

### G5 — PriceStream instruments list never updates after retrain
**Severity:** Medium
**Location:** `bot/main.py:169-177`

`PriceStream` is instantiated once with `self.active_instruments`. If retrain changes `ACTIVE_INSTRUMENTS`, the stream still subscribes to the original list — new instruments get no real-time SL protection.

**Fix:** After retrain, compare old vs new `ACTIVE_INSTRUMENTS`. If different, call `self._stream.stop()` and reinitialise with the new list.

---

### G6 — `_best_confidence` not updated when news-blocked
**Severity:** Medium
**Location:** `bot/main.py:369-372`

When a news block triggers early return, `evaluate_entry` is not called. `_best_confidence[instrument]` stays at 0.0. The next day's skip alert shows 0.0 confidence even if a genuine high-confidence setup was blocked.

**Fix:** Call `evaluate_entry` before the news block check and store confidence in `_best_confidence` regardless, or log news-blocked bars separately.

---

### G7 — `get_bot_metrics_split` missing `live` trade_type split
**Severity:** Low
**Location:** `dashboard/db.py:877-919`

CLAUDE.md states `get_bot_metrics()` returns `{overall, backtest, practice, live}`. The implementation only returns `overall`, `backtest`, `practice`. A `live` filter is never applied.

**Fix:** Add `live_rows = [r for r in all_closed if r.get("trade_type") == "live"]` and include it in the return dict.

---

### G8 — No daily summary record persisted to DB
**Severity:** Low
**Location:** `bot/main.py` — `on_daily_summary`

Daily summary sent to Telegram but no DB record written. Counts of news-blocked, skipped, and traded instruments are lost. Only trade-level P&L is recoverable via `get_daily_pnl`.

---

## 4. REAL-TIME IMPROVEMENTS

---

### RT1 — Index EOD close should trigger from streaming thread

Currently relies on HH:01 scheduler, leaving up to a 31-minute gap for DE30. Adding to `manage_on_tick`:

```python
if instrument in INDEX_INSTRUMENTS:
    if (now.hour, now.minute) >= _SESSION_CLOSE_UTC[instrument]:
        # trigger force_close("EOD")
```

fires within seconds of session close. No model integrity concern — it is a hard constraint, not a model decision.

---

### RT2 — Pre-news close should trigger from streaming thread

Currently checked at HH:01. A high-impact news event at 14:30 is not caught by the bot until the next candle close. Adding `should_close_pre_news(instrument, now)` to `manage_on_tick` would catch it within a tick. The function is a pure datetime comparison (no network call), so latency cost is negligible.

---

### RT3 — Unrealised P&L not visible on dashboard between trade closes

The equity curve is flat between midnight snapshots. Adding a lightweight NAV-based snapshot every 15–30 minutes from the main keep-alive loop (or a dedicated `CronTrigger`) would give the dashboard a live equity line during active trading sessions.

---

### RT4 — Tick-path SL close fetches GBP/USD rate via HTTP on the streaming thread

`manage_on_tick` calls `_compute_pnl_gbp` → `get_current_price()` HTTP request on the streaming parser thread, blocking tick processing for the duration of the API call. The fill price from `close_trade` already provides accurate close data. The conversion rate could use a background-refreshed cached value (updated every minute in the main loop) instead of a synchronous fetch.

---

## 5. DASHBOARD

---

### D1 — No live bot status / heartbeat
**Severity:** High

The dashboard has no way to know if the bot is running. No heartbeat timestamp in the DB. If the bot crashes between midnight snapshots, the dashboard shows stale "open" positions with no indication of bot health.

**Fix:** Write a heartbeat row (or update a `last_seen` column on the bots table) every 30 minutes from the main keep-alive loop. Display "last seen X minutes ago" in the dashboard.

---

### D2 — Equity curve is daily-only; intraday changes invisible
**Severity:** High

`write_equity_snapshot` is called at startup and at 00:05 daily reset only. The equity chart shows a step function with 23-hour flat segments. Only tick-SL closes get a mid-day snapshot (via `manage_on_tick`). See also G1.

**Fix:** Call `write_equity_snapshot` after every trade close in all code paths.

---

### D3 — acieral's `dashboard/db.py` is missing functions that `api.py` calls — multiple endpoints crash at runtime
**Severity:** Critical

The shared API (`acieral-dashboard/dashboard/api.py`) is ahead of acieral's `dashboard/db.py`. Every call to these missing or mismatched functions returns HTTP 500:

| Endpoint | Function called | Status in acieral db.py |
|---|---|---|
| `GET /api/public/bots/{bot_id}/metrics` | `db.get_bot_pnl_since_start(bot_id)` | Missing |
| `GET /api/bots/{bot_id}/metrics` | `db.get_bot_pnl_since_start(bot_id)` | Missing |
| `GET /api/public/account/equity` | `db.get_account_equity()` | Missing |
| `GET /api/bots/{bot_id}/daily_pnl?trade_type=...` | `get_daily_pnl(..., trade_type=...)` | No `trade_type` param |
| `GET /api/bots/{bot_id}/heatmap?trade_type=...` | `get_heatmap_data(..., trade_type=...)` | No `trade_type` param |
| `GET /api/bots/{bot_id}/monthly?trade_type=...` | `get_monthly_returns(..., trade_type=...)` | No `trade_type` param |

**Why it matters:** BotDetail metrics panel, account equity widget, daily P&L chart with trade-type filtering, heatmap, and monthly returns are all broken in the live dashboard right now.

**Fix:** Sync acieral's `dashboard/db.py` with `acieral-dashboard/dashboard/db.py`. Specifically add `get_bot_pnl_since_start`, `get_account_equity`, and `trade_type` filter parameters to `get_daily_pnl`, `get_heatmap_data`, `get_monthly_returns`.

---

### D4 — No kill-switch status exposed to dashboard
**Severity:** Medium

`self.risk.kill_switch` is not persisted or exposed via any API endpoint. The dashboard shows the bot as operational when it has stopped trading for the day due to drawdown.

---

### D5 — `_find_bot` in `api.py` triggers double N+1 query on every metrics request
**Severity:** Medium
**Location:** `acieral-dashboard/dashboard/api.py:51-55`, `api.py:75`

`public_bot_metrics` calls `_find_bot(bot_id)` → `db.get_all_bots()` (N+1 per bot), then calls `db.get_bot_metrics(bot_id)` separately. With 2 bots registered, each metrics request fires ~5 SQL queries.

**Fix:** Add `db.get_bot(bot_id)` that fetches a single bot row and use it in `_find_bot` instead of loading all bots.

---

### D6 — `get_monthly_returns` uses registration-time `initial_equity` as perpetual denominator
**Severity:** Medium
**Location:** `dashboard/db.py:1013` (acieral), `acieral-dashboard/dashboard/db.py:1101`

Monthly returns calculated as `pnl / initial_equity * 100`. As equity grows the denominator stays fixed, increasingly overstating monthly return percentages.

**Fix:** Use the equity at the first snapshot of each month as the denominator.

---

### D7 — `public_news_status` silently returns `"clear"` on any exception
**Severity:** Medium
**Location:** `acieral-dashboard/dashboard/api.py:177-183`

Bare `except Exception: return {"status": "clear", ...}` means a corrupt cache file or `news_filter` bug shows "no risk" when trading may actually be blocked.

**Fix:** Log the exception before returning the fallback. Return `"status": "unknown"` to distinguish a genuine clear from a failed check.

---

### D8 — `get_heatmap_data` and `get_monthly_returns` mix all trade types
**Severity:** Low
**Location:** `dashboard/db.py` — `get_heatmap_data`, `get_monthly_returns`

Both functions include all trade types. Backtest trades (thousands of rows) dominate both views, masking live/practice performance patterns. Already partially addressed in `api.py` which passes `trade_type` — but acieral's `db.py` ignores that parameter (see D3).

---

### D9 — `CORS allow_origins=["*"]` on authenticated endpoints
**Severity:** Low
**Location:** `acieral-dashboard/dashboard/api.py:19-24`

CORS middleware allows all origins on all routes, including authenticated `/api/bots/*`. Any web page can make requests with a valid token.

**Fix:** Restrict `allow_origins` to `["https://acieral.aureyn.io"]` in production.

---

## TOP 3 FIXES — Live Right Now

### Fix 1 — Double-debit on equity (B1 + R1)

Most dangerous financial bug. Currently occurring any time an SL is hit via the stream and the scheduler fires within the same second. The fix is two lines added to the 404 recovery branches in `manage_open_trade` and `manage_on_tick`:

```python
if self.risk.get_open_trade(instrument) is None:
    return actual_exit_reason   # streaming thread already handled it
```

### Fix 2 — Sync acieral `dashboard/db.py` with acieral-dashboard `dashboard/db.py` (D3)

The metrics panel, account equity widget, and all trade-type filtered views on the live dashboard are broken right now. This is the primary monitoring surface while the bot is running live trades. Without it you cannot verify the bot's equity, P&L split by type, or per-instrument performance.

### Fix 3 — Trade dict unguarded concurrent mutation (R1)

Trail SL integrity depends on `mfe_atr` being correct. Concurrent writes from both threads can silently suppress trail activation or move the SL backwards, leaving the position protected only by the initial hard SL and not the ratcheted trail. Intermediate fix without full refactor: wrap all trade dict mutations inside `manage_on_tick` in `with self.risk._lock:` after obtaining the trade reference.
