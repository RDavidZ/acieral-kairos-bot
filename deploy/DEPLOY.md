# Acieral Kairos Bot — Deployment Guide

## Prerequisites
- VPS: 132.145.33.221 (ubuntu@)
- SSH key: ~/.ssh/id_ed25519
- Supabase aureyn-acieral project URLs in .env
- acieral-dashboard.service already running (nginx configured)

---

## Step 1 — Run backup and package locally (PowerShell)

```powershell
cd C:\Users\zinda\projects\acieral-kairos-bot
.\deploy\package.ps1
```

The script prints all included files and total size.
Backup is written to `backups\{YYYY-MM-DD_HHMM}_acieral_kairos\`.

---

## Step 2 — SCP to VPS

```powershell
scp -r -i ~/.ssh/id_ed25519 . ubuntu@132.145.33.221:/home/ubuntu/acieral-kairos-bot
```

This copies the full project (excluding venv, cache parquets, pyc files).
`ml/models/*.pkl` are included — do not omit them.

---

## Step 3 — SSH into VPS and run setup

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@132.145.33.221
cd /home/ubuntu/acieral-kairos-bot
chmod +x deploy/setup_vps.sh
./deploy/setup_vps.sh
```

`setup_vps.sh` does:
1. Creates venv (Python 3.11)
2. `pip install -r requirements.txt`
3. Creates log and cache directories
4. Fetches all market data fresh from OANDA + yfinance (~5 min)
5. Runs preprocessor and feature builder

---

## Step 4 — Install systemd service

```bash
sudo cp deploy/acieral-kairos.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable acieral-kairos
sudo systemctl start acieral-kairos
sudo systemctl status acieral-kairos
```

---

## Step 5 — Verify running

```bash
# Check service status
sudo systemctl status acieral-kairos

# Watch live logs
sudo journalctl -u acieral-kairos -f

# Expected within 60 seconds:
# INFO | bot.main | === Acieral Kairos Bot initialising ===
# INFO | bot.main | Bot ready | instruments=8 | equity=£...
# INFO | bot.main | Acieral Kairos Bot running. Press Ctrl+C to stop.
# (Telegram startup message arrives)
```

---

## Step 6 — Seed dashboard with backtest history

```bash
# On VPS, from /home/ubuntu/acieral-kairos-bot
source venv/bin/activate
python -m deploy.seed_dashboard
```

Seeds all 8 instruments' backtest trade histories and equity curve
from `backtest/results_exit/` into Supabase.

---

## Step 7 — Verify dashboard

Visit https://acieral.aureyn.io

`acieral_kairos_v1` should appear as a second bot card alongside the forex bot.
Backtest equity curve and trade history should be visible.

---

## Redeployment (after code changes)

```powershell
# Local: backup and SCP updated files
.\deploy\package.ps1
scp -r -i ~/.ssh/id_ed25519 . ubuntu@132.145.33.221:/home/ubuntu/acieral-kairos-bot
```

```bash
# VPS: restart service (no setup needed after first deploy)
sudo systemctl restart acieral-kairos
sudo journalctl -u acieral-kairos -f
```

---

## Monitoring commands

```bash
sudo journalctl -u acieral-kairos -f                  # live logs
sudo journalctl -u acieral-kairos --since today        # today's logs
sudo journalctl -u acieral-kairos --since "1 hour ago" # last hour
sudo systemctl status acieral-kairos                   # service status
```

---

## Both bots on same VPS

| Service | Directory | bot_id |
|---|---|---|
| `forex-bot.service` | `/home/ubuntu/acieral-dashboard/` | `acieral_forex_v1` |
| `acieral-kairos.service` | `/home/ubuntu/acieral-kairos-bot/` | `acieral_kairos_v1` |

Both write to the same `aureyn-acieral` Supabase project under different `bot_id` values.
Dashboard shows both as separate cards.

---

## Emergency stop

```bash
sudo systemctl stop acieral-kairos
```

All open OANDA trades remain open — close manually via OANDA web if needed.
Bot resumes cleanly on `systemctl start acieral-kairos` (state reconstructed
from OANDA open trades at startup).

---

## File layout on VPS

```
/home/ubuntu/acieral-kairos-bot/
├── .env                         ← credentials (never commit to git)
├── config.py                    ← with DISCOVERED_PARAMS populated
├── requirements.txt
├── bot/main.py                  ← entry point
├── data/cache/                  ← fetched fresh by setup_vps.sh
├── ml/models/                   ← deployed with SCP (entry + exit .pkl)
├── logs/acieral_kairos.log      ← rotating, 10MB × 5 backups
└── ...
```

---

## Troubleshooting

**Service fails to start:**
```bash
sudo journalctl -u acieral-kairos -n 50 --no-pager
```
Common causes: missing .env, missing model files, Python 3.11 not installed.

**OANDA API errors on startup:**
Check `.env` has valid `OANDA_API_KEY` and `OANDA_ACCOUNT_ID`.
```bash
python -c "from execution.oanda_client import OandaClient; print(OandaClient().get_account())"
```

**Data cache missing after reboot:**
Cache parquets persist in `/home/ubuntu/acieral-kairos-bot/data/cache/`.
They do not need to be re-fetched unless the bot has been offline for
more than a few days. If stale, run:
```bash
source venv/bin/activate && python -c "
from data.fetcher import fetch_all
from data.preprocessor import preprocess_all
from strategy.feature_builder import build_all
fetch_all(); preprocess_all(); build_all()
"
```

**Models missing:**
Ensure SCP included `ml/models/*.pkl`. Re-SCP if needed.
