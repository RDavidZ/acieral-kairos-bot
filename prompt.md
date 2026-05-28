Read CLAUDE.md in acieral-kairos-bot before doing anything.
Extract the VPS IP, SSH user, and any connection details from that file.
Do not print credentials or keys.

Connect to the VPS via SSH using the details in CLAUDE.md.
No password — use the existing SSH key.

Then do the following steps in order, confirming each before proceeding:

STEP 1 — Check current bot status:
    sudo systemctl status acieral-kairos
Report: is it running, how long has it been up, any recent errors in journalctl.

STEP 2 — Stop the bot gracefully:
    sudo systemctl stop acieral-kairos
    sudo systemctl disable acieral-kairos
Confirm it is stopped: sudo systemctl status acieral-kairos

STEP 3 — Check for any open OANDA positions:
    cd ~/acieral-kairos-bot
    source venv/bin/activate
    python -c "
from execution.oanda_client import OandaClient
import os
from dotenv import load_dotenv
load_dotenv()
c = OandaClient(os.getenv('OANDA_API_KEY'), os.getenv('OANDA_ACCOUNT_ID'), 'practice')
trades = c.get_open_trades()
print(f'Open trades: {len(trades)}')
for t in trades:
    print(t)
"
If any open trades exist: report them and wait for instruction before closing.
If zero open trades: proceed to Step 4.

STEP 4 — Commit all files to git:
    cd ~/acieral-kairos-bot
    git add -A
    git commit -m "archive: final state before v4 replacement — $(date -u +%Y-%m-%d)"
    git push origin main
Report the commit hash.

STEP 5 — Remove the bot files (keep the git repo, remove working directory):
    cd ~
    rm -rf ~/acieral-kairos-bot
Confirm directory no longer exists: ls ~ | grep acieral

STEP 6 — Remove the systemd service file:
    sudo rm /etc/systemd/system/acieral-kairos.service
    sudo systemctl daemon-reload
Confirm removal.

Report completion of all 6 steps with status of each.
Do not touch forex-bot.service or any other service on the VPS.
Do not touch acieral-kairos-v4 local build.