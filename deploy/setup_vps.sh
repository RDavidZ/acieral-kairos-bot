#!/bin/bash
set -e

BOT_DIR=/home/ubuntu/acieral-kairos-bot
VENV=$BOT_DIR/venv

echo "=== Acieral Kairos Bot VPS Setup ==="

# 1. Create directory
mkdir -p $BOT_DIR
cd $BOT_DIR

# 2. Create venv if not exists
if [ ! -d "$VENV" ]; then
    python3.11 -m venv venv
    echo "venv created"
else
    echo "venv already exists — skipping"
fi

# 3. Install requirements
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Requirements installed"

# 4. Create required directories
mkdir -p logs
mkdir -p data/cache
mkdir -p ml/labels_train
mkdir -p ml/exit_labels
mkdir -p ml/multi_combo_results
mkdir -p backtest/results
mkdir -p backtest/results_exit
mkdir -p backtest/results_train
echo "Directories created"

# 5. Fetch all data fresh (this takes ~5 minutes)
echo ""
echo "Fetching market data — this takes ~5 minutes..."
python -c "
from data.fetcher import fetch_all
from data.fetcher_supplementary import fetch_supplementary
from data.preprocessor import preprocess_all
from strategy.feature_builder import build_all
print('Step 1/4: fetching candle data...')
fetch_all()
print('Step 2/4: fetching supplementary data (VIX, SPY)...')
fetch_supplementary()
print('Step 3/4: preprocessing...')
preprocess_all()
print('Step 4/4: building features...')
build_all()
print('Data pipeline complete')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  sudo cp deploy/acieral-kairos.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable acieral-kairos"
echo "  sudo systemctl start acieral-kairos"
echo "  sudo systemctl status acieral-kairos"
