# Confidence Threshold Comparison — May 19–22 2026

**Test window:** 2026-05-19 to 2026-05-23  |  **Models:** retrained 2026-05-22 (corrected HTF features)

## Per-Instrument Results

### EUR_USD

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 3 | 66.7% | £2.44 | 1.31 | 0.1% | 1.07 |
| flat_0.5 | 4 | 25.0% | £-0.29 | 0.00 | 1.1% | 1.00 |
| flat_0.6 | 4 | 50.0% | £105.78 | -0.68 | 0.4% | 2.44 |

### GBP_USD

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 4 | 25.0% | £-72.84 | -17.82 | 0.7% | 0.34 |
| flat_0.5 | 4 | 25.0% | £-91.25 | -15.69 | 0.5% | 0.17 |
| flat_0.6 | 4 | 50.0% | £-17.08 | -8.63 | 0.2% | 0.77 |

### AUD_USD

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 0 | 0.0% | £0.00 | 0.00 | 0.0% | 0.00 |
| flat_0.5 | 4 | 75.0% | £138.38 | 5.99 | 0.4% | 4.76 |
| flat_0.6 | 4 | 100.0% | £205.34 | 22.18 | 0.0% | 999.99 |

### USD_JPY

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 1 | 0.0% | £-22.01 | -9.17 | 0.2% | 0.00 |
| flat_0.5 | 4 | 50.0% | £-11.92 | -1.12 | 0.5% | 0.85 |
| flat_0.6 | 4 | 75.0% | £47.75 | 4.55 | 0.3% | 2.63 |

### USD_CHF

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 4 | 75.0% | £169.82 | 4.39 | 0.5% | 4.39 |
| flat_0.5 | 4 | 75.0% | £260.47 | 9.37 | 0.5% | 6.22 |
| flat_0.6 | 4 | 75.0% | £161.18 | 3.89 | 0.5% | 4.22 |

### SPX500_USD

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 4 | 100.0% | £227.68 | 43.60 | 0.0% | 999.99 |
| flat_0.5 | 4 | 50.0% | £34.15 | 6.56 | 0.5% | 1.34 |
| flat_0.6 | 4 | 50.0% | £38.66 | 6.80 | 0.5% | 1.39 |

### NAS100_USD

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 3 | 66.7% | £41.61 | 2.32 | 0.5% | 1.83 |
| flat_0.5 | 4 | 50.0% | £13.51 | 5.44 | 0.5% | 1.14 |
| flat_0.6 | 4 | 50.0% | £23.10 | 6.01 | 0.5% | 1.23 |

### DE30_EUR

| Scenario | Trades | Win% | P&L £ | Sharpe | Max DD | PF |
|----------|--------|------|-------|--------|--------|----|
| curve | 4 | 50.0% | £-40.06 | 1.04 | 0.5% | 0.60 |
| flat_0.5 | 4 | 50.0% | £-53.83 | -0.43 | 0.5% | 0.46 |
| flat_0.6 | 4 | 25.0% | £-140.51 | -14.21 | 1.0% | 0.06 |

## Portfolio Summary

| Scenario | Total P&L £ | Avg Win% | Avg Sharpe | Avg Max DD | Total Trades |
|----------|-------------|----------|------------|------------|--------------|
| curve | £306.64 | 47.9% | 3.21 | 0.3% | 23 |
| flat_0.5 | £289.22 | 50.0% | 1.27 | 0.6% | 32 |
| flat_0.6 | £424.22 | 59.4% | 2.49 | 0.4% | 32 |

## Notes

- `curve`: CONFIDENCE_CURVE from DISCOVERED_PARAMS (per-hour, per-instrument ~0.74–0.83+)
- `flat_0.5`: flat 0.50 — more permissive, no hour-based filtering
- `flat_0.6`: flat 0.60 — small safety floor
- Feature parquets: rebuilt 2026-05-22 with corrected H4/D1 merge-shift
- Only the entry confidence gate differs between scenarios