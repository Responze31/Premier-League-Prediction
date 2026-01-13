# Premier League Prediction

Neural network + Monte Carlo simulation for predicting Premier League match outcomes and final standings.

## Features

- Match outcome prediction (Home/Away/Draw)
- Temperature-scaled probability calibration
- Monte Carlo season simulation (1000+ runs)
- Final league table with title probabilities
- Data inspection and visualization throughout analysis

## Tech Stack

PyTorch, scikit-learn, pandas, scipy, matplotlib, seaborn

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with your [football-data.org](https://www.football-data.org/) API token:
   ```
   TOKEN=your_api_token_here
   ```

3. Run `notebooks/footballPred.ipynb`

## How It Works

### Features
- Rolling home/away form (goals for/against, last 5 games)
- Team strength (PPG, goal difference)

### Model
- 3-layer MLP: 128 → 64 → 32 → 3 (with dropout)
- 80/20 time-based train/val split
- Temperature scaling for calibration

### Simulation
- Sample from calibrated probabilities
- Add noise (σ=0.20) and momentum effects
- Aggregate over 1000 simulations

## Performance

- Accuracy: ~50% (baseline ~45%)
- Log-loss: ~1.05

## Outputs

Generated in `outputs/`:
- `predicted_table_[timestamp].csv` - Final standings
- `metrics_[timestamp].json` - Model metrics
- `confusion_matrix_[timestamp].png`
- `calibration_[timestamp].png`

## Project Structure

```
├── notebooks/footballPred.ipynb  # Main notebook
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   └── model.py
├── outputs/
└── requirements.txt
```

## Limitations

- No player-level data or injury info
- Draws are hardest to predict
- Single season validation set

## Data

Match data from [football-data.org](https://www.football-data.org/)
