# Premier League Prediction (ML + Monte Carlo)

This is a football prediction project I made using real Premier League match data and a simple neural network model.  
The goal is to:

- train a model to predict match outcomes (**Home win / Away win / Draw**)
- use those probabilities to **simulate the rest of the season** with Monte Carlo
- output a **predicted final league table** and rough **title chances**

I’m a 2nd year university student, so this project is mainly for learning + showing my workflow.

---

## What it does

### 1) Pull match data (football-data.org API)
- Downloads Premier League match results by season
- Builds a dataset of finished matches

### 2) Feature engineering (basic but useful)
For each match, it creates features like:
- recent **home team home form** (goals for/against)
- recent **away team away form**
- team strength estimates from historical results:
  - points per game (PPG)
  - goal difference per game (GD)

### 3) Train a neural network classifier
- Input: engineered features
- Output: probability of **Home / Away / Draw**
- Uses a **time-based train/validation split** (train on earlier matches, validate on later matches)
- Uses **StandardScaler** (fit on train only to avoid leakage)

### 4) Calibrate probabilities
I use **temperature scaling** to make predicted probabilities less overconfident and improve log-loss.

### 5) Monte Carlo season simulation
For remaining fixtures:
- sample outcomes based on model probabilities
- add a bit of randomness (“match noise”) and optional momentum (“form”) as a simple heuristic
- repeat many times (ex: 1000 simulations)
- average points + count how often each team finishes 1st

---

## Results (what to expect)
- You��ll see metrics like accuracy + log-loss on the validation set
- A printed final table showing:
  - current points
  - projected points
  - estimated title %

These results will change slightly each time because the season simulation is random.

---

## How to run (Colab recommended)

### Option A: Google Colab
1. Open the notebook in Colab
2. Add your API key in **Colab Secrets**:
   - Key name: `TOKEN`
   - Value: your football-data.org API token
3. Run all cells

### Option B: Local (optional)
You’ll need Python with:
- numpy, pandas, requests
- torch
- scikit-learn
- matplotlib, seaborn
- scipy

Then set an environment variable / secret for your token and run the notebook.

---

## Notes / Limitations
- This is not meant to be a perfect betting model (football is chaotic)
- Draws are harder to predict than home/away wins
- The Monte Carlo “noise” + “form boost” is a **simple heuristic**, mainly to make simulations more realistic

---

## Future improvements (ideas)
- add more seasons for training
- include team rating systems (Elo) or rolling xG stats (if available)
- better handling of class imbalance (draws)
- compare against classic models (logistic regression, XGBoost)
- save the model + scaler for reuse

---

## Credits / Data
- Match data: https://www.football-data.org/
- Built with: PyTorch + scikit-learn
