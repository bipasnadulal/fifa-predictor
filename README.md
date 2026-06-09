# FIFA World Cup 2026 Match Predictor

I built this project to predict the outcome of every match in the 2026 FIFA World Cup — all 104 of them, from the group stage all the way to the Final.

The model uses real historical match data and machine learning to predict scorelines, win probabilities, and simulate the entire tournament bracket.

🔗 **Live app:** https://fifa-predictor-10.streamlit.app/

---

## What it does

- Predicts the score and winner for every group stage match
- Simulates full group standings and who advances
- Predicts every knockout round match up to the Final
- Lets you pick any two teams and get an instant prediction
- Shows win probabilities for every match

---

## How I built it

I started with a simple statistical model and upgraded it to a proper machine learning model. Here's the full journey:

**1. Data collection**
Downloaded over 47,000 historical international football match results from Kaggle, going all the way back to 1872. Also pulled FIFA world rankings data.

**2. Data cleaning**
Filtered down to matches from 2014 onwards and kept only teams that qualified for the 2026 World Cup. Removed noise and standardized team names.

**3. Feature engineering**
For each team I calculated things like average goals scored, average goals conceded, win rate, recent form (last 10 matches), and FIFA ranking — all measured _before_ each match so the model never sees future data.

**4. Building the training dataset**
Turned every historical match into a row of features representing what we knew about both teams before that match was played. This gave us a clean dataset to train on.

**5. Training the ML models**
Trained three XGBoost models:

- An outcome classifier (home win / draw / away win)
- A home goals regressor
- An away goals regressor

**6. Tournament simulation**
Used the trained models to simulate all 72 group stage matches, calculate standings, pick who advances, and then simulate the full knockout bracket round by round.

**7. Building the dashboard**
Built a simple interactive web app using Streamlit and Plotly so anyone can explore the predictions without touching any code.

**8. Deployment**
Deployed for free on Streamlit Cloud, connected directly to this GitHub repo.

---

## Tech stack

- **Python** — core language
- **Pandas & NumPy** — data cleaning and feature engineering
- **XGBoost** — machine learning models
- **Scikit-learn** — model evaluation
- **SciPy** — Poisson distribution (used in early baseline model)
- **Plotly** — interactive charts
- **Streamlit** — web dashboard
- **Git & GitHub** — version control
- **Streamlit Cloud** — free deployment

---

## Project structure

```
fifa-predictor/
│
├── data/
│   ├── results.csv                  # raw historical match data
│   ├── fifa_ranking.csv             # FIFA rankings data
│   ├── wc_matches_clean.csv         # cleaned and filtered matches
│   ├── team_stats.csv               # engineered team features
│   └── training_data.csv            # ML training dataset
│
├── notebooks/
│   ├── 01_explore_data.ipynb        # first look at the data
│   ├── 02_clean_data.ipynb          # cleaning and filtering
│   ├── 03_feature_engineering.ipynb # building team stats
│   ├── 04_predict_scores.ipynb      # baseline Poisson model
│   ├── 05_group_stage.ipynb         # group stage simulation
│   ├── 06_knockout_stage.ipynb      # knockout simulation
│   ├── 07_ml_training_data.ipynb    # building ML dataset
│   ├── 08_ml_model.ipynb            # training XGBoost models
│   └── 09_ml_tournament.ipynb       # full ML tournament simulation
│
├── src/
│   ├── outcome_model.pkl            # trained outcome classifier
│   ├── home_goals_model.pkl         # trained home goals model
│   └── away_goals_model.pkl         # trained away goals model
│
├── outputs/
│   ├── group_stage_predictions.csv  # all 72 group match predictions
│   ├── group_standings.csv          # final group standings
│   └── knockout_predictions.csv     # full knockout bracket predictions
│
├── app.py                           # Streamlit web app
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## How to run it locally

```bash
git clone https://github.com/YOUR_USERNAME/fifa-predictor
cd fifa-predictor
python -m venv env
env\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Data sources

- Match results: [International Football Results 1872–2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) — Kaggle
- FIFA Rankings: [FIFA World Ranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) — Kaggle

---

## A note on accuracy

The outcome model hits around 54–58% accuracy on test data. That might sound low, but it's actually realistic — football is genuinely unpredictable. Even the best professional models in the world rarely exceed 60%. The value of this project isn't in perfect predictions, it's in building a real end-to-end data science pipeline from raw data to a live deployed app.

---

_Built independently as a portfolio project, inspired by the DataCamp FIFA World Cup 2026 prediction competition._
