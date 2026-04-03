# F1 Dashboard

A machine learning-powered web dashboard for Formula 1 race predictions. Built with Flask and trained on real FastF1 telemetry data (2018–2026).

## Status

| Feature | Status |
|---|---|
| Finishing position predictor | Live — RandomForest, 80.9% exact accuracy |
| Lap time estimator | Live — Ridge Regression (in iteration) |
| Overtake & safety car probability | Live — Logistic Regression (in iteration) |
| Constructor standings projector | Live — Ridge Regression (in iteration) |

## Folder Structure

```
F1-dashboard/
├── app.py
├── models/
│   ├── finishing.pkl        ← RandomForest (2018–2022)
│   ├── laptime.pkl          ← Ridge Regression (2018–2023)
│   ├── overtake.pkl         ← Logistic Regression (2018–2024)
│   └── constructor.pkl      ← Ridge Regression (2018–2026)
├── training/
│   ├── cache/                         ← FastF1 cache (gitignored)
│   ├── data_finishing.csv             ← (gitignored)
│   ├── data_laptime.csv               ← (gitignored)
│   ├── data_overtake.csv              ← (gitignored)
│   ├── data_constructor.csv           ← (gitignored)
│   ├── train_model1_finishing.py
│   ├── train_model2_laptime.py
│   ├── train_model3_overtake.py
│   ├── train_model4_constructor.py
│   ├── export_model1.py
│   ├── export_model2.py
│   ├── export_model3.py
│   ├── export_model4.py
│   └── model_eda.ipynb                ← EDA & evaluation for all models
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── finishing.html
│   ├── laptime.html
│   ├── overtake.html
│   └── constructor.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5001

## Models

### Finishing Position
**Algorithm:** RandomForestClassifier (200 trees, max depth 12)
**Data:** FastF1 race results 2018–2022
**Features:** driver, circuit, season, grid position
**Target:** finishing position (1–20)
**Accuracy:** 80.9% exact · MAE 1.04 positions · 92.3% within 5 places

### Lap Time
**Algorithm:** Ridge Regression (α=1.0)
**Data:** FastF1 accurate laps 2018–2023 (~105k laps)
**Features:** driver, circuit, compound, lap number, season
**Target:** lap time (ms)

### Overtake & Safety Car
**Algorithm:** Logistic Regression with StandardScaler (two separate models)
**Data:** FastF1 race results + weather 2018–2024 (149 races)
**Features:** circuit, weather, rain fraction, wind speed, DRS zones, circuit length, lap time std, compound count
**Target (overtake):** High / Low overtaking activity
**Target (safety car):** SC deployed / not deployed

### Constructor Championship
**Algorithm:** Ridge Regression with StandardScaler (α=10.0)
**Data:** FastF1 constructor points 2018–2026
**Features:** constructor, season, prior season points/rank, 2-year trend, pace per race, rounds in season
**Target:** total season points
**Validation:** LeaveOneGroupOut by season (no future data leakage)
