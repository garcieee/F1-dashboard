# F1 Dashboard

A machine learning-powered web dashboard for Formula 1 race predictions. Built with Flask and trained on real FastF1 telemetry data (2018–2024).

## Status

| Feature | Status |
|---|---|
| Finishing position predictor | Live — real RandomForest model |
| Lap time estimator | Dummy logic (model not yet trained) |
| Overtake & safety car probability | Dummy logic (model not yet trained) |
| Constructor standings projector | Dummy logic (model not yet trained) |

## Folder Structure

```
F1-dashboard/
├── app.py                        ← Flask app with real finishing model logic
├── models/
│   └── finishing.pkl             ← Trained RandomForest model (2018–2024)
├── training/
│   ├── cache/                    ← FastF1 cache (gitignored)
│   ├── data_finishing.csv        ← Training data fetched via FastF1 (gitignored)
│   ├── export_model1.py          ← Trains on full dataset and exports finishing.pkl
│   └── train_model1_finishing.py ← Training script with evaluation metrics
├── templates/
│   ├── base.html                 ← Shared layout, navbar, Tailwind CDN
│   ├── index.html                ← Dashboard landing page
│   ├── finishing.html            ← Finishing position predictor
│   ├── laptime.html              ← Lap time estimator
│   ├── overtake.html             ← Overtake & safety car probability
│   └── constructor.html          ← Constructor championship projector
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

## Model: Finishing Position

**Algorithm:** RandomForestClassifier (200 trees, max depth 12)
**Data:** FastF1 race results 2018–2024
**Features:** driver, circuit, season, grid position
**Target:** finishing position (1–20)

The training script fetches data round-by-round and saves after every race, so it's safe to interrupt and resume.
