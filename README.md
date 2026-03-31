# F1-dashboard
F1 Dashboard is a machine learning-based prediction system for Formula 1 racing. The system allows users to select race inputs such as driver, circuit, tire compound, and lap number, then uses trained machine learning models to generate predictions about race outcomes and performance.

## Folder Structure

```
f1dashboard/
├── app.py                  ← Flask app + dummy prediction logic
├── requirements.txt
├── models/                 ← Drop your trained .pkl files here
│   ├── finishing.pkl
│   ├── laptime.pkl
│   ├── overtake.pkl
│   └── constructor.pkl
├── templates/
│   ├── base.html           ← Shared layout, navbar, Tailwind CDN
│   ├── index.html          ← Home / dashboard landing
│   ├── finishing.html      ← Finishing position predictor
│   ├── laptime.html        ← Lap time estimator
│   ├── overtake.html       ← Overtake & safety car probability
│   └── constructor.html    ← Constructor championship projector
└── static/
    ├── css/style.css
    └── js/main.js
```

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## Swapping in Real Models

The dummy logic lives in `app.py` inside four helper functions:
- `dummy_finishing_position()` → replace with `joblib.load('models/finishing.pkl').predict()`
- `dummy_lap_time()`          → replace with `joblib.load('models/laptime.pkl').predict()`
- `dummy_overtake_safety()`   → replace with `joblib.load('models/overtake.pkl').predict_proba()`
- `dummy_constructor_standings()` → replace with `joblib.load('models/constructor.pkl').predict()`

Make sure your input features are encoded to match what the model was trained on.