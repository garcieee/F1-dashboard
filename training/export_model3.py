# training/export_model3.py  (v2)
# ─────────────────────────────────────────────────────────────────────────────
# Train final Overtake & Safety Car models on ALL data and export
# to models/overtake.pkl
# Run AFTER train_model3_overtake.py has finished fetching data.
# Run:  python training/export_model3.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR   = os.path.dirname(__file__)
CSV_PATH   = os.path.join(BASE_DIR, "data_overtake.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

df = pd.read_csv(CSV_PATH).dropna()
print(f"Training on {len(df)} rows (full dataset)")

# Label
threshold = df["avg_pos_change"].quantile(0.40)
df["overtake_label"] = (df["avg_pos_change"] >= threshold).astype(int)
print(f"Overtake threshold (40th pct): {threshold:.3f}")
print(f"SC rate: {df['sc_deployed'].mean()*100:.1f}%")

le_circuit = LabelEncoder()
le_weather = LabelEncoder()
df["circuit_enc"] = le_circuit.fit_transform(df["circuit"])
df["weather_enc"] = le_weather.fit_transform(df["weather"])

FEATURES = [
    "circuit_enc", "weather_enc",
    "rain_frac", "wind_speed",
    "drs_zones", "circuit_length",
    "laptime_std_s", "n_compounds",
]
X = df[FEATURES].copy()

# Overtake model
pipe_ov = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight="balanced",
                                  max_iter=1000, C=0.5, random_state=42)),
])
pipe_ov.fit(X, df["overtake_label"])

# Safety car model
pipe_sc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight="balanced",
                                  max_iter=1000, C=0.5, random_state=42)),
])
pipe_sc.fit(X, df["sc_deployed"])

os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "overtake.pkl")
joblib.dump({
    "model_overtake":        pipe_ov,
    "model_sc":              pipe_sc,
    "le_circuit":            le_circuit,
    "le_weather":            le_weather,
    "ov_threshold":          threshold,
    "ov_pred_threshold":     0.40,      # saved so app.py can use it
    "features":              FEATURES,
}, out_path)

print(f"\nSaved → {out_path}")
print(f"Circuits : {list(le_circuit.classes_)}")
print(f"Weathers : {list(le_weather.classes_)}")
print(f"Features : {FEATURES}")