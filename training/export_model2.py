# training/export_model2.py
# ─────────────────────────────────────────────────────────────────────────────
# Train the final Lap Time model on ALL available data and export to models/laptime.pkl
# Run AFTER train_model2_laptime.py has finished fetching data.
# Run:  python training/export_model2.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import joblib

BASE_DIR   = os.path.dirname(__file__)
CSV_PATH   = os.path.join(BASE_DIR, "data_laptime.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

df = pd.read_csv(CSV_PATH).dropna()
print(f"Training on {len(df)} rows (full dataset, no test split)")

le_driver   = LabelEncoder()
le_circuit  = LabelEncoder()
le_compound = LabelEncoder()

df["driver_enc"]   = le_driver.fit_transform(df["driver"])
df["circuit_enc"]  = le_circuit.fit_transform(df["circuit"])
df["compound_enc"] = le_compound.fit_transform(df["compound"])

X = df[["driver_enc", "circuit_enc", "compound_enc", "lap_number", "season"]]
y = df["lap_time_ms"]

model = Ridge(alpha=1.0)
model.fit(X, y)

os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "laptime.pkl")
joblib.dump({
    "model":      model,
    "le_driver":  le_driver,
    "le_circuit": le_circuit,
    "le_compound": le_compound,
}, out_path)

print(f"\nSaved → {out_path}")
print(f"Drivers  : {list(le_driver.classes_)}")
print(f"Circuits : {list(le_circuit.classes_)}")
print(f"Compounds: {list(le_compound.classes_)}")