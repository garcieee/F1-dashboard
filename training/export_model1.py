# training/export_model1.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

BASE_DIR   = os.path.dirname(__file__)
CSV_PATH   = os.path.join(BASE_DIR, "data_finishing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

df = pd.read_csv(CSV_PATH).dropna()

le_driver  = LabelEncoder()
le_circuit = LabelEncoder()
df["driver_enc"]  = le_driver.fit_transform(df["driver"])
df["circuit_enc"] = le_circuit.fit_transform(df["circuit"])

X = df[["driver_enc", "circuit_enc", "season", "grid_position"]]
y = df["finishing_position"]

model = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    min_samples_leaf=3, random_state=42, n_jobs=-1
)
model.fit(X, y)  # train on ALL data this time, no test split needed for export

os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "finishing.pkl")
joblib.dump({
    "model":      model,
    "le_driver":  le_driver,
    "le_circuit": le_circuit,
}, out_path, compress=3)

print(f"Saved to {out_path}")
print(f"Drivers known to model: {list(le_driver.classes_)}")
print(f"Circuits known to model: {list(le_circuit.classes_)}")