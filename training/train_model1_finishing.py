import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import time

BASE_DIR   = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "cache")
CSV_PATH   = os.path.join(BASE_DIR, "data_finishing.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(CACHE_PATH, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_PATH)

# ══════════════════════════════════════════════════════
#  STEP 1 — FETCH DATA  (resume-safe, saves every round)
# ══════════════════════════════════════════════════════

if os.path.exists(CSV_PATH):
    existing = pd.read_csv(CSV_PATH)
    records  = existing.to_dict("records")
    print(f"Loaded {len(records)} existing rows from {CSV_PATH}")

    completed_seasons = existing.groupby("season")["circuit"].nunique()
    print("Rounds collected per season:")
    print(completed_seasons.to_string())

    last_done  = int(input("\nEnter the last FULLY completed season (e.g. 2019): "))
    start_year = last_done + 1
    print(f"\nResuming from {start_year}...")
else:
    records    = []
    start_year = 2018
    print("No existing data found, starting fresh from 2018.")

call_count = 0

for year in range(start_year, 2025):

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        call_count += 1
    except Exception as e:
        print(f"  Could not get schedule for {year}: {e}")
        continue

    for _, event in schedule.iterrows():
        round_number = event["RoundNumber"]
        circuit_name = event["Location"]

        # Only sleep if this round is NOT already in cache
        cache_hit = os.path.exists(
            os.path.join(CACHE_PATH, str(year))
        )
        if not cache_hit:
            time.sleep(8)

        if call_count > 0 and call_count % 50 == 0:
            print(f"  [{call_count} calls] Pausing 90s...")
            time.sleep(90)

        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=False, weather=False, messages=False)
            call_count += 1

            results = session.results
            for _, row in results.iterrows():
                if pd.isna(row["Position"]) or pd.isna(row["GridPosition"]):
                    continue
                records.append({
                    "season":             year,
                    "circuit":            circuit_name,
                    "driver":             row["FullName"],
                    "grid_position":      int(row["GridPosition"]),
                    "finishing_position": int(row["Position"]),
                })

            # Save after every single round so nothing is lost on crash
            pd.DataFrame(records).to_csv(CSV_PATH, index=False)
            print(f"  {year} R{round_number:02d} {circuit_name} — {len(records)} rows saved")

        except Exception as e:
            print(f"  Skipping {year} Round {round_number}: {e}")

    print(f"  Finished {year}")

df = pd.read_csv(CSV_PATH)
print(f"\nFetch complete. {len(df)} total rows.")

# ══════════════════════════════════════════════════════
#  STEP 2 — QUICK DATA CHECK
# ══════════════════════════════════════════════════════

df = df.dropna()
print(f"\nClean rows: {len(df)}")
print(f"Seasons:    {sorted(df['season'].unique())}")
print(f"Drivers:    {df['driver'].nunique()} unique")
print(f"Circuits:   {df['circuit'].nunique()} unique")

# ══════════════════════════════════════════════════════
#  STEP 3 — ENCODE, TRAIN, EVALUATE, SAVE
# ══════════════════════════════════════════════════════

le_driver  = LabelEncoder()
le_circuit = LabelEncoder()

df["driver_enc"]  = le_driver.fit_transform(df["driver"])
df["circuit_enc"] = le_circuit.fit_transform(df["circuit"])

X = df[["driver_enc", "circuit_enc", "season", "grid_position"]]
y = df["finishing_position"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining rows: {len(X_train)}  |  Test rows: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("Training complete.")

y_pred    = model.predict(X_test)
exact_acc = accuracy_score(y_test, y_pred)
within_3  = np.mean(np.abs(y_pred - y_test) <= 3)
print(f"Exact accuracy    : {exact_acc:.2%}")
print(f"Within 3 positions: {within_3:.2%}")

print("\nFeature importances:")
for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.4f}")

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump({
    "model":      model,
    "le_driver":  le_driver,
    "le_circuit": le_circuit,
}, os.path.join(MODELS_DIR, "finishing.pkl"))

print("\n✓ Saved → models/finishing.pkl")