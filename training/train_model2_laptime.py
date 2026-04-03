# training/train_model2_laptime.py
# ─────────────────────────────────────────────────────────────────────────────
# Fetch FastF1 lap data (2018–2024) and train a Linear Regression lap-time model.
# Saves data incrementally so it is safe to interrupt and resume.
# Run:  python training/train_model2_laptime.py
# ─────────────────────────────────────────────────────────────────────────────

import os, time
import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR   = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "cache")
CSV_PATH   = os.path.join(BASE_DIR, "data_laptime.csv")

os.makedirs(CACHE_PATH, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_PATH)

# ── Resume from existing CSV if present ──────────────────────────────────────
if os.path.exists(CSV_PATH):
    df_existing = pd.read_csv(CSV_PATH)
    records     = df_existing.to_dict("records")
    seasons_done = sorted(df_existing["season"].unique().tolist())
    last_done    = int(max(seasons_done)) if seasons_done else 2017
    print(f"Found existing CSV — {len(records)} rows, seasons: {seasons_done}")
    start_year = last_done + 1
    print(f"Resuming from {start_year}...")
else:
    records    = []
    start_year = 2018
    print("No existing data — starting fresh from 2018.")

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
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

        if call_count > 0 and call_count % 50 == 0:
            print(f"  [{call_count} calls] Pausing 90s to respect rate limits...")
            time.sleep(90)

        try:
            session = fastf1.get_session(year, round_number, "R")
            # We need laps — no telemetry needed (saves bandwidth + time)
            session.load(telemetry=False, weather=False, messages=False)
            call_count += 1

            laps = session.laps

            # Keep only accurate laps with valid compound data
            laps = laps[laps["IsAccurate"] == True].copy()
            laps = laps.dropna(subset=["LapTime", "Compound", "LapNumber", "Driver"])
            laps = laps[laps["Compound"].str.upper().isin(VALID_COMPOUNDS)]

            # Convert LapTime (timedelta) → milliseconds
            laps["lap_time_ms"] = laps["LapTime"].dt.total_seconds() * 1000

            # Filter out pit laps, formation laps, obvious outliers
            laps = laps[laps["PitOutTime"].isna() & laps["PitInTime"].isna()]
            laps = laps[(laps["lap_time_ms"] > 60_000) & (laps["lap_time_ms"] < 180_000)]

            # Map driver abbreviation → FullName via session results
            abbr_to_name = {}
            try:
                for _, r in session.results.iterrows():
                    abbr_to_name[r["Abbreviation"]] = r["FullName"]
            except Exception:
                pass

            for _, row in laps.iterrows():
                full_name = abbr_to_name.get(row["Driver"], row["Driver"])
                records.append({
                    "season":      year,
                    "circuit":     circuit_name,
                    "driver":      full_name,
                    "compound":    row["Compound"].title(),   # Soft / Medium / Hard …
                    "lap_number":  int(row["LapNumber"]),
                    "lap_time_ms": round(row["lap_time_ms"], 1),
                })

            pd.DataFrame(records).to_csv(CSV_PATH, index=False)
            print(f"  {year} R{round_number:02d} {circuit_name:20s} — {len(records)} rows total")

        except Exception as e:
            print(f"  Skipping {year} Round {round_number} ({circuit_name}): {e}")
            time.sleep(3)

    print(f"  ✓ Finished {year}")

print(f"\nFetch complete — {len(records)} total rows saved to {CSV_PATH}")

# ── Train & Evaluate ──────────────────────────────────────────────────────────
print("\nTraining Linear Regression (Ridge) model...")

df = pd.read_csv(CSV_PATH).dropna()
print(f"Training on {len(df)} rows")
print(f"Compounds:  {sorted(df['compound'].unique())}")
print(f"Drivers:    {df['driver'].nunique()} unique")
print(f"Circuits:   {df['circuit'].nunique()} unique")

le_driver   = LabelEncoder()
le_circuit  = LabelEncoder()
le_compound = LabelEncoder()

df["driver_enc"]   = le_driver.fit_transform(df["driver"])
df["circuit_enc"]  = le_circuit.fit_transform(df["circuit"])
df["compound_enc"] = le_compound.fit_transform(df["compound"])

FEATURES = ["driver_enc", "circuit_enc", "compound_enc", "lap_number", "season"]
X = df[FEATURES]
y = df["lap_time_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae_ms = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n── Evaluation ────────────────────────────")
print(f"  MAE : {mae_ms:.0f} ms  ({mae_ms/1000:.3f} s)")
print(f"  R²  : {r2:.4f}")
print(f"──────────────────────────────────────────")
print("\nRun export_model2.py to save the final model trained on ALL data.")