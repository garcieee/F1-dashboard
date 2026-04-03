# training/train_model3_overtake.py  (v3 — label quality fix)
# ─────────────────────────────────────────────────────────────────────────────
# Key fixes over v2:
#   1. SC label: only count DEPLOYED safety cars during the race (lap > 1),
#      excluding VSC, red flags, and pre-race messages
#   2. Overtake label: ordinal exclusion — drop the middle third of races
#      (ambiguous cases), train only on clear High vs clear Low
#   3. Diagnostic output: prints SC message samples so you can verify
#   4. laptime_std_s: explicit fallback logging so silent 0.0s are visible
#   5. Keeps all 8 features + Pipeline + StratifiedKFold from v2
#
# IMPORTANT: Delete data_overtake.csv before running if you have the old one.
# The SC labels in the old CSV are wrong and must be regenerated.
# Run:  python training/train_model3_overtake.py
# ─────────────────────────────────────────────────────────────────────────────

import os, time
import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

BASE_DIR   = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "cache")
CSV_PATH   = os.path.join(BASE_DIR, "data_overtake.csv")

os.makedirs(CACHE_PATH, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_PATH)

# ── Static circuit knowledge ──────────────────────────────────────────────────
CIRCUIT_STATIC = {
    "Sakhir":            (3, 5.412),
    "Jeddah":            (3, 6.174),
    "Melbourne":         (4, 5.278),
    "Imola":             (2, 4.909),
    "Miami":             (3, 5.412),
    "Monaco":            (1, 3.337),
    "Barcelona":         (2, 4.657),
    "Montréal":          (2, 4.361),
    "Spielberg":         (3, 4.318),
    "Silverstone":       (2, 5.891),
    "Budapest":          (1, 4.381),
    "Spa-Francorchamps": (2, 7.004),
    "Zandvoort":         (2, 4.259),
    "Monza":             (3, 5.793),
    "Singapore":         (3, 4.940),
    "Suzuka":            (1, 5.807),
    "Lusail":            (2, 5.380),
    "Austin":            (2, 5.513),
    "Mexico City":       (3, 4.304),
    "São Paulo":         (2, 4.309),
    "Las Vegas":         (2, 6.201),
    "Yas Island":        (2, 5.281),
    "Yas Marina":        (2, 5.281),
    "Baku":              (2, 6.003),
    "Shanghai":          (2, 5.451),
    "Sochi":             (3, 5.848),
    "Istanbul":          (2, 5.338),
    "Nürburgring":       (2, 5.148),
    "Mugello":           (2, 5.245),
    "Portimão":          (3, 4.653),
    "Hockenheim":        (2, 4.574),
    "Le Castellet":      (2, 5.842),
    "Bahrain":           (3, 5.412),
}


def get_circuit_static(location):
    return CIRCUIT_STATIC.get(location, (2, 5.1))


# ── Weather features ──────────────────────────────────────────────────────────
def get_weather_features(session):
    try:
        w = session.weather_data
        if w is None or w.empty:
            return 0.0, 10.0, "Dry"
        rain_frac  = float(w["Rainfall"].mean())  if "Rainfall"  in w.columns else 0.0
        wind_speed = float(w["WindSpeed"].mean()) if "WindSpeed" in w.columns else 10.0
        if rain_frac > 0.6:   label = "Heavy Rain"
        elif rain_frac > 0.3: label = "Mixed"
        elif rain_frac > 0.05: label = "Light Rain"
        elif wind_speed > 25:  label = "Overcast"
        else:                  label = "Dry"
        return round(rain_frac, 3), round(wind_speed, 2), label
    except Exception:
        return 0.0, 10.0, "Dry"


# ── Safety car — FIXED ────────────────────────────────────────────────────────
def check_safety_car(session, year, round_number):
    """
    Returns 1 ONLY if a full Safety Car (not VSC, not red flag) was deployed
    during racing laps (not formation lap, not pre-race).

    Key changes from v2:
    - Explicitly excludes "VIRTUAL SAFETY CAR" and "RED FLAG"
    - Only counts messages where Status = "DEPLOYED" (not "ENDING" or "IN THIS LAP")
    - Prints a diagnostic sample for the first 5 races so you can verify
    """
    try:
        msgs = session.race_control_messages
        if msgs is None or msgs.empty:
            return 0

        # Work with uppercase for reliable matching
        msgs = msgs.copy()
        msgs["Message_upper"] = msgs["Message"].str.upper().str.strip()

        # Exclude VSC, red flags, and informational SC messages
        # We ONLY want "SAFETY CAR DEPLOYED" — not endings, not VSC
        sc_deployed = msgs[
            msgs["Message_upper"].str.contains("SAFETY CAR", na=False) &
            ~msgs["Message_upper"].str.contains("VIRTUAL", na=False) &
            ~msgs["Message_upper"].str.contains("ENDING", na=False) &
            ~msgs["Message_upper"].str.contains("IN THIS LAP", na=False) &
            ~msgs["Message_upper"].str.contains("WITHDRAWN", na=False) &
            ~msgs["Message_upper"].str.contains("RED FLAG", na=False)
        ]

        result = 1 if len(sc_deployed) > 0 else 0

        # Diagnostic: show what messages we found (first call only)
        if round_number <= 2 and year == 2018:
            print(f"\n    [SC DIAG] {year} R{round_number} — {len(msgs)} total msgs, "
                  f"{len(sc_deployed)} SC-DEPLOYED msgs, result={result}")
            if len(sc_deployed) > 0:
                for _, row in sc_deployed.head(3).iterrows():
                    print(f"      → '{row['Message']}'")

        return result

    except Exception as e:
        return 0


# ── Lap features ──────────────────────────────────────────────────────────────
def get_lap_features(session, circuit_name, year, round_number):
    """
    Returns (laptime_std_s, n_compounds).
    Explicit logging when fallback is used so you can see how often it fires.
    """
    try:
        laps = session.laps
        if laps is None or laps.empty:
            print(f"      [LAP DIAG] {year} R{round_number} {circuit_name}: "
                  f"no laps data — using fallback (0.0, 2)")
            return 0.0, 2

        valid = laps[laps["IsAccurate"] == True].copy()
        valid = valid.dropna(subset=["LapTime", "Compound"])
        valid = valid[valid["PitOutTime"].isna() & valid["PitInTime"].isna()]

        if valid.empty:
            print(f"      [LAP DIAG] {year} R{round_number} {circuit_name}: "
                  f"0 valid laps after filter — using fallback (0.0, 2)")
            return 0.0, 2

        valid["lap_s"]  = valid["LapTime"].dt.total_seconds()
        driver_medians  = valid.groupby("Driver")["lap_s"].median()
        laptime_std     = float(driver_medians.std()) if len(driver_medians) > 1 else 0.0

        VALID_COMP = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
        n_comp = len([c for c in valid["Compound"].str.upper().unique()
                      if c in VALID_COMP])

        return round(laptime_std, 3), max(n_comp, 1)

    except Exception as e:
        print(f"      [LAP DIAG] {year} R{round_number} {circuit_name}: "
              f"exception {e} — using fallback (0.0, 2)")
        return 0.0, 2


# ── Fetch data ────────────────────────────────────────────────────────────────
# IMPORTANT: always start fresh — old CSV has incorrect SC labels
if os.path.exists(CSV_PATH):
    print(f"Found existing {CSV_PATH}")
    print("Checking if it has correct columns...")
    df_check = pd.read_csv(CSV_PATH)
    required = ["rain_frac", "wind_speed", "laptime_std_s",
                "n_compounds", "drs_zones", "circuit_length"]
    if not all(c in df_check.columns for c in required):
        print("Missing columns — deleting and starting fresh.")
        os.remove(CSV_PATH)
        records    = []
        start_year = 2018
    else:
        records      = df_check.to_dict("records")
        seasons_done = sorted(df_check["season"].unique().tolist())
        last_done    = int(max(seasons_done))
        print(f"{len(records)} rows found, seasons: {seasons_done}")
        print(f"\n*** SC rate in existing data: "
              f"{df_check['sc_deployed'].mean()*100:.1f}% ***")
        print("If SC rate > 55%, the SC labels are likely wrong.")
        print("Delete data_overtake.csv and re-run to regenerate with fixed SC logic.\n")
        start_year = last_done + 1
else:
    records    = []
    start_year = 2018
    print("No existing data — starting fresh from 2018.")

call_count = 0

for year in range(start_year, 2025):
    try:
        schedule   = fastf1.get_event_schedule(year, include_testing=False)
        call_count += 1
    except Exception as e:
        print(f"  Could not get schedule for {year}: {e}")
        continue

    for _, event in schedule.iterrows():
        round_number = event["RoundNumber"]
        circuit_name = event["Location"]

        if call_count > 0 and call_count % 50 == 0:
            print(f"  [{call_count} calls] Pausing 90s...")
            time.sleep(90)

        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=False, weather=True, messages=True)
            call_count += 1

            results = session.results
            if results is None or results.empty:
                continue

            # Overtake proxy
            pos_changes = []
            for _, row in results.iterrows():
                gp = row.get("GridPosition")
                fp = row.get("Position")
                if pd.notna(gp) and pd.notna(fp) and 1 <= gp <= 20 and 1 <= fp <= 20:
                    pos_changes.append(abs(int(gp) - int(fp)))

            if not pos_changes:
                continue

            avg_pos_change           = np.mean(pos_changes)
            sc                       = check_safety_car(session, year, round_number)
            rain_frac, wind_speed, wx = get_weather_features(session)
            laptime_std, n_compounds = get_lap_features(
                session, circuit_name, year, round_number)
            drs_zones, circuit_length = get_circuit_static(circuit_name)

            records.append({
                "season":         year,
                "circuit":        circuit_name,
                "weather":        wx,
                "rain_frac":      rain_frac,
                "wind_speed":     wind_speed,
                "laptime_std_s":  laptime_std,
                "n_compounds":    n_compounds,
                "drs_zones":      drs_zones,
                "circuit_length": circuit_length,
                "avg_pos_change": round(avg_pos_change, 3),
                "sc_deployed":    sc,
            })

            pd.DataFrame(records).to_csv(CSV_PATH, index=False)
            print(f"  {year} R{round_number:02d} {circuit_name:20s} | "
                  f"Δpos={avg_pos_change:.2f}  SC={sc}  "
                  f"rain={rain_frac:.2f}  DRS={drs_zones}  std={laptime_std:.2f}s")

        except Exception as e:
            print(f"  Skipping {year} R{round_number} ({circuit_name}): {e}")
            time.sleep(3)

    print(f"  ✓ Finished {year}")

print(f"\nFetch complete — {len(records)} total rows saved to {CSV_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TRAINING — LOGISTIC REGRESSION v3")
print("="*60)

df = pd.read_csv(CSV_PATH).dropna()
print(f"\nDataset : {len(df)} rows, {df['season'].nunique()} seasons")
print(f"Circuits: {df['circuit'].nunique()} unique")
print(f"\nSC rate : {df['sc_deployed'].mean()*100:.1f}%  "
      f"({df['sc_deployed'].sum()} races with SC out of {len(df)})")
print(f"avg_pos_change stats:\n{df['avg_pos_change'].describe().to_string()}")

# ── Overtake label: ordinal exclusion ────────────────────────────────────────
# Drop the middle third — those races are genuinely ambiguous.
# Train only on races that are clearly High or clearly Low overtaking events.
# This reduces noise at the label boundary, the single biggest source of error.
low_cut  = df["avg_pos_change"].quantile(0.33)
high_cut = df["avg_pos_change"].quantile(0.67)

df_ov = df[
    (df["avg_pos_change"] <= low_cut) |
    (df["avg_pos_change"] >= high_cut)
].copy()
df_ov["overtake_label"] = (df_ov["avg_pos_change"] >= high_cut).astype(int)

print(f"\nOrdinal exclusion: kept {len(df_ov)}/{len(df)} rows "
      f"(dropped middle third between {low_cut:.2f} and {high_cut:.2f})")
print(f"Overtake label distribution:\n{df_ov['overtake_label'].value_counts().to_string()}")

le_circuit_ov = LabelEncoder()
le_weather_ov = LabelEncoder()
df_ov["circuit_enc"] = le_circuit_ov.fit_transform(df_ov["circuit"])
df_ov["weather_enc"] = le_weather_ov.fit_transform(df_ov["weather"])

# Fit separate encoders on the FULL dataset for SC model (uses all rows)
le_circuit_sc = LabelEncoder()
le_weather_sc = LabelEncoder()
df["circuit_enc"] = le_circuit_sc.fit_transform(df["circuit"])
df["weather_enc"] = le_weather_sc.fit_transform(df["weather"])

FEATURES = [
    "circuit_enc", "weather_enc",
    "rain_frac", "wind_speed",
    "drs_zones", "circuit_length",
    "laptime_std_s", "n_compounds",
]

X_ov = df_ov[FEATURES].copy()
X_sc = df[FEATURES].copy()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── MODEL A: Overtake (trained on cleaned subset) ─────────────────────────────
print("\n── OVERTAKE MODEL ──────────────────────────────────────────")
y_ov = df_ov["overtake_label"]

pipe_ov = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
                   class_weight="balanced",
                   max_iter=1000, C=1.0,
                   random_state=42)),
])

cv_acc = cross_val_score(pipe_ov, X_ov, y_ov, cv=skf, scoring="accuracy")
cv_f1  = cross_val_score(pipe_ov, X_ov, y_ov, cv=skf, scoring="f1_macro")
cv_rec = cross_val_score(pipe_ov, X_ov, y_ov, cv=skf, scoring="recall_macro")
print(f"5-Fold CV Accuracy   : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print(f"5-Fold CV F1 Macro   : {cv_f1.mean():.3f}  ± {cv_f1.std():.3f}")
print(f"5-Fold CV Recall Mac : {cv_rec.mean():.3f}  ± {cv_rec.std():.3f}")

pipe_ov.fit(X_ov, y_ov)
print(f"\nFull-subset report (threshold=0.40):")
y_prob  = pipe_ov.predict_proba(X_ov)[:, 1]
y_tuned = (y_prob >= 0.40).astype(int)
print(classification_report(y_ov, y_tuned,
                             target_names=["Low Activity", "High Activity"]))

# ── MODEL B: Safety Car (trained on full dataset) ─────────────────────────────
print("── SAFETY CAR MODEL ────────────────────────────────────────")
y_sc = df["sc_deployed"]
print(f"SC distribution:\n{y_sc.value_counts().to_string()}")
print(f"SC rate: {y_sc.mean()*100:.1f}%  (expected: 40–55% for real F1)")

pipe_sc = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
                   class_weight="balanced",
                   max_iter=1000, C=1.0,
                   random_state=42)),
])

cv_acc_sc = cross_val_score(pipe_sc, X_sc, y_sc, cv=skf, scoring="accuracy")
cv_f1_sc  = cross_val_score(pipe_sc, X_sc, y_sc, cv=skf, scoring="f1_macro")
print(f"5-Fold CV Accuracy  : {cv_acc_sc.mean():.3f} ± {cv_acc_sc.std():.3f}")
print(f"5-Fold CV F1 Macro  : {cv_f1_sc.mean():.3f}  ± {cv_f1_sc.std():.3f}")

pipe_sc.fit(X_sc, y_sc)
print(f"\nFull-data report:")
print(classification_report(y_sc, pipe_sc.predict(X_sc),
                             target_names=["No SC", "SC Deployed"]))

# ── Summary ───────────────────────────────────────────────────────────────────
print("="*60)
print("SUMMARY")
print(f"  Overtake rows used     : {len(df_ov)} / {len(df)} (middle third excluded)")
print(f"  Overtake CV F1 Macro   : {cv_f1.mean():.3f}")
print(f"  Safety Car CV F1 Macro : {cv_f1_sc.mean():.3f}")
print(f"  SC rate in data        : {y_sc.mean()*100:.1f}%")
if y_sc.mean() > 0.55:
    print("\n  ⚠ WARNING: SC rate is still > 55%.")
    print("  This suggests the SC labels are still over-counting.")
    print("  Delete data_overtake.csv and re-run — the v3 SC filter should fix it.")
    print("  If it persists, check the [SC DIAG] messages printed during fetch.")
else:
    print("\n  ✓ SC rate looks realistic.")
print("="*60)
print("\nRun export_model3.py to save the final .pkl")