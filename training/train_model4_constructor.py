# training/train_model4_constructor.py
# ─────────────────────────────────────────────────────────────────────────────
# Fetch FastF1 constructor points data (2018 → latest available 2025/2026)
# and train a Ridge Regression model to project final season points per
# constructor.
#
# DATA STRUCTURE:  one row = one constructor × one season
# TARGET:          total_points  (final championship points for that season)
#
# FEATURES:
#   constructor_enc    — label-encoded team identity
#   season             — calendar year (captures regulation eras)
#   prev_points        — points scored the prior season  ← strongest predictor
#   prev_rank          — championship rank prior season
#   points_trend       — 2-year rolling delta (momentum)
#   avg_pts_per_race   — prev_points ÷ races_in_prev_season (pace normalised)
#   races_in_season    — total rounds this season (affects absolute totals)
#
# VALIDATION:  LeaveOneGroupOut where groups = seasons (walk-forward)
#              This is correct for time-structured sports data — no future
#              seasons leak into training for any fold.
#
# Run:  python training/train_model4_constructor.py
# ─────────────────────────────────────────────────────────────────────────────

import os, time, datetime
import fastf1
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR   = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "cache")
CSV_PATH   = os.path.join(BASE_DIR, "data_constructor.csv")

os.makedirs(CACHE_PATH, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_PATH)

# ── Points system (2010-present) ──────────────────────────────────────────────
POINTS_MAP = {1:25, 2:18, 3:15, 4:12, 5:10,
              6:8,  7:6,  8:4,  9:2,  10:1}
# Fastest lap bonus point (awarded since 2019 to top-10 finisher)
# FastF1 results include this in Points column so we just use that directly.

# ── Team name normalisation ───────────────────────────────────────────────────
# FastF1 team names vary across seasons and rebrands.
# Map everything to 10 canonical names used in the dashboard.
TEAM_ALIASES = {
    # Red Bull
    "Red Bull Racing":              "Red Bull",
    "Red Bull Racing Honda":        "Red Bull",
    "Red Bull Racing RBPT":         "Red Bull",
    "Oracle Red Bull Racing":       "Red Bull",
    # Ferrari
    "Scuderia Ferrari":             "Ferrari",
    "Ferrari":                      "Ferrari",
    # Mercedes
    "Mercedes":                     "Mercedes",
    "Mercedes-AMG Petronas":        "Mercedes",
    "Mercedes-AMG":                 "Mercedes",
    # McLaren
    "McLaren":                      "McLaren",
    "McLaren F1 Team":              "McLaren",
    "McLaren Mercedes":             "McLaren",
    # Aston Martin / Racing Point / Force India lineage
    "Aston Martin":                 "Aston Martin",
    "Aston Martin F1 Team":         "Aston Martin",
    "Aston Martin Aramco":          "Aston Martin",
    "Aston Martin Aramco F1 Team":  "Aston Martin",
    "Racing Point":                 "Aston Martin",
    "BWT Racing Point":             "Aston Martin",
    "Force India":                  "Aston Martin",
    # Alpine / Renault
    "Alpine":                       "Alpine",
    "Alpine F1 Team":               "Alpine",
    "BWT Alpine F1 Team":           "Alpine",
    "Renault":                      "Alpine",
    # Williams
    "Williams":                     "Williams",
    "Williams Racing":              "Williams",
    # AlphaTauri / Toro Rosso / VCARB
    "AlphaTauri":                   "AlphaTauri",
    "Scuderia AlphaTauri":          "AlphaTauri",
    "Scuderia AlphaTauri Honda":    "AlphaTauri",
    "Scuderia AlphaTauri RBPT":     "AlphaTauri",
    "RB F1 Team":                   "AlphaTauri",
    "Visa Cash App RB":             "AlphaTauri",
    "Visa Cash App RB F1 Team":     "AlphaTauri",
    "Toro Rosso":                   "AlphaTauri",
    "Scuderia Toro Rosso":          "AlphaTauri",
    # Alfa Romeo / Sauber
    "Alfa Romeo":                   "Alfa Romeo",
    "Alfa Romeo Racing":            "Alfa Romeo",
    "Alfa Romeo F1 Team":           "Alfa Romeo",
    "Alfa Romeo Racing ORLEN":      "Alfa Romeo",
    "Alfa Romeo F1 Team ORLEN":     "Alfa Romeo",
    "Sauber":                       "Alfa Romeo",
    "Stake F1 Team Kick Sauber":    "Alfa Romeo",
    "Kick Sauber":                  "Alfa Romeo",
    # Haas
    "Haas":                         "Haas",
    "Haas F1 Team":                 "Haas",
    "MoneyGram Haas F1 Team":       "Haas",
    "Uralkali Haas F1 Team":        "Haas",
}

KNOWN_CONSTRUCTORS = [
    "Red Bull", "Mercedes", "Ferrari", "McLaren",
    "Aston Martin", "Alpine", "Williams", "AlphaTauri",
    "Alfa Romeo", "Haas",
]


def normalise_team(name):
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


# ── Determine fetch range ─────────────────────────────────────────────────────
# Always try to get the most recent data available.
# current_year from datetime so 2026 is fetched when we're in 2026.
_now         = datetime.datetime.now()
FETCH_UNTIL  = _now.year          # e.g. 2026 if running in 2026
FETCH_UNTIL  = max(FETCH_UNTIL, 2025)   # floor at 2025


# ── Resume from existing CSV ──────────────────────────────────────────────────
if os.path.exists(CSV_PATH):
    df_existing  = pd.read_csv(CSV_PATH)
    records      = df_existing.to_dict("records")
    seasons_done = sorted(df_existing["season"].unique().tolist())
    last_done    = int(max(seasons_done)) if seasons_done else 2017
    print(f"Found existing CSV — {len(records)} rows")
    print(f"Seasons already collected: {seasons_done}")
    start_year = last_done + 1
    print(f"Resuming from {start_year} → {FETCH_UNTIL}...")
else:
    records    = []
    start_year = 2018
    print(f"No existing data — fetching {start_year} → {FETCH_UNTIL}")

call_count = 0

for year in range(start_year, FETCH_UNTIL + 1):
    print(f"\n── Season {year} ──────────────────────────────────────")
    try:
        schedule   = fastf1.get_event_schedule(year, include_testing=False)
        call_count += 1
    except Exception as e:
        print(f"  Could not get schedule for {year}: {e}")
        continue

    # Only process races that have already happened
    today          = pd.Timestamp(_now.date())
    past_events    = schedule[pd.to_datetime(schedule["EventDate"]) <= today]
    races_in_season = len(past_events)
    print(f"  {races_in_season} completed rounds found")

    if races_in_season == 0:
        print(f"  No completed rounds yet for {year} — skipping")
        continue

    # Accumulate points per constructor across all completed rounds
    season_pts    = {}   # constructor → cumulative points
    season_rounds = 0    # how many rounds we successfully processed

    for _, event in past_events.iterrows():
        round_number = event["RoundNumber"]
        circuit_name = event["Location"]

        if call_count > 0 and call_count % 50 == 0:
            print(f"  [{call_count} API calls] Pausing 90s to respect rate limits...")
            time.sleep(90)

        try:
            session = fastf1.get_session(year, round_number, "R")
            session.load(telemetry=False, weather=False, messages=False)
            call_count += 1

            results = session.results
            if results is None or results.empty:
                continue

            for _, row in results.iterrows():
                raw_team = row.get("TeamName", "")
                team     = normalise_team(raw_team)
                if team not in KNOWN_CONSTRUCTORS:
                    continue

                # Use FastF1 Points column directly — it includes fastest-lap bonus
                pts = float(row.get("Points", 0) or 0)
                season_pts[team] = season_pts.get(team, 0.0) + pts

            season_rounds += 1
            print(f"  {year} R{round_number:02d} {circuit_name:22s} — "
                  f"processed ({season_rounds} rounds so far)")

        except Exception as e:
            print(f"  Skipping {year} R{round_number} ({circuit_name}): {e}")
            time.sleep(3)

    if not season_pts:
        print(f"  No data collected for {year} — skipping")
        continue

    # Build one record per constructor for this season
    for team in KNOWN_CONSTRUCTORS:
        pts = season_pts.get(team, 0.0)
        records.append({
            "season":          year,
            "constructor":     team,
            "total_points":    round(pts, 1),
            "rounds_counted":  season_rounds,
        })

    pd.DataFrame(records).to_csv(CSV_PATH, index=False)
    print(f"  ✓ Season {year} saved — {season_rounds} rounds, "
          f"top team: {max(season_pts, key=season_pts.get)} "
          f"({max(season_pts.values()):.0f} pts)")

print(f"\n{'='*60}")
print(f"FETCH COMPLETE — {len(records)} total rows saved to {CSV_PATH}")
print(f"Seasons collected: {sorted(set(r['season'] for r in records))}")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("FEATURE ENGINEERING")
print(f"{'='*60}")

df = pd.read_csv(CSV_PATH)
print(f"\nRaw rows: {len(df)}")
print(df.groupby("season")["total_points"].sum().rename("total_pts_all_teams").to_string())

# Sort so lag features are computed correctly
df = df.sort_values(["constructor", "season"]).reset_index(drop=True)

# ── Compute championship rank per season ──────────────────────────────────────
df["rank"] = df.groupby("season")["total_points"].rank(
    ascending=False, method="min"
).astype(int)

# ── Lag features (per constructor) ───────────────────────────────────────────
df["prev_points"]      = df.groupby("constructor")["total_points"].shift(1)
df["prev_rank"]        = df.groupby("constructor")["rank"].shift(1)
df["prev_rounds"]      = df.groupby("constructor")["rounds_counted"].shift(1)
df["prev2_points"]     = df.groupby("constructor")["total_points"].shift(2)

# Momentum: how much did they improve/decline vs the season before that?
df["points_trend"]     = df["prev_points"] - df["prev2_points"]

# Pace normalised by calendar length (avoids inflation from sprint-race seasons)
df["avg_pts_per_race"] = df["prev_points"] / df["prev_rounds"].replace(0, np.nan)

# ── Drop first season per constructor (no lag data available) ─────────────────
df_train = df.dropna(subset=[
    "prev_points", "prev_rank", "points_trend", "avg_pts_per_race"
]).copy()

print(f"\nTraining rows (after dropping first year per constructor): {len(df_train)}")
print(f"Season range: {df_train['season'].min()} – {df_train['season'].max()}")
print(f"\nSample (sorted by season, pts desc):")
print(df_train.sort_values(["season","total_points"], ascending=[True,False])
      [["season","constructor","total_points","rank","prev_points","points_trend"]]
      .head(20).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TRAINING — RIDGE REGRESSION")
print(f"{'='*60}")

le = LabelEncoder()
df_train["constructor_enc"] = le.fit_transform(df_train["constructor"])

FEATURES = [
    "constructor_enc",  # team identity
    "season",           # regulation era
    "prev_points",      # prior season result  ← strongest signal
    "prev_rank",        # prior championship position
    "points_trend",     # momentum (2-year delta)
    "avg_pts_per_race", # pace normalised for calendar length
    "rounds_counted",   # this season's race count
]

X = df_train[FEATURES].values
y = df_train["total_points"].values
groups = df_train["season"].values   # for LeaveOneGroupOut

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=10.0)),   # alpha=10 regularises well on ~80 rows
])

# ── LeaveOneGroupOut CV (one fold = one season held out) ─────────────────────
logo    = LeaveOneGroupOut()
cv_mae  = cross_val_score(pipe, X, y, groups=groups,
                          cv=logo, scoring="neg_mean_absolute_error")
cv_r2   = cross_val_score(pipe, X, y, groups=groups,
                          cv=logo, scoring="r2")

print(f"\nLeaveOneGroupOut (season-wise) CV:")
print(f"  MAE  : {-cv_mae.mean():.1f} ± {cv_mae.std():.1f} points")
print(f"  R²   : {cv_r2.mean():.3f}  ± {cv_r2.std():.3f}")

unique_seasons = sorted(set(groups))
print(f"\nPer-season CV breakdown:")
print(f"  {'Season':<8} {'MAE':>8} {'R²':>8}")
for season, mae_val, r2_val in zip(
        unique_seasons,
        [-v for v in cv_mae],
        cv_r2):
    print(f"  {season:<8} {mae_val:>8.1f} {r2_val:>8.3f}")

# ── Full-data fit (for export) ────────────────────────────────────────────────
pipe.fit(X, y)
y_pred_full = pipe.predict(X)
print(f"\nFull-data fit MAE : {mean_absolute_error(y, y_pred_full):.1f} pts")
print(f"Full-data fit R²  : {r2_score(y, y_pred_full):.3f}")

print(f"\n{'='*60}")
print("DONE. Run export_model4.py to save the final .pkl")
print(f"{'='*60}")