# training/export_model4.py
# ─────────────────────────────────────────────────────────────────────────────
# Train final Constructor Championship model on ALL available data and export
# to models/constructor.pkl
# Run AFTER train_model4_constructor.py has finished fetching data.
# Run:  python training/export_model4.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

BASE_DIR   = os.path.dirname(__file__)
CSV_PATH   = os.path.join(BASE_DIR, "data_constructor.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(f"Seasons: {sorted(df['season'].unique().tolist())}")

# ── Feature engineering (same as training script) ────────────────────────────
df = df.sort_values(["constructor", "season"]).reset_index(drop=True)

df["rank"] = df.groupby("season")["total_points"].rank(
    ascending=False, method="min"
).astype(int)

df["prev_points"]      = df.groupby("constructor")["total_points"].shift(1)
df["prev_rank"]        = df.groupby("constructor")["rank"].shift(1)
df["prev_rounds"]      = df.groupby("constructor")["rounds_counted"].shift(1)
df["prev2_points"]     = df.groupby("constructor")["total_points"].shift(2)
df["points_trend"]     = df["prev_points"] - df["prev2_points"]
df["avg_pts_per_race"] = df["prev_points"] / df["prev_rounds"].replace(0, np.nan)

df_train = df.dropna(subset=[
    "prev_points", "prev_rank", "points_trend", "avg_pts_per_race"
]).copy()

print(f"Training on {len(df_train)} rows "
      f"(seasons {df_train['season'].min()}–{df_train['season'].max()})")

le = LabelEncoder()
df_train["constructor_enc"] = le.fit_transform(df_train["constructor"])

FEATURES = [
    "constructor_enc",
    "season",
    "prev_points",
    "prev_rank",
    "points_trend",
    "avg_pts_per_race",
    "rounds_counted",
]

X = df_train[FEATURES].values
y = df_train["total_points"].values

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge",  Ridge(alpha=10.0)),
])
pipe.fit(X, y)

# ── Build the "latest known data" lookup for inference ───────────────────────
# At prediction time (user picks a future season), app.py needs to look up
# each team's most recent prev_points, prev_rank, etc.
# We store the latest complete season's stats per constructor.
latest_season = df["season"].max()
latest = (df[df["season"] == latest_season]
          .set_index("constructor")[["total_points", "rank", "rounds_counted"]]
          .rename(columns={
              "total_points": "last_points",
              "rank":         "last_rank",
              "rounds_counted": "last_rounds",
          }))

# Also store the season before that for trend calculation
prev_season = latest_season - 1
prev_df = df[df["season"] == prev_season].set_index("constructor")[["total_points"]]
prev_df = prev_df.rename(columns={"total_points": "prev2_points"})

latest = latest.join(prev_df, how="left")
latest_dict = latest.to_dict("index")

os.makedirs(MODELS_DIR, exist_ok=True)
out_path = os.path.join(MODELS_DIR, "constructor.pkl")
joblib.dump({
    "model":           pipe,
    "le_constructor":  le,
    "features":        FEATURES,
    "latest_season":   int(latest_season),
    "latest_stats":    latest_dict,    # {team: {last_points, last_rank, ...}}
}, out_path)

print(f"\nSaved → {out_path}")
print(f"Constructors known : {list(le.classes_)}")
print(f"Latest season stats (used as inference baseline):")
for team, stats in latest_dict.items():
    print(f"  {team:<15} last_pts={stats['last_points']:.0f}  "
          f"last_rank={stats['last_rank']}  "
          f"last_rounds={stats['last_rounds']}")