from flask import Flask, render_template, request
import random
import joblib
import numpy as np
import os
import datetime

app = Flask(__name__)

# ── Seasons ───────────────────────────────────────────────────────────────────
_current_year = datetime.datetime.now().year
SEASONS = [str(y) for y in range(_current_year, _current_year + 2)]

# ── Dropdowns ─────────────────────────────────────────────────────────────────

DRIVERS = [
    # Current / recent grid (2023–2026)
    "Alexander Albon", "Andrea Kimi Antonelli", "Carlos Sainz",
    "Charles Leclerc", "Esteban Ocon", "Fernando Alonso",
    "Franco Colapinto", "Gabriel Bortoleto", "George Russell",
    "Isack Hadjar", "Jack Doohan", "Jack Aitken",
    "Lance Stroll", "Lando Norris", "Lewis Hamilton",
    "Liam Lawson", "Logan Sargeant", "Max Verstappen",
    "Nico Hulkenberg", "Nyck De Vries", "Oliver Bearman",
    "Oscar Piastri", "Pierre Gasly", "Sergio Perez",
    "Valtteri Bottas", "Yuki Tsunoda",
    # Historical grid (2018–2022, covered by training data)
    "Antonio Giovinazzi", "Brendon Hartley", "Daniel Ricciardo",
    "Daniil Kvyat", "Guanyu Zhou", "Kevin Magnussen",
    "Kimi Räikkönen", "Marcus Ericsson", "Mick Schumacher",
    "Nicholas Latifi", "Nikita Mazepin", "Pietro Fittipaldi",
    "Robert Kubica", "Romain Grosjean", "Sebastian Vettel",
    "Sergey Sirotkin", "Stoffel Vandoorne",
]

CIRCUITS = [
    "Austin", "Baku", "Barcelona", "Budapest", "Hockenheim",
    "Imola", "Istanbul", "Jeddah", "Le Castellet", "Lusail",
    "Melbourne", "Mexico City", "Miami", "Monaco", "Montréal",
    "Monza", "Mugello", "Nürburgring", "Portimão", "Sakhir",
    "Shanghai", "Silverstone", "Singapore", "Sochi",
    "Spa-Francorchamps", "Spielberg", "Suzuka", "São Paulo",
    "Yas Island", "Yas Marina", "Zandvoort",
]

TIRE_COMPOUNDS     = ["Soft", "Medium", "Hard", "Intermediate", "Wet"]
WEATHER_CONDITIONS = ["Dry", "Overcast", "Light Rain", "Heavy Rain", "Mixed"]
CONSTRUCTORS = [
    "Red Bull", "Mercedes", "Ferrari", "McLaren",
    "Aston Martin", "Alpine", "Williams", "AlphaTauri",
    "Alfa Romeo", "Haas",
]

# ── Model 3 static lookup tables ──────────────────────────────────────────────
# (drs_zones, circuit_length_km) — must match training script exactly
CIRCUIT_STATIC_OV = {
    "Sakhir":            (3, 5.412), "Jeddah":            (3, 6.174),
    "Melbourne":         (4, 5.278), "Imola":             (2, 4.909),
    "Miami":             (3, 5.412), "Monaco":            (1, 3.337),
    "Barcelona":         (2, 4.657), "Montréal":          (2, 4.361),
    "Spielberg":         (3, 4.318), "Silverstone":       (2, 5.891),
    "Budapest":          (1, 4.381), "Spa-Francorchamps": (2, 7.004),
    "Zandvoort":         (2, 4.259), "Monza":             (3, 5.793),
    "Singapore":         (3, 4.940), "Suzuka":            (1, 5.807),
    "Lusail":            (2, 5.380), "Austin":            (2, 5.513),
    "Mexico City":       (3, 4.304), "São Paulo":         (2, 4.309),
    "Las Vegas":         (2, 6.201), "Yas Island":        (2, 5.281),
    "Yas Marina":        (2, 5.281), "Baku":              (2, 6.003),
    "Shanghai":          (2, 5.451), "Sochi":             (3, 5.848),
    "Istanbul":          (2, 5.338), "Nürburgring":       (2, 5.148),
    "Mugello":           (2, 5.245), "Portimão":          (3, 4.653),
    "Hockenheim":        (2, 4.574), "Le Castellet":      (2, 5.842),
    "Bahrain":           (3, 5.412),
}

# Representative (rain_frac, wind_speed) per weather label for inference
WEATHER_DEFAULTS = {
    "Dry":        (0.00, 12.0),
    "Overcast":   (0.02, 28.0),
    "Light Rain": (0.10, 15.0),
    "Heavy Rain": (0.75, 20.0),
    "Mixed":      (0.35, 18.0),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# ── Model 1: Finishing Position ───────────────────────────────────────────────
FINISHING_MODEL  = None
FINISHING_LE_DRV = None
FINISHING_LE_CIR = None

try:
    _finishing       = joblib.load("models/finishing.pkl")
    FINISHING_MODEL  = _finishing["model"]
    FINISHING_LE_DRV = _finishing["le_driver"]
    FINISHING_LE_CIR = _finishing["le_circuit"]
    print("✓ Finishing model loaded")
except Exception as e:
    print(f"✗ Finishing model not loaded: {e}")

# ── Model 2: Lap Time ─────────────────────────────────────────────────────────
LAPTIME_MODEL   = None
LAPTIME_LE_DRV  = None
LAPTIME_LE_CIR  = None
LAPTIME_LE_COMP = None

try:
    _laptime        = joblib.load("models/laptime.pkl")
    LAPTIME_MODEL   = _laptime["model"]
    LAPTIME_LE_DRV  = _laptime["le_driver"]
    LAPTIME_LE_CIR  = _laptime["le_circuit"]
    LAPTIME_LE_COMP = _laptime["le_compound"]
    print(f"✓ Lap time model loaded  | "
          f"drivers={len(LAPTIME_LE_DRV.classes_)}  "
          f"circuits={len(LAPTIME_LE_CIR.classes_)}  "
          f"compounds={list(LAPTIME_LE_COMP.classes_)}")
except Exception as e:
    print(f"✗ Lap time model not loaded: {e}")

# ── Model 3: Overtake & Safety Car (v3 Pipeline) ─────────────────────────────
OVERTAKE_MODEL     = None
SC_MODEL           = None
OVERTAKE_LE_CIR_OV = None
OVERTAKE_LE_WX_OV  = None
OVERTAKE_LE_CIR_SC = None
OVERTAKE_LE_WX_SC  = None
OVERTAKE_THRESHOLD = None
OVERTAKE_PRED_THR  = 0.40

try:
    _ov                = joblib.load("models/overtake.pkl")
    OVERTAKE_MODEL     = _ov["model_overtake"]
    SC_MODEL           = _ov["model_sc"]
    OVERTAKE_LE_CIR_OV = _ov.get("le_circuit_ov", _ov["le_circuit"])
    OVERTAKE_LE_WX_OV  = _ov.get("le_weather_ov",  _ov["le_weather"])
    OVERTAKE_LE_CIR_SC = _ov.get("le_circuit_sc", _ov["le_circuit"])
    OVERTAKE_LE_WX_SC  = _ov.get("le_weather_sc",  _ov["le_weather"])
    OVERTAKE_THRESHOLD = _ov["ov_threshold"]
    OVERTAKE_PRED_THR  = _ov.get("ov_pred_threshold", 0.40)
    print(f"✓ Overtake model loaded  | "
          f"ov_circuits={len(OVERTAKE_LE_CIR_OV.classes_)}  "
          f"sc_circuits={len(OVERTAKE_LE_CIR_SC.classes_)}  "
          f"pred_thr={OVERTAKE_PRED_THR}")
except Exception as e:
    print(f"✗ Overtake model not loaded: {e}")

# ── Model 4: Constructor Championship ────────────────────────────────────────
CONSTRUCTOR_MODEL         = None
CONSTRUCTOR_LE            = None
CONSTRUCTOR_LATEST        = None
CONSTRUCTOR_LATEST_SEASON = None

try:
    _cons                    = joblib.load("models/constructor.pkl")
    CONSTRUCTOR_MODEL        = _cons["model"]
    CONSTRUCTOR_LE           = _cons["le_constructor"]
    CONSTRUCTOR_LATEST       = _cons["latest_stats"]
    CONSTRUCTOR_LATEST_SEASON = _cons["latest_season"]
    print(f"✓ Constructor model loaded | "
          f"teams={list(CONSTRUCTOR_LE.classes_)}  "
          f"latest_season={CONSTRUCTOR_LATEST_SEASON}")
except Exception as e:
    print(f"✗ Constructor model not loaded: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Model 1 ───────────────────────────────────────────────────────────────────

def predict_finishing_position(driver, circuit, season, grid_pos=10):
    if FINISHING_MODEL is None:
        seed = hash(f"{driver}{circuit}{season}") % 20
        return (seed % 10) + 1, 55.0
    try:
        driver_enc  = FINISHING_LE_DRV.transform([driver])[0]
        circuit_enc = FINISHING_LE_CIR.transform([circuit])[0]
        features    = np.array([[driver_enc, circuit_enc, int(season), int(grid_pos)]])
        position    = int(FINISHING_MODEL.predict(features)[0])
        proba       = FINISHING_MODEL.predict_proba(features)[0]
        classes     = list(FINISHING_MODEL.classes_)
        confidence  = round(float(proba[classes.index(position)]) * 100, 1) \
                      if position in classes else 50.0
        return position, confidence
    except ValueError:
        seed = hash(f"{driver}{circuit}{season}") % 20
        return (seed % 10) + 1, 50.0


def predict_position_sweep(driver, circuit, season):
    if FINISHING_MODEL is None:
        return None
    try:
        driver_enc  = FINISHING_LE_DRV.transform([driver])[0]
        circuit_enc = FINISHING_LE_CIR.transform([circuit])[0]
        sweep = []
        for grid in range(1, 21):
            features = np.array([[driver_enc, circuit_enc, int(season), grid]])
            pos = int(FINISHING_MODEL.predict(features)[0])
            sweep.append({"grid": grid, "predicted": pos})
        return sweep
    except ValueError:
        return None


RACE_GRID = [
    "Lewis Hamilton", "Max Verstappen", "Valtteri Bottas", "Sebastian Vettel",
    "Charles Leclerc", "Sergio Perez", "Carlos Sainz", "Lando Norris",
    "Daniel Ricciardo", "Fernando Alonso", "Pierre Gasly", "Esteban Ocon",
    "Lance Stroll", "Kimi Räikkönen", "Antonio Giovinazzi", "George Russell",
    "Yuki Tsunoda", "Nicholas Latifi", "Nikita Mazepin", "Mick Schumacher",
]


def predict_full_standings(selected_driver, circuit, season, selected_grid_pos):
    selected_grid_pos = int(selected_grid_pos)
    grid = list(RACE_GRID)
    if selected_driver in grid:
        grid.remove(selected_driver)
    else:
        grid = grid[:19]
    insert_idx = min(selected_grid_pos - 1, len(grid))
    grid.insert(insert_idx, selected_driver)

    raw = []
    for slot, driver in enumerate(grid, start=1):
        if FINISHING_MODEL is not None:
            try:
                driver_enc  = FINISHING_LE_DRV.transform([driver])[0]
                circuit_enc = FINISHING_LE_CIR.transform([circuit])[0]
                features    = np.array([[driver_enc, circuit_enc, int(season), slot]])
                predicted   = int(FINISHING_MODEL.predict(features)[0])
            except ValueError:
                predicted = slot
        else:
            predicted = ((hash(f"{driver}{circuit}{season}{slot}") % 20) + 1)
        raw.append({
            "driver": driver, "grid": slot,
            "predicted": predicted, "selected": driver == selected_driver,
        })

    raw.sort(key=lambda x: (x["predicted"], x["grid"]))
    return [
        {"position": pos, "driver": e["driver"],
         "grid": e["grid"], "selected": e["selected"]}
        for pos, e in enumerate(raw, start=1)
    ]


# ── Model 2 ───────────────────────────────────────────────────────────────────

def _ms_to_laptime(ms: float) -> str:
    ms   = max(0, ms)
    mins = int(ms // 60_000)
    secs = (ms % 60_000) / 1000
    return f"{mins}:{secs:06.3f}"


def predict_lap_time(driver, circuit, compound, lap_number, season):
    if LAPTIME_MODEL is None:
        lt_str, lt_ms = _dummy_lap_time(driver, compound, lap_number)
        return lt_str, lt_ms, False
    try:
        drv_enc  = LAPTIME_LE_DRV.transform([driver])[0]
        cir_enc  = LAPTIME_LE_CIR.transform([circuit])[0]
        comp_enc = LAPTIME_LE_COMP.transform([compound])[0]
    except ValueError:
        lt_str, lt_ms = _dummy_lap_time(driver, compound, lap_number)
        return lt_str, lt_ms, False
    features = np.array([[drv_enc, cir_enc, comp_enc, int(lap_number), int(season)]])
    pred_ms  = float(LAPTIME_MODEL.predict(features)[0])
    return _ms_to_laptime(pred_ms), round(pred_ms, 1), True


def predict_lap_sweep(driver, circuit, compound, season, total_laps=57):
    results = []
    for lap in range(1, total_laps + 1):
        lt_str, lt_ms, _ = predict_lap_time(driver, circuit, compound, lap, season)
        results.append({"lap": lap, "lap_time_ms": lt_ms, "lap_time_str": lt_str})
    return results


# ── Model 3 ───────────────────────────────────────────────────────────────────

def predict_overtake_safety(circuit, weather):
    """
    Returns (overtake_pct, sc_pct, model_live).
    Uses separate encoders for overtake (subset) vs SC (full dataset).
    """
    if OVERTAKE_MODEL is None or SC_MODEL is None:
        return _dummy_overtake_safety(circuit, weather) + (False,)

    rain_frac, wind_speed  = WEATHER_DEFAULTS.get(weather, (0.0, 12.0))
    drs_zones, circuit_len = CIRCUIT_STATIC_OV.get(circuit, (2, 5.1))

    try:
        cir_enc_ov  = OVERTAKE_LE_CIR_OV.transform([circuit])[0]
        wx_enc_ov   = OVERTAKE_LE_WX_OV.transform([weather])[0]
        features_ov = np.array([[
            cir_enc_ov, wx_enc_ov, rain_frac, wind_speed,
            drs_zones, circuit_len, 0.0, 2,
        ]])
        overtake_pct = round(float(OVERTAKE_MODEL.predict_proba(features_ov)[0][1]) * 100)
    except ValueError:
        overtake_pct = _dummy_overtake_safety(circuit, weather)[0]

    try:
        cir_enc_sc  = OVERTAKE_LE_CIR_SC.transform([circuit])[0]
        wx_enc_sc   = OVERTAKE_LE_WX_SC.transform([weather])[0]
        features_sc = np.array([[
            cir_enc_sc, wx_enc_sc, rain_frac, wind_speed,
            drs_zones, circuit_len, 0.0, 2,
        ]])
        sc_pct = round(float(SC_MODEL.predict_proba(features_sc)[0][1]) * 100)
    except ValueError:
        sc_pct = _dummy_overtake_safety(circuit, weather)[1]

    return overtake_pct, sc_pct, True


# ── Model 4 ───────────────────────────────────────────────────────────────────

def predict_constructor_standings(season: int):
    """
    Returns (standings_list, model_live).

    standings_list = [{"team": str, "points": int, "position": int}, ...]
                     sorted P1 → P10.

    Feature vector (must match FEATURES order in export_model4.py):
      constructor_enc, season, prev_points, prev_rank,
      points_trend, avg_pts_per_race, rounds_counted
    """
    if CONSTRUCTOR_MODEL is None or CONSTRUCTOR_LE is None:
        return _dummy_constructor_standings_v2(season), False

    season = int(season)
    predictions = []

    for team in CONSTRUCTORS:
        try:
            team_enc = CONSTRUCTOR_LE.transform([team])[0]
        except ValueError:
            predictions.append({"team": team, "points": 50})
            continue

        stats            = CONSTRUCTOR_LATEST.get(team, {})
        prev_points      = float(stats.get("last_points",  100))
        prev_rank        = float(stats.get("last_rank",    5))
        last_rounds      = float(stats.get("last_rounds",  22))
        prev2_points     = float(stats.get("prev2_points", prev_points))
        points_trend     = prev_points - prev2_points
        avg_pts_per_race = prev_points / last_rounds if last_rounds > 0 else 5.0
        projected_rounds = 24.0 if season >= 2025 else 22.0

        features = np.array([[
            team_enc,
            season,
            prev_points,
            prev_rank,
            points_trend,
            avg_pts_per_race,
            projected_rounds,
        ]])

        pred_pts = float(CONSTRUCTOR_MODEL.predict(features)[0])
        pred_pts = max(0, round(pred_pts))
        predictions.append({"team": team, "points": pred_pts})

    predictions.sort(key=lambda x: -x["points"])
    return [
        {"team": p["team"], "points": p["points"], "position": i + 1}
        for i, p in enumerate(predictions)
    ], True


# ── Dummy fallbacks ───────────────────────────────────────────────────────────

def _dummy_lap_time(driver, compound, lap_number):
    base_ms = {
        "Soft": 88000, "Medium": 89500, "Hard": 91000,
        "Intermediate": 95000, "Wet": 102000,
    }.get(compound, 89000)
    variance    = random.randint(-800, 800)
    lap_penalty = int(lap_number) * random.randint(20, 60)
    total_ms    = base_ms + variance + lap_penalty
    return _ms_to_laptime(total_ms), total_ms


def _dummy_overtake_safety(circuit, weather):
    seed     = hash(f"{circuit}{weather}") % 100
    overtake = max(10, min(90, seed + random.randint(-10, 10)))
    safety   = max(5,  min(70, (100 - seed) // 2 + random.randint(-5, 5)))
    return overtake, safety


def _dummy_constructor_standings_v2(season):
    teams = [
        "Red Bull", "Mercedes", "Ferrari", "McLaren",
        "Aston Martin", "Alpine", "Williams", "AlphaTauri",
        "Alfa Romeo", "Haas",
    ]
    random.seed(hash(str(season)))
    base_pts = [600, 560, 530, 490, 340, 220, 130, 80, 50, 30]
    variance = [random.randint(-30, 30) for _ in teams]
    combined = sorted(
        [{"team": t, "points": b + v}
         for t, b, v in zip(teams, base_pts, variance)],
        key=lambda x: -x["points"]
    )
    return [dict(d, position=i + 1) for i, d in enumerate(combined)]


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/finishing", methods=["GET", "POST"])
def finishing():
    result = None
    inputs = {}
    if request.method == "POST":
        driver   = request.form.get("driver")
        circuit  = request.form.get("circuit")
        season   = request.form.get("season")
        grid_pos = request.form.get("grid_pos", 10)
        inputs   = {"driver": driver, "circuit": circuit,
                    "season": season, "grid_pos": grid_pos}
        position, confidence = predict_finishing_position(
            driver, circuit, season, grid_pos
        )
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(position, "th")
        result = {
            "position":   position,
            "label":      f"P{position}",
            "suffix":     suffix,
            "confidence": confidence,
            "model_live": FINISHING_MODEL is not None,
            "sweep":      predict_position_sweep(driver, circuit, season),
            "standings":  predict_full_standings(driver, circuit, season, grid_pos),
        }
    return render_template(
        "finishing.html",
        drivers=DRIVERS, circuits=CIRCUITS, seasons=SEASONS,
        result=result, inputs=inputs,
    )


@app.route("/laptime", methods=["GET", "POST"])
def laptime():
    result = None
    inputs = {}
    sweep  = []
    if request.method == "POST":
        driver     = request.form.get("driver")
        circuit    = request.form.get("circuit")
        compound   = request.form.get("compound")
        lap_number = int(request.form.get("lap_number", 1))
        season     = int(request.form.get("season", _current_year))
        inputs = {
            "driver": driver, "circuit": circuit, "compound": compound,
            "lap_number": lap_number, "season": season,
        }
        lt_str, lt_ms, model_live = predict_lap_time(
            driver, circuit, compound, lap_number, season
        )
        sweep  = predict_lap_sweep(driver, circuit, compound, season, total_laps=57)
        result = {
            "lap_time":   lt_str,
            "lap_ms":     lt_ms,
            "compound":   compound,
            "model_live": model_live,
        }
    return render_template(
        "laptime.html",
        drivers=DRIVERS, circuits=CIRCUITS, compounds=TIRE_COMPOUNDS,
        seasons=SEASONS, laps=list(range(1, 71)),
        result=result, inputs=inputs, sweep=sweep,
    )


@app.route("/overtake", methods=["GET", "POST"])
def overtake():
    result = None
    inputs = {}
    if request.method == "POST":
        circuit = request.form.get("circuit")
        weather = request.form.get("weather")
        inputs  = {"circuit": circuit, "weather": weather}
        ov_pct, sc_pct, model_live = predict_overtake_safety(circuit, weather)
        result  = {
            "overtake_pct": ov_pct,
            "safety_pct":   sc_pct,
            "model_live":   model_live,
        }
    return render_template(
        "overtake.html",
        circuits=CIRCUITS, weathers=WEATHER_CONDITIONS,
        result=result, inputs=inputs,
    )


@app.route("/constructor", methods=["GET", "POST"])
def constructor():
    result = None
    inputs = {}
    if request.method == "POST":
        season  = request.form.get("season")
        inputs  = {"season": season}
        standings, model_live = predict_constructor_standings(int(season))
        result  = {
            "season":     season,
            "standings":  standings,   # list of {team, points, position}
            "model_live": model_live,
        }
    return render_template(
        "constructor.html",
        seasons=SEASONS,
        result=result, inputs=inputs,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=5001)