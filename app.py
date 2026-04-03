from flask import Flask, render_template, request
import random
import joblib
import numpy as np
import os
import datetime

app = Flask(__name__)

# ── Seasons ───────────────────────────────────────────────────────────────────
_current_year = datetime.datetime.now().year
SEASONS = [str(_current_year), str(_current_year + 1)]

# ── Data for dropdowns ────────────────────────────────────────────────────────

DRIVERS = [
    "Alexander Albon",
    "Antonio Giovinazzi",
    "Brendon Hartley",
    "Carlos Sainz",
    "Charles Leclerc",
    "Daniel Ricciardo",
    "Daniil Kvyat",
    "Esteban Ocon",
    "Fernando Alonso",
    "George Russell",
    "Guanyu Zhou",
    "Jack Aitken",
    "Kevin Magnussen",
    "Kimi Räikkönen",
    "Lance Stroll",
    "Lando Norris",
    "Lewis Hamilton",
    "Marcus Ericsson",
    "Max Verstappen",
    "Mick Schumacher",
    "Nicholas Latifi",
    "Nico Hulkenberg",
    "Nikita Mazepin",
    "Nyck De Vries",
    "Pierre Gasly",
    "Pietro Fittipaldi",
    "Robert Kubica",
    "Romain Grosjean",
    "Sebastian Vettel",
    "Sergey Sirotkin",
    "Sergio Perez",
    "Stoffel Vandoorne",
    "Valtteri Bottas",
    "Yuki Tsunoda",
]

CIRCUITS = [
    "Austin",
    "Baku",
    "Barcelona",
    "Budapest",
    "Hockenheim",
    "Imola",
    "Istanbul",
    "Jeddah",
    "Le Castellet",
    "Lusail",
    "Melbourne",
    "Mexico City",
    "Miami",
    "Monaco",
    "Montréal",
    "Monza",
    "Mugello",
    "Nürburgring",
    "Portimão",
    "Sakhir",
    "Shanghai",
    "Silverstone",
    "Singapore",
    "Sochi",
    "Spa-Francorchamps",
    "Spielberg",
    "Suzuka",
    "São Paulo",
    "Yas Island",
    "Yas Marina",
    "Zandvoort",
]

TIRE_COMPOUNDS     = ["Soft", "Medium", "Hard", "Intermediate", "Wet"]
WEATHER_CONDITIONS = ["Dry", "Overcast", "Light Rain", "Heavy Rain", "Mixed"]
CONSTRUCTORS = [
    "Red Bull", "Mercedes", "Ferrari", "McLaren",
    "Aston Martin", "Alpine", "Williams", "AlphaTauri",
    "Alfa Romeo", "Haas",
]

# ── Load Model 1: Finishing Position ─────────────────────────────────────────

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

# ── Load Model 2: Lap Time ────────────────────────────────────────────────────

LAPTIME_MODEL    = None
LAPTIME_LE_DRV   = None
LAPTIME_LE_CIR   = None
LAPTIME_LE_COMP  = None

try:
    _laptime         = joblib.load("models/laptime.pkl")
    LAPTIME_MODEL    = _laptime["model"]
    LAPTIME_LE_DRV   = _laptime["le_driver"]
    LAPTIME_LE_CIR   = _laptime["le_circuit"]
    LAPTIME_LE_COMP  = _laptime["le_compound"]
    print(f"✓ Lap time model loaded  | "
          f"drivers={len(LAPTIME_LE_DRV.classes_)}  "
          f"circuits={len(LAPTIME_LE_CIR.classes_)}  "
          f"compounds={list(LAPTIME_LE_COMP.classes_)}")
except Exception as e:
    print(f"✗ Lap time model not loaded: {e}")

# ── Prediction helpers — Model 1 ──────────────────────────────────────────────

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
    """P1–P20 grid sweep for a single driver."""
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
    standings = []
    for final_pos, entry in enumerate(raw, start=1):
        standings.append({
            "position": final_pos, "driver": entry["driver"],
            "grid": entry["grid"], "selected": entry["selected"],
        })
    return standings


# ── Prediction helpers — Model 2 ──────────────────────────────────────────────

def _ms_to_laptime(ms: float) -> str:
    """Convert milliseconds → 'M:SS.mmm' string."""
    ms   = max(0, ms)
    mins = int(ms // 60_000)
    secs = (ms % 60_000) / 1000
    return f"{mins}:{secs:06.3f}"


def predict_lap_time(driver: str, circuit: str, compound: str,
                     lap_number: int, season: int):
    """
    Returns (lap_time_str, lap_time_ms, model_live).
    Falls back to dummy if model missing or inputs are unseen.
    """
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


def predict_lap_sweep(driver: str, circuit: str, compound: str,
                      season: int, total_laps: int = 57):
    """
    Predict lap times for laps 1..total_laps.
    Returns list of {lap, lap_time_ms, lap_time_str}.
    """
    results = []
    for lap in range(1, total_laps + 1):
        lt_str, lt_ms, _ = predict_lap_time(driver, circuit, compound, lap, season)
        results.append({"lap": lap, "lap_time_ms": lt_ms, "lap_time_str": lt_str})
    return results


# ── Dummy helpers (fallback / Models 3–4) ────────────────────────────────────

def _dummy_lap_time(driver, compound, lap_number):
    base_ms = {
        "Soft": 88000, "Medium": 89500, "Hard": 91000,
        "Intermediate": 95000, "Wet": 102000,
    }.get(compound, 89000)
    variance    = random.randint(-800, 800)
    lap_penalty = int(lap_number) * random.randint(20, 60)
    total_ms    = base_ms + variance + lap_penalty
    return _ms_to_laptime(total_ms), total_ms


def dummy_overtake_safety(circuit, weather):
    seed     = hash(f"{circuit}{weather}") % 100
    overtake = max(10, min(90, seed + random.randint(-10, 10)))
    safety   = max(5,  min(70, (100 - seed) // 2 + random.randint(-5, 5)))
    return overtake, safety


def dummy_constructor_standings(season):
    constructors = [
        "McLaren", "Ferrari", "Red Bull", "Mercedes",
        "Aston Martin", "Alpine", "Williams", "AlphaTauri",
        "Alfa Romeo", "Haas",
    ]
    random.seed(hash(season))
    base_pts = [600, 560, 530, 490, 340, 220, 130, 80, 50, 30]
    variance = [random.randint(-30, 30) for _ in constructors]
    standings = sorted(
        zip(constructors, [b + v for b, v in zip(base_pts, variance)]),
        key=lambda x: -x[1]
    )
    return standings


# ── Routes ────────────────────────────────────────────────────────────────────

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
            "driver":     driver,
            "circuit":    circuit,
            "compound":   compound,
            "lap_number": lap_number,
            "season":     season,
        }

        lt_str, lt_ms, model_live = predict_lap_time(
            driver, circuit, compound, lap_number, season
        )
        sweep = predict_lap_sweep(driver, circuit, compound, season, total_laps=57)

        result = {
            "lap_time":   lt_str,
            "lap_ms":     lt_ms,
            "compound":   compound,
            "model_live": model_live,
        }

    return render_template(
        "laptime.html",
        drivers=DRIVERS,
        circuits=CIRCUITS,
        compounds=TIRE_COMPOUNDS,
        seasons=SEASONS,
        laps=list(range(1, 71)),
        result=result,
        inputs=inputs,
        sweep=sweep,
    )


@app.route("/overtake", methods=["GET", "POST"])
def overtake():
    result = None
    inputs = {}
    if request.method == "POST":
        circuit = request.form.get("circuit")
        weather = request.form.get("weather")
        inputs  = {"circuit": circuit, "weather": weather}
        ov_pct, sc_pct = dummy_overtake_safety(circuit, weather)
        result  = {"overtake_pct": ov_pct, "safety_pct": sc_pct}
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
        season    = request.form.get("season")
        inputs    = {"season": season}
        standings = dummy_constructor_standings(season)
        result    = {"season": season, "standings": standings}
    return render_template(
        "constructor.html",
        seasons=SEASONS,
        result=result, inputs=inputs,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5001)