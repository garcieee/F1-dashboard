from flask import Flask, render_template, request
import random

app = Flask(__name__)

# ── Data for dropdowns ────────────────────────────────────────────────────────

DRIVERS = [
    "Max Verstappen", "Sergio Perez", "Charles Leclerc", "Carlos Sainz",
    "Lewis Hamilton", "George Russell", "Lando Norris", "Oscar Piastri",
    "Fernando Alonso", "Lance Stroll", "Esteban Ocon", "Pierre Gasly",
    "Valtteri Bottas", "Zhou Guanyu", "Kevin Magnussen", "Nico Hulkenberg",
    "Yuki Tsunoda", "Daniel Ricciardo", "Alexander Albon", "Logan Sargeant",
]

CIRCUITS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy (Monza)", "Azerbaijan", "Singapore", "United States (COTA)",
    "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

TIRE_COMPOUNDS = ["Soft", "Medium", "Hard", "Intermediate", "Wet"]

SEASONS = [str(y) for y in range(2018, 2026)]

WEATHER_CONDITIONS = ["Dry", "Overcast", "Light Rain", "Heavy Rain", "Mixed"]

CONSTRUCTORS = [
    "Red Bull", "Mercedes", "Ferrari", "McLaren",
    "Aston Martin", "Alpine", "Williams", "AlphaTauri",
    "Alfa Romeo", "Haas",
]

# ── Dummy prediction helpers ──────────────────────────────────────────────────

def dummy_finishing_position(driver, circuit, season):
    """Return a fake finishing position and confidence."""
    seed = hash(f"{driver}{circuit}{season}") % 20
    position = (seed % 10) + 1
    confidence = round(random.uniform(62, 94), 1)
    return position, confidence


def dummy_lap_time(driver, compound, lap_number):
    """Return a fake lap time in mm:ss.mmm format."""
    base_ms = {
        "Soft": 88000, "Medium": 89500, "Hard": 91000,
        "Intermediate": 95000, "Wet": 102000,
    }.get(compound, 89000)
    variance = random.randint(-800, 800)
    lap_penalty = int(lap_number) * random.randint(20, 60)
    total_ms = base_ms + variance + lap_penalty
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) / 1000
    return f"{minutes}:{seconds:06.3f}", total_ms


def dummy_overtake_safety(circuit, weather):
    """Return overtake% and safety car% as integers."""
    seed = hash(f"{circuit}{weather}") % 100
    overtake = max(10, min(90, seed + random.randint(-10, 10)))
    safety = max(5, min(70, (100 - seed) // 2 + random.randint(-5, 5)))
    return overtake, safety


def dummy_constructor_standings(season):
    """Return a fake ranked list of (constructor, points)."""
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
        driver  = request.form.get("driver")
        circuit = request.form.get("circuit")
        season  = request.form.get("season")
        inputs  = {"driver": driver, "circuit": circuit, "season": season}
        position, confidence = dummy_finishing_position(driver, circuit, season)
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(position, "th")
        result = {
            "position": position,
            "label":    f"P{position}",
            "suffix":   suffix,
            "confidence": confidence,
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
    if request.method == "POST":
        driver   = request.form.get("driver")
        compound = request.form.get("compound")
        lap      = request.form.get("lap_number")
        inputs   = {"driver": driver, "compound": compound, "lap_number": lap}
        lap_str, lap_ms = dummy_lap_time(driver, compound, lap)
        result = {
            "lap_time": lap_str,
            "lap_ms":   lap_ms,
            "compound": compound,
        }
    return render_template(
        "laptime.html",
        drivers=DRIVERS, compounds=TIRE_COMPOUNDS,
        laps=list(range(1, 71)),
        result=result, inputs=inputs,
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
        result = {
            "overtake_pct": ov_pct,
            "safety_pct":   sc_pct,
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
        season = request.form.get("season")
        inputs = {"season": season}
        standings = dummy_constructor_standings(season)
        result = {"season": season, "standings": standings}
    return render_template(
        "constructor.html",
        seasons=SEASONS,
        result=result, inputs=inputs,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True)