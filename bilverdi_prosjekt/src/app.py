"""Flask-app: tar imot bildata og returnerer prisestimat fra ML-modellen."""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
app = Flask(__name__, template_folder=str(ROOT / "templates"))
model = joblib.load(ROOT / "models/model.joblib")
meta = json.loads((ROOT / "models/meta.json").read_text())


def predict(form: dict) -> dict:
    d = meta["defaults"]
    liters = float(form.get("liters") or d["liters"])
    hp = form.get("hp")
    gears = form.get("gears")
    row = pd.DataFrame([{
        "brand": form["brand"], "model": form["model"],
        "model_year": int(form["model_year"]), "milage": float(form["milage"]),
        "fuel_type": form.get("fuel_type", "Gasoline"),
        "engine": f"{hp + 'HP ' if hp else ''}{liters}L {form.get('cylinders') or int(d['cylinders'])} Cylinder Engine",
        "transmission": f"{gears + '-Speed ' if gears else ''}{ {'manual': 'M/T', 'dual_clutch': 'Dual Shift Mode', 'cvt': 'CVT', 'automatic': 'A/T'}.get(form.get('trans_type'), 'A/T') }",
        "ext_col": form.get("ext_col", "black"), "int_col": form.get("int_col", "black"),
        "accident": "At least 1 accident or damage reported" if form.get("accident") == "yes" else "None reported",
        "clean_title": "Yes" if form.get("clean_title", "yes") == "yes" else None,
    }])
    p = float(model.predict(build_features(row))[0])
    return {"estimate": round(p, -1), "low": round(p * np.exp(meta["q10"]), -1),
            "high": round(p * np.exp(meta["q90"]), -1)}


@app.get("/")
def index():
    return render_template("index.html", meta=meta, year=2025)


@app.post("/predict")
def predict_route():
    try:
        return jsonify(predict(request.get_json(force=True)))
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Ugyldig input: {e}"), 400


@app.get("/health")
def health():
    return jsonify(status="ok", model=meta["model"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
