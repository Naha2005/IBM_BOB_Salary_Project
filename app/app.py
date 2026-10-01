"""
app.py
------
Flask web application for the Salary Prediction project.
Run from the project root:

    python app/app.py

Then open: http://127.0.0.1:5000
"""

import sys
import os
import json

# Allow imports from src/ regardless of where the script is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify, render_template, send_from_directory
from predict import predict_salary

# ── Flask setup ────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
REPORTS_DIR  = os.path.join(BASE_DIR, 'reports')
METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'metrics.json')

app = Flask(__name__, template_folder=TEMPLATE_DIR)


def _load_metrics() -> dict:
    """Load metrics.json once at startup (read-only, no retraining)."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {
            "r2": 0.9684, "mae": 3830.88, "rmse": 4837.65,
            "intercept": 25613.67, "slope": 9550.03,
            "n_train": 22, "n_test": 6, "n_total": 28
        }


METRICS = _load_metrics()


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html", metrics=METRICS)


@app.route("/reports/<path:filename>")
def reports(filename):
    """Serve static files from the reports/ directory (e.g. regression_plot.png)."""
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Accepts JSON:  { "years": <float> }
    Returns JSON:  { "salary": <float>, "formatted": "$xx,xxx.xx" }
                or { "error": "<message>" }
    """
    data = request.get_json(silent=True) or {}
    raw  = data.get("years", "")

    # ── Validation ────────────────────────────────────────────────────────
    if raw == "" or raw is None:
        return jsonify({"error": "Please enter your years of experience."}), 400

    try:
        years = float(raw)
    except (ValueError, TypeError):
        return jsonify({"error": "Please enter a valid number."}), 400

    if years < 0:
        return jsonify({"error": "Years of experience cannot be negative."}), 400

    # ── Prediction ────────────────────────────────────────────────────────
    try:
        salary = predict_salary(years)
        return jsonify({
            "salary":    salary,
            "formatted": f"${salary:,.2f}",
            "years":     years
        })
    except FileNotFoundError:
        return jsonify({"error": "Model file not found. Please contact the administrator."}), 500
    except Exception:
        return jsonify({"error": "An unexpected error occurred. Please try again."}), 500


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Salary Prediction web app...")
    print("Open your browser at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
