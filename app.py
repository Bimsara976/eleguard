import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from functools import wraps
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------------------------------------------------------------------
# Auth credentials
# ---------------------------------------------------------------------------
USERS = {
    "admin": "Eids@2025!",
    "analyst": "Sensor#42",
}

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Seismic notebook artifacts
SEISMIC_BINARY_MODEL_PATH    = os.path.join(MODELS_DIR, "best_binary_proxy_intrusion_model.joblib")
SEISMIC_MULTI_MODEL_PATH     = os.path.join(MODELS_DIR, "best_multiclass_vibration_model.joblib")
SEISMIC_METADATA_PATH        = os.path.join(MODELS_DIR, "model_metadata.json")

# Movebank notebook artifacts
MOVEBANK_RISK_MODEL_PATH     = os.path.join(MODELS_DIR, "intrusion_risk_model.joblib")
MOVEBANK_STATE_MODEL_PATH    = os.path.join(MODELS_DIR, "movement_state_model.joblib")
MOVEBANK_METADATA_PATH       = os.path.join(MODELS_DIR, "model_metadata.joblib")

# ---------------------------------------------------------------------------
# Load models at startup (graceful if missing)
# ---------------------------------------------------------------------------
def _load(path, loader=joblib.load):
    try:
        return loader(path)
    except Exception as e:
        app.logger.warning(f"Could not load {path}: {e}")
        return None

seismic_binary_model = _load(SEISMIC_BINARY_MODEL_PATH)
seismic_multi_model  = _load(SEISMIC_MULTI_MODEL_PATH)

with open(SEISMIC_METADATA_PATH) as f:
    seismic_meta = json.load(f)

seismic_feature_cols = seismic_meta.get("feature_columns", [
    "mean", "top_3_mean", "min", "max", "std_dev",
    "median", "q1", "q3", "skewness", "dominant_freq", "energy"
])

movebank_risk_model  = _load(MOVEBANK_RISK_MODEL_PATH)
movebank_state_model = _load(MOVEBANK_STATE_MODEL_PATH)
movebank_meta        = _load(MOVEBANK_METADATA_PATH)

MOVEBANK_SUP_FEATURES   = [
    "speed_mps", "turning_angle_rad", "step_length_m",
    "roll_speed_mean", "roll_speed_std",
    "dist_to_boundary_m", "toward_human_mps",
    "external-temperature", "hour_sin", "hour_cos",
]
MOVEBANK_STATE_FEATURES = [
    "speed_mps", "turning_angle_rad", "step_length_m",
    "roll_speed_mean", "roll_speed_std",
]

MOVEBANK_THRESHOLD = 0.5
if isinstance(movebank_meta, dict):
    MOVEBANK_THRESHOLD = float(movebank_meta.get("threshold", 0.5))

# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s
    return model.predict(X).astype(float)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if USERS.get(username) == password:
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Please try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    seismic_ready  = seismic_binary_model is not None and seismic_multi_model is not None
    movebank_ready = movebank_risk_model is not None and movebank_state_model is not None
    return render_template(
        "dashboard.html",
        username=session["username"],
        seismic_ready=seismic_ready,
        movebank_ready=movebank_ready,
        seismic_meta=seismic_meta,
        movebank_meta=movebank_meta if isinstance(movebank_meta, dict) else {},
    )


@app.route("/seismic", methods=["GET", "POST"])
@login_required
def seismic():
    result = None
    form_values = {}

    defaults = {
        "mean": 2041.29, "top_3_mean": 2052.33, "min": 2031.0, "max": 2053.0,
        "std_dev": 3.46, "median": 2041.0, "q1": 2037.0, "q3": 2044.0,
        "skewness": -0.267, "dominant_freq": 300.0, "energy": 1955327527.0
    }

    if request.method == "POST":
        try:
            form_values = {col: float(request.form[col]) for col in seismic_feature_cols}
            sample_df = pd.DataFrame([form_values])[seismic_feature_cols]

            # Multiclass prediction
            multi_label = seismic_multi_model.predict(sample_df)[0]
            multi_probs_raw = seismic_multi_model.predict_proba(sample_df)[0]
            multi_classes   = list(seismic_multi_model.named_steps["model"].classes_)
            multi_probs = {cls: round(float(p) * 100, 2)
                           for cls, p in zip(multi_classes, multi_probs_raw)}

            # Binary proxy prediction
            binary_label = int(seismic_binary_model.predict(sample_df)[0])
            binary_prob  = round(float(seismic_binary_model.predict_proba(sample_df)[0, 1]) * 100, 2)

            result = {
                "multi_label":  multi_label,
                "multi_probs":  multi_probs,
                "binary_label": binary_label,
                "binary_prob":  binary_prob,
                "binary_text":  "⚠ Movement Detected (Intrusion Proxy Positive)"
                                if binary_label == 1
                                else "✓ No Movement (Proxy Negative)",
            }
        except Exception as e:
            flash(f"Prediction error: {e}", "error")

    return render_template(
        "seismic.html",
        username=session["username"],
        result=result,
        feature_cols=seismic_feature_cols,
        defaults=defaults,
        form_values=form_values or defaults,
    )


@app.route("/movement", methods=["GET", "POST"])
@login_required
def movement():
    result = None
    form_values = {}

    defaults = {
        "speed_mps": 1.2,
        "turning_angle_rad": 0.3,
        "step_length_m": 180.0,
        "roll_speed_mean": 1.1,
        "roll_speed_std": 0.25,
        "dist_to_boundary_m": 4500.0,
        "toward_human_mps": 0.1,
        "external-temperature": 26.0,
        "hour": 14,
    }

    if request.method == "POST":
        try:
            hour = int(request.form.get("hour", 14))
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)

            form_values = {
                "speed_mps":           float(request.form["speed_mps"]),
                "turning_angle_rad":   float(request.form["turning_angle_rad"]),
                "step_length_m":       float(request.form["step_length_m"]),
                "roll_speed_mean":     float(request.form["roll_speed_mean"]),
                "roll_speed_std":      float(request.form["roll_speed_std"]),
                "dist_to_boundary_m":  float(request.form["dist_to_boundary_m"]),
                "toward_human_mps":    float(request.form["toward_human_mps"]),
                "external-temperature": float(request.form["external-temperature"]),
                "hour_sin":            hour_sin,
                "hour_cos":            hour_cos,
                "hour":                hour,
            }

            sup_df   = pd.DataFrame([form_values])[MOVEBANK_SUP_FEATURES]
            state_df = pd.DataFrame([form_values])[MOVEBANK_STATE_FEATURES]

            risk_prob  = float(_proba_safe(movebank_risk_model, sup_df)[0])
            risk_label = int(risk_prob >= MOVEBANK_THRESHOLD)
            move_state = int(movebank_state_model.predict(state_df.values)[0])

            result = {
                "risk_prob":   round(risk_prob * 100, 2),
                "risk_label":  risk_label,
                "risk_text":   "🚨 Intrusion Likely — Elephant may cross soon"
                               if risk_label == 1
                               else "✅ No Imminent Intrusion",
                "move_state":  move_state,
                "state_label": f"Movement State {move_state}",
                "threshold":   round(MOVEBANK_THRESHOLD * 100, 1),
            }
        except Exception as e:
            flash(f"Prediction error: {e}", "error")

    return render_template(
        "movement.html",
        username=session["username"],
        result=result,
        defaults=defaults,
        form_values=form_values or defaults,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
