"""
Eleguard — Elephant Intrusion Detection System
Flask Application
"""

import os, json, math, time, queue, threading, joblib
from collections import deque
from datetime import datetime
from functools import wraps
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response, stream_with_context
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────
USERS = {
    "admin":   "Eids@2025!",
    "analyst": "Sensor#42",
}

# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        app.logger.warning(f"Model not loaded ({path}): {e}")
        return None

seismic_binary_model = _load(os.path.join(MODELS_DIR, "best_binary_proxy_intrusion_model.joblib"))
seismic_multi_model  = _load(os.path.join(MODELS_DIR, "best_multiclass_vibration_model.joblib"))
movebank_risk_model  = _load(os.path.join(MODELS_DIR, "intrusion_risk_model.joblib"))
movebank_state_model = _load(os.path.join(MODELS_DIR, "movement_state_model.joblib"))
movebank_meta        = _load(os.path.join(MODELS_DIR, "model_metadata.joblib"))

with open(os.path.join(MODELS_DIR, "model_metadata.json")) as f:
    seismic_meta = json.load(f)

SEISMIC_FEATURES = seismic_meta.get("feature_columns", [
    "mean", "top_3_mean", "min", "max", "std_dev",
    "median", "q1", "q3", "skewness", "dominant_freq", "energy"
])
MOVEBANK_SUP_FEATURES = [
    "speed_mps", "turning_angle_rad", "step_length_m",
    "roll_speed_mean", "roll_speed_std", "dist_to_boundary_m",
    "toward_human_mps", "external-temperature", "hour_sin", "hour_cos",
]
MOVEBANK_STATE_FEATURES = [
    "speed_mps", "turning_angle_rad", "step_length_m",
    "roll_speed_mean", "roll_speed_std",
]
MOVEBANK_THRESHOLD = 0.5
if isinstance(movebank_meta, dict):
    MOVEBANK_THRESHOLD = float(movebank_meta.get("threshold", 0.5))

# ─────────────────────────────────────────────
# Real-time signal store (in-memory)
# ─────────────────────────────────────────────
_signal_lock  = threading.Lock()
_signal_store = deque(maxlen=200)   # all received signals
_sse_clients  = []                   # list of Queue objects, one per open SSE connection

NODE_REGISTRY = {
    "N01": {"x": 210, "y": 350, "battery": 86, "link": 92, "rssi": -78,  "status": "ok"},
    "N02": {"x": 360, "y": 410, "battery": 61, "link": 81, "rssi": -88,  "status": "ok"},
    "N03": {"x": 520, "y": 370, "battery": 44, "link": 68, "rssi": -96,  "status": "warn"},
    "N04": {"x": 690, "y": 410, "battery": 73, "link": 84, "rssi": -83,  "status": "ok"},
    "N05": {"x": 820, "y": 380, "battery": 29, "link": 55, "rssi": -103, "status": "warn"},
}

def _broadcast(event_type: str, payload: dict):
    msg = json.dumps({"event": event_type, "data": payload})
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    for q in dead:
        try:
            _sse_clients.remove(q)
        except ValueError:
            pass

def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _fuse(seismic: float, thermal: float, pir: bool) -> float:
    score = 0.5 * seismic + 0.5 * thermal + (8.0 if pir else 0.0)
    return min(max(score, 0.0), 100.0)

# ─────────────────────────────────────────────
# Auth decorator
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def _proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    return model.predict(X).astype(float)

# ─────────────────────────────────────────────
# Routes — Auth
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard") if "username" in session else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")
        if USERS.get(u) == p:
            session["username"] = u
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Please try again."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ─────────────────────────────────────────────
# Routes — Dashboard
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    with _signal_lock:
        recent = list(_signal_store)[-120:]
        nodes  = {k: dict(v) for k, v in NODE_REGISTRY.items()}
    return render_template(
        "dashboard.html",
        username=session["username"],
        nodes_json=json.dumps(nodes),
        recent_json=json.dumps(recent),
        seismic_meta=seismic_meta,
        movebank_meta=movebank_meta if isinstance(movebank_meta, dict) else {},
    )

# ─────────────────────────────────────────────
# Routes — ESP8266 signal receiver
# ─────────────────────────────────────────────
@app.route("/api/signal", methods=["POST"])
def api_signal():
    """
    Expected JSON from ESP8266:
    {
      "node":    "N01",
      "seismic": 35.2,      // 0–100
      "thermal": 42.1,      // 0–100
      "pir":     false,
      "battery": 78.4,      // 0–100 %
      "rssi":    -82         // dBm
    }
    """
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"ok": False, "error": "no body"}), 400

        node_id  = str(body.get("node", "N01")).upper()
        seismic  = float(body.get("seismic", 0))
        thermal  = float(body.get("thermal", 0))
        pir      = bool(body.get("pir", False))
        battery  = float(body.get("battery", 100))
        rssi     = int(body.get("rssi", -80))

        fusion   = _fuse(seismic, thermal, pir)
        threshold = float(body.get("threshold", 65))
        status   = "suspected" if fusion >= threshold else "cleared"
        ts       = _ts()

        # Update node registry
        with _signal_lock:
            if node_id in NODE_REGISTRY:
                NODE_REGISTRY[node_id]["battery"] = battery
                NODE_REGISTRY[node_id]["rssi"]    = rssi
                NODE_REGISTRY[node_id]["link"]    = min(100, max(0, 100 + rssi + 60))
                NODE_REGISTRY[node_id]["status"]  = "alert" if status == "suspected" else (
                    "warn" if battery < 25 or NODE_REGISTRY[node_id]["link"] < 65 else "ok"
                )
            else:
                NODE_REGISTRY[node_id] = {
                    "x": 400, "y": 380,
                    "battery": battery, "link": 80,
                    "rssi": rssi, "status": "ok"
                }

            record = {
                "id":       f"A-{int(time.time()*1000) % 9000 + 1000}",
                "ts":       ts,
                "node":     node_id,
                "seismic":  round(seismic, 2),
                "thermal":  round(thermal, 2),
                "pir":      pir,
                "battery":  round(battery, 1),
                "rssi":     rssi,
                "fusion":   round(fusion, 2),
                "status":   status,
                "type":     "Fusion",
            }
            _signal_store.append(record)

        # Broadcast to SSE clients
        payload = {**record, "nodes": {k: dict(v) for k, v in NODE_REGISTRY.items()}}
        _broadcast("signal", payload)

        # Trigger intrusion alert broadcast
        if status == "suspected":
            _broadcast("intrusion", {
                "node":    node_id,
                "fusion":  round(fusion, 2),
                "seismic": round(seismic, 2),
                "thermal": round(thermal, 2),
                "pir":     pir,
                "ts":      ts,
            })

        return jsonify({"ok": True, "fusion": round(fusion, 2), "status": status}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ─────────────────────────────────────────────
# Routes — SSE stream
# ─────────────────────────────────────────────
@app.route("/stream")
@login_required
def stream():
    client_q = queue.Queue(maxsize=50)
    _sse_clients.append(client_q)

    # Send current state immediately
    with _signal_lock:
        init_payload = json.dumps({
            "event": "init",
            "data": {
                "signals": list(_signal_store)[-50:],
                "nodes":   {k: dict(v) for k, v in NODE_REGISTRY.items()},
            }
        })

    def generate():
        yield f"data: {init_payload}\n\n"
        try:
            while True:
                try:
                    msg = client_q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                _sse_clients.remove(client_q)
            except ValueError:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

# ─────────────────────────────────────────────
# Routes — Seismic page
# ─────────────────────────────────────────────
@app.route("/seismic", methods=["GET", "POST"])
@login_required
def seismic():
    result = None
    defaults = {
        "mean": 2041.29, "top_3_mean": 2052.33, "min": 2031.0, "max": 2053.0,
        "std_dev": 3.46, "median": 2041.0, "q1": 2037.0, "q3": 2044.0,
        "skewness": -0.267, "dominant_freq": 300.0, "energy": 1955327527.0
    }
    form_values = dict(defaults)
    if request.method == "POST":
        try:
            form_values = {col: float(request.form[col]) for col in SEISMIC_FEATURES}
            sample_df = pd.DataFrame([form_values])[SEISMIC_FEATURES]
            multi_label = seismic_multi_model.predict(sample_df)[0]
            multi_probs_raw = seismic_multi_model.predict_proba(sample_df)[0]
            multi_classes   = list(seismic_multi_model.named_steps["model"].classes_)
            multi_probs = {cls: round(float(p)*100, 2) for cls, p in zip(multi_classes, multi_probs_raw)}
            binary_label = int(seismic_binary_model.predict(sample_df)[0])
            binary_prob  = round(float(seismic_binary_model.predict_proba(sample_df)[0, 1])*100, 2)
            result = {
                "multi_label": multi_label, "multi_probs": multi_probs,
                "binary_label": binary_label, "binary_prob": binary_prob,
                "binary_text": "⚠ Movement Detected" if binary_label == 1 else "✓ No Movement",
            }
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
    return render_template("seismic.html",
        username=session["username"], result=result,
        feature_cols=SEISMIC_FEATURES, defaults=defaults, form_values=form_values)

# ─────────────────────────────────────────────
# Routes — Movement page
# ─────────────────────────────────────────────
@app.route("/movement", methods=["GET", "POST"])
@login_required
def movement():
    result = None
    defaults = {
        "speed_mps": 1.2, "turning_angle_rad": 0.3, "step_length_m": 180.0,
        "roll_speed_mean": 1.1, "roll_speed_std": 0.25, "dist_to_boundary_m": 4500.0,
        "toward_human_mps": 0.1, "external-temperature": 26.0, "hour": 14,
    }
    form_values = dict(defaults)
    if request.method == "POST":
        try:
            hour = int(request.form.get("hour", 14))
            form_values = {
                "speed_mps":           float(request.form["speed_mps"]),
                "turning_angle_rad":   float(request.form["turning_angle_rad"]),
                "step_length_m":       float(request.form["step_length_m"]),
                "roll_speed_mean":     float(request.form["roll_speed_mean"]),
                "roll_speed_std":      float(request.form["roll_speed_std"]),
                "dist_to_boundary_m":  float(request.form["dist_to_boundary_m"]),
                "toward_human_mps":    float(request.form["toward_human_mps"]),
                "external-temperature": float(request.form["external-temperature"]),
                "hour_sin": math.sin(2*math.pi*hour/24),
                "hour_cos": math.cos(2*math.pi*hour/24),
                "hour":     hour,
            }
            sup_df   = pd.DataFrame([form_values])[MOVEBANK_SUP_FEATURES]
            state_df = pd.DataFrame([form_values])[MOVEBANK_STATE_FEATURES]
            risk_prob  = float(_proba_safe(movebank_risk_model, sup_df)[0])
            risk_label = int(risk_prob >= MOVEBANK_THRESHOLD)
            move_state = int(movebank_state_model.predict(state_df.values)[0])
            result = {
                "risk_prob": round(risk_prob*100, 2), "risk_label": risk_label,
                "risk_text": "🚨 Intrusion Likely" if risk_label == 1 else "✅ No Imminent Intrusion",
                "move_state": move_state,
                "state_label": f"Movement State {move_state}",
                "threshold": round(MOVEBANK_THRESHOLD*100, 1),
            }
        except Exception as e:
            flash(f"Prediction error: {e}", "error")
    return render_template("movement.html",
        username=session["username"], result=result,
        defaults=defaults, form_values=form_values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000, threaded=True)
