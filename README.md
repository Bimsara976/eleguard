# EIDS — Elephant Intrusion Detection System
## Flask Web Application · 25-26J-015

---

## Project Structure

```
eids_app/
├── app.py                         # Main Flask application
├── requirements.txt
├── README.md
├── models/                        # ← Place all .joblib / .json files here
│   ├── best_binary_proxy_intrusion_model.joblib   (Notebook 1 - Seismic)
│   ├── best_multiclass_vibration_model.joblib      (Notebook 1 - Seismic)
│   ├── model_metadata.json                         (Notebook 1 - Seismic)
│   ├── intrusion_risk_model.joblib                 (Notebook 2 - Movebank)
│   ├── movement_state_model.joblib                 (Notebook 2 - Movebank)
│   └── model_metadata.joblib                       (Notebook 2 - Movebank)
└── templates/
    ├── base.html
    ├── login.html
    ├── dashboard.html
    ├── seismic.html
    └── movement.html
```

---

## Components & Models

### Component 3 — Seismic Detection (`elephant_seismic_detection_prototype.ipynb`)
| Artifact | Purpose | Output |
|---|---|---|
| `best_multiclass_vibration_model.joblib` | 3-class vibration classifier | `walking` / `running` / `waiting` + per-class probabilities |
| `best_binary_proxy_intrusion_model.joblib` | Binary seismic intrusion proxy | `0` (no movement) / `1` (movement detected) + probability |
| `model_metadata.json` | Feature column list + best model names | — |

**Input features (11):** `mean`, `top_3_mean`, `min`, `max`, `std_dev`, `median`, `q1`, `q3`, `skewness`, `dominant_freq`, `energy`

---

### Component 4 — Movement Prediction (`Elephant_Movement_Movebank.ipynb`)
| Artifact | Purpose | Output |
|---|---|---|
| `intrusion_risk_model.joblib` | Supervised boundary-crossing risk classifier | Risk probability + binary label (horizon = 60 min) |
| `movement_state_model.joblib` | Unsupervised GMM movement state discovery | Integer state ID |
| `model_metadata.joblib` | Threshold, features, UTM EPSG, boundary_x | — |

**Input features (10):** `speed_mps`, `turning_angle_rad`, `step_length_m`, `roll_speed_mean`, `roll_speed_std`, `dist_to_boundary_m`, `toward_human_mps`, `external-temperature`, `hour_sin`, `hour_cos`  
*(hour_sin / hour_cos are computed automatically from the hour-of-day input)*

---

## Routes

| Route | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | — | Redirect to login or dashboard |
| `/login` | GET, POST | — | Hardcoded credential login |
| `/logout` | GET | ✓ | Clear session |
| `/dashboard` | GET | ✓ | System overview & module status |
| `/seismic` | GET, POST | ✓ | Seismic vibration prediction |
| `/movement` | GET, POST | ✓ | Movement & intrusion risk prediction |

---

## Setup

```bash
cd eids_app
pip install -r requirements.txt

# Copy your trained model files into models/
cp /path/to/your/models/*.joblib models/
cp /path/to/your/models/model_metadata.json models/

python app.py
# → http://127.0.0.1:5000
```
