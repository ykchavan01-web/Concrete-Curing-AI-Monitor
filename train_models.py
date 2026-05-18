"""
train_models.py
Trains three complementary AI models for concrete curing prediction:
  1. XGBoost        → Curing Rate regression (snapshot prediction)
  2. Random Forest  → Curing Grade classification (A–F)
  3. XGBoost        → Concrete Health classification
  4. MLP Regressor  → 6-step ahead curing rate forecast (replaces LSTM)
                      Works on Python 3.12 — no TensorFlow required.

Run once before starting the monitor.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.ensemble           import RandomForestClassifier
from sklearn.neural_network     import MLPRegressor
from sklearn.multioutput        import MultiOutputRegressor
from sklearn.metrics            import (classification_report,
                                        mean_absolute_error,
                                        mean_squared_error, r2_score)
from xgboost import XGBRegressor, XGBClassifier

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs",   exist_ok=True)

# ─── LOAD DATA ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  CONCRETE CURING AI — MODEL TRAINING  (no TensorFlow)")
print("=" * 60)

df = pd.read_csv("data/training_data.csv")
print(f"\n📂 Loaded {len(df):,} records")

FEATURES = [
    "temperature_c", "humidity_pct", "soil_analog",
    "elapsed_hours", "maturity_factor",
    "temp_humidity_ix", "water_deviation"
]
TARGET_RATE   = "curing_rate_pct"
TARGET_GRADE  = "curing_grade"
TARGET_HEALTH = "concrete_health"

# ─── ENCODERS ───────────────────────────────────────────────────────────────
le_grade  = LabelEncoder()
le_health = LabelEncoder()
df["grade_enc"]  = le_grade.fit_transform(df[TARGET_GRADE])
df["health_enc"] = le_health.fit_transform(df[TARGET_HEALTH])

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])

joblib.dump(le_grade,  "models/label_encoder_grade.pkl")
joblib.dump(le_health, "models/label_encoder_health.pkl")
joblib.dump(scaler,    "models/feature_scaler.pkl")
print("✅ Encoders & scaler saved")

# ─── 1. XGBOOST — CURING RATE REGRESSION ────────────────────────────────────
print("\n[1/4] Training XGBoost Curing Rate Regressor...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df[TARGET_RATE], test_size=0.2, random_state=42
)
xgb_reg = XGBRegressor(
    n_estimators=400, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbosity=0
)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_test)
print(f"   MAE={mean_absolute_error(y_test,y_pred):.2f}%  "
      f"RMSE={mean_squared_error(y_test,y_pred)**0.5:.2f}%  "
      f"R²={r2_score(y_test,y_pred):.4f}")
joblib.dump(xgb_reg, "models/xgb_curing_rate.pkl")
print("   ✅ Saved → models/xgb_curing_rate.pkl")

# ─── 2. RANDOM FOREST — CURING GRADE (A-F) ──────────────────────────────────
print("\n[2/4] Training Random Forest Curing Grade Classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["grade_enc"], test_size=0.2,
    random_state=42, stratify=df["grade_enc"]
)
rf_grade = RandomForestClassifier(
    n_estimators=300, max_depth=18,
    min_samples_split=4, class_weight="balanced",
    n_jobs=-1, random_state=42
)
rf_grade.fit(X_train, y_train)
y_pred = rf_grade.predict(X_test)
print(classification_report(
    le_grade.inverse_transform(y_test),
    le_grade.inverse_transform(y_pred),
    target_names=le_grade.classes_
))
joblib.dump(rf_grade, "models/rf_curing_grade.pkl")
print("   ✅ Saved → models/rf_curing_grade.pkl")

# ─── 3. XGBOOST — CONCRETE HEALTH ───────────────────────────────────────────
print("\n[3/4] Training XGBoost Health Classifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["health_enc"], test_size=0.2,
    random_state=42, stratify=df["health_enc"]
)
xgb_health = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.8, eval_metric="mlogloss",
    n_jobs=-1, random_state=42, verbosity=0
)
xgb_health.fit(X_train, y_train)
y_pred = xgb_health.predict(X_test)
print(classification_report(
    le_health.inverse_transform(y_test),
    le_health.inverse_transform(y_pred),
    target_names=le_health.classes_
))
joblib.dump(xgb_health, "models/xgb_concrete_health.pkl")
print("   ✅ Saved → models/xgb_concrete_health.pkl")

# ─── 4. MLP — CURING RATE FORECAST (replaces LSTM) ──────────────────────────
print("\n[4/4] Training MLP Time-Series Forecaster (sklearn, no TensorFlow)...")

SEQ_LEN    = 10   # lookback window
PRED_STEPS = 6    # steps to forecast ahead

# Build flattened sliding-window sequences per session
X_seq, y_seq = [], []
for sid, grp in df.groupby("session_id"):
    grp  = grp.sort_values("elapsed_hours")
    vals = grp[FEATURES + [TARGET_RATE]].values.astype(np.float32)
    for i in range(len(vals) - SEQ_LEN - PRED_STEPS + 1):
        # Flatten the SEQ_LEN feature window into a 1-D input vector
        X_seq.append(vals[i : i + SEQ_LEN, :-1].flatten())
        y_seq.append(vals[i + SEQ_LEN : i + SEQ_LEN + PRED_STEPS, -1])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
print(f"   Sequences: {X_seq.shape}  →  Targets: {y_seq.shape}")

# Scale the flattened sequences with the same scaler tiled SEQ_LEN times
# (re-scale each feature block individually)
n_feat   = len(FEATURES)
X_scaled_seq = np.hstack([
    scaler.transform(X_seq[:, i*n_feat:(i+1)*n_feat])
    for i in range(SEQ_LEN)
])

split  = int(0.85 * len(X_scaled_seq))
Xtr, Xte = X_scaled_seq[:split], X_scaled_seq[split:]
ytr, yte  = y_seq[:split],        y_seq[split:]

# MLP with two hidden layers — fast to train, accurate for smooth curing curves
mlp = MultiOutputRegressor(
    MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False
    ),
    n_jobs=-1
)
mlp.fit(Xtr, ytr)

y_pred_mlp = mlp.predict(Xte)
mlp_mae    = mean_absolute_error(yte.flatten(), y_pred_mlp.flatten())
mlp_rmse   = mean_squared_error(yte.flatten(),  y_pred_mlp.flatten()) ** 0.5
print(f"\n   MLP Forecast — MAE={mlp_mae:.2f}%  RMSE={mlp_rmse:.2f}%")

joblib.dump(mlp, "models/mlp_forecast.pkl")
np.save("models/forecast_config.npy", np.array([SEQ_LEN, PRED_STEPS, n_feat]))
print("   ✅ Saved → models/mlp_forecast.pkl")

# ─── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ALL MODELS TRAINED SUCCESSFULLY  (Python 3.12 compatible)")
print("=" * 60)
print("""
  models/
  ├── xgb_curing_rate.pkl      → Snapshot curing rate (%)
  ├── rf_curing_grade.pkl      → Grade classifier (A–F)
  ├── xgb_concrete_health.pkl  → Health classifier
  ├── mlp_forecast.pkl         → 6-step rate forecast (MLP)
  ├── forecast_config.npy      → [SEQ_LEN, PRED_STEPS, N_FEATURES]
  ├── feature_scaler.pkl       → StandardScaler
  ├── label_encoder_grade.pkl  → Grade LabelEncoder
  └── label_encoder_health.pkl → Health LabelEncoder

Next step: python concrete_monitor.py --demo
""")
