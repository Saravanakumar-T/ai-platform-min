import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputRegressor

DATA_DIR = "data"
MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

print("Training Machine Reliability Model (Enhanced)...")

# ---------------- Load data ----------------
df = pd.read_csv(os.path.join(DATA_DIR, "machine_data.csv")).copy()

# Basic clean
df = df.dropna().reset_index(drop=True)

# Features/target
feature_cols = ["Temperature", "Vibration", "Working_Hours"]
if not set(feature_cols).issubset(df.columns):
    raise ValueError(f"Dataset missing required columns: {feature_cols}")

X = df[feature_cols].copy()
y = df["Failure"].astype(int).values

# ---------------- Weak-supervised KPI targets ----------------
def _rng(s): return max(1e-6, (s.max() - s.min()))
t = np.clip((df["Temperature"] - df["Temperature"].min()) / _rng(df["Temperature"]), 0, 1)
v = np.clip((df["Vibration"] - df["Vibration"].min()) / _rng(df["Vibration"]), 0, 1)
h = np.clip((df["Working_Hours"] - df["Working_Hours"].min()) / _rng(df["Working_Hours"]), 0, 1)

health_score = (100.0 * (1.0 - (0.45*t + 0.35*v + 0.20*h))).clip(0, 100)
rul_days = (365.0 * (1.0 - (0.40*h + 0.35*v + 0.25*t))).clip(0, 365)
quality_idx = (100.0 * (1.0 - (0.20*t + 0.50*v + 0.30*h))).clip(0, 100)
base_maint = (90.0*(1.0 - health_score/100.0)*0.6 + (90.0*(1.0 - rul_days/365.0)*0.4))
maint_days = np.clip(base_maint, 3, 90)

Y_kpi = pd.DataFrame({
    "Health_Score": health_score.round(1),
    "Remaining_Useful_Life_days": rul_days.round(0),
    "Quality_Index": quality_idx.round(1),
    "Recommended_Maintenance_in_days": maint_days.round(0)
})

# ---------------- Preprocessing ----------------
preprocessor = ColumnTransformer(
    transformers=[("scale", StandardScaler(), feature_cols)],
    remainder="drop"
)

# ---------------- Classifier (with probability) ----------------
clf_base = Pipeline(steps=[
    ("prep", preprocessor),
    ("rf", RandomForestClassifier(
        n_estimators=300, max_depth=12,
        min_samples_split=6, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ))
])

# Calibrated probabilities (new API uses estimator=)
cal_clf = CalibratedClassifierCV(estimator=clf_base, method="sigmoid", cv=5)

# ---------------- KPI multi-output regressor ----------------
kpi_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("rf", MultiOutputRegressor(RandomForestRegressor(
        n_estimators=250, max_depth=12,
        min_samples_split=6, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )))
])

# ---------------- Train/test split ----------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Fit models ----------------
print("Fitting calibrated classifier...")
cal_clf.fit(X_train, y_train)

print("Fitting KPI model...")
kpi_model.fit(X, Y_kpi)  # use all rows for stability

# ---------------- Evaluate classifier ----------------
y_pred = cal_clf.predict(X_test)
y_proba = cal_clf.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_proba)
except Exception:
    auc = float("nan")
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f} | AUC: {auc:.3f} | F1: {f1:.3f}")

# ---------------- Persist artifacts ----------------
joblib.dump(cal_clf, os.path.join(MODEL_DIR, "machine_failure_model.pkl"))
joblib.dump(kpi_model, os.path.join(MODEL_DIR, "machine_kpi_model.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "machine_feature_names.pkl"))

meta = {
    "version": "v1.1",
    "features": feature_cols,
    "targets": ["Failure_proba","Health_Score","Remaining_Useful_Life_days","Quality_Index","Recommended_Maintenance_in_days"],
    "notes": "Uses CalibratedClassifierCV(estimator=...). KPI model via MultiOutputRegressor."
}
with open(os.path.join(MODEL_DIR, "machine_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved: machine_failure_model.pkl, machine_kpi_model.pkl, machine_feature_names.pkl, machine_metadata.json")
