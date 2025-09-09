import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ----------------- Paths -----------------
DATA_DIR = "data"
MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

print("Training Energy Model (Enhanced)...")

# ----------------- Load -----------------
df = pd.read_csv(os.path.join(DATA_DIR, "energy_dataset.csv")).copy()

# ----------------- Basic cleaning -----------------
# Ensure expected columns exist
required_cols = ["Ore_Type", "Tons_Processed", "Machine_Efficiency", "Energy_MWh"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in dataset: {missing}")

# Remove obvious invalid rows (e.g., negative energy)
df = df[(df["Energy_MWh"] >= 0) & (df["Tons_Processed"] > 0)]
# Clamp efficiency to [0, 100]
df["Machine_Efficiency"] = df["Machine_Efficiency"].clip(0, 100)

# Handle negative or zero energy if any slipped through
df["Energy_MWh"] = df["Energy_MWh"].clip(lower=0)

# ----------------- Feature/Target setup -----------------
cat_cols = ["Ore_Type"]
num_cols = ["Tons_Processed", "Machine_Efficiency"]
outcome = "Energy_MWh"

X = df[cat_cols + num_cols].copy()
y_energy = df[outcome].astype(float).copy()

# ----------------- Engineer auxiliary targets (KPIs) -----------------
# Grid emission factor (tons CO2 per MWh) – set per region later if needed
GRID_TONS_CO2_PER_MWH = 0.4

# Derived KPIs for training multi-output regressors
# Energy intensity: kWh per ton
energy_intensity = (df["Energy_MWh"] * 1000.0 / df["Tons_Processed"]).replace([np.inf, -np.inf], np.nan)
energy_intensity = energy_intensity.fillna(energy_intensity.median()).clip(10, 2000)  # clamp to reasonable bounds

# CO2 tons (scope-2)
co2_tons = df["Energy_MWh"] * GRID_TONS_CO2_PER_MWH

# Peak load MW (rough estimate): base on energy and utilization
# Assume energy over an 8h shift equivalent; adjust by efficiency (lower eff => higher peak)
peak_load_mw = (df["Energy_MWh"] / 8.0) * (100.0 / (df["Machine_Efficiency"] + 1e-6))
peak_load_mw = peak_load_mw.clip(0.1, np.percentile(peak_load_mw, 99))

# Preventable loss MWh (data-driven proxy): gap to an empirical frontier
# Estimate a baseline linear-efficiency model to get expected energy, then take positive residual as preventable
eps = 1e-9
baseline_intensity = 1000.0 / (df["Machine_Efficiency"] + 10.0)  # simple decreasing curve with efficiency
expected_mwh = (baseline_intensity / 1000.0) * df["Tons_Processed"]
preventable_loss = (df["Energy_MWh"] - expected_mwh).clip(lower=0)
# Smooth and clamp
preventable_loss = preventable_loss.rolling(5, min_periods=1).mean().clip(0, np.percentile(preventable_loss, 99))

Y_multi = pd.DataFrame({
    "Predicted_Energy_MWh": y_energy.values,
    "Energy_Intensity_kWh_per_ton": energy_intensity.values,
    "CO2_tons": co2_tons.values,
    "Peak_Load_MW": peak_load_mw.values,
    "Preventable_Loss_MWh": preventable_loss.values
})

# ----------------- Preprocessing -----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ],
    remainder="drop"
)

# ----------------- Models -----------------
# Primary energy model (single-output for feature importance inspection)
energy_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=16,
        min_samples_split=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

# Multi-output KPI model
kpi_model = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    ))
])

# ----------------- Train / Eval split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_energy, test_size=0.2, random_state=42
)
_, X_test_multi, Y_train_multi, Y_test_multi = train_test_split(
    X, Y_multi, test_size=0.2, random_state=42
)

# ----------------- Fit -----------------
print("Fitting primary energy model...")
energy_model.fit(X_train, y_train)

print("Fitting KPI multi-output model...")
kpi_model.fit(X, Y_multi)  # use all data for stable KPI estimates

# ----------------- Evaluation -----------------
print("\n=== Primary Energy Model Performance ===")
y_pred = energy_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Energy_MWh: MAE={mae:.2f}, R2={r2:.3f}")

# Light CV for robustness
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(energy_model, X, y_energy, cv=kf, scoring="r2", n_jobs=-1)
print(f"CV R2: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

print("\n=== KPI Model Performance (holdout check) ===")
Y_pred_multi = pd.DataFrame(kpi_model.predict(X_test_multi), columns=Y_multi.columns)
for col in Y_multi.columns:
    mae_c = mean_absolute_error(Y_test_multi[col], Y_pred_multi[col])
    r2_c  = r2_score(Y_test_multi[col], Y_pred_multi[col])
    print(f"{col}: MAE={mae_c:.2f}, R2={r2_c:.3f}")

# ----------------- Persist artifacts -----------------
joblib.dump(energy_model, os.path.join(MODEL_DIR, "energy_model.pkl"))
joblib.dump(kpi_model, os.path.join(MODEL_DIR, "energy_kpi_model.pkl"))

# Extract feature names after encoding for app plotting and SHAP-like summaries
ohe = energy_model.named_steps["prep"].named_transformers_["cat"]
ohe_features = list(ohe.get_feature_names_out(cat_cols))
all_features = ohe_features + num_cols
joblib.dump(all_features, os.path.join(MODEL_DIR, "energy_feature_names.pkl"))

metadata = {
    "version": "v2.0",
    "grid_factor_tCO2_per_MWh": GRID_TONS_CO2_PER_MWH,
    "features_original": cat_cols + num_cols,
    "features_encoded": all_features,
    "targets": list(Y_multi.columns),
    "notes": "Primary RF for energy; MultiOutput RF for KPIs."
}
with open(os.path.join(MODEL_DIR, "energy_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved:")
print(" - energy_model.pkl (primary)")
print(" - energy_kpi_model.pkl (multi-output KPIs)")
print(" - energy_feature_names.pkl")
print(" - energy_metadata.json")
