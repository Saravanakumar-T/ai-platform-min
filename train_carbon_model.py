import os, json, joblib, numpy as np, pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ----------------- Paths -----------------
DATA_DIR = "data"
MODEL_DIR = "models"
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

# ----------------- Load -----------------
print("Training Carbon Capture Models (Enhanced)...")
df = pd.read_csv(os.path.join(DATA_DIR, "carbon_capture_dataset.csv")).copy()

# ----------------- Features/Target -----------------
# X: numeric features
num_cols = ["CO2_ppm","Plant_Size_MW","Energy_Cost","Pressure_bar"]
X = df[num_cols].copy()

# y_class: technology label
y_class = df["Technology"].copy()

# ----------------- Create realistic secondary targets (labels for regression) -----------------
# Note: These are physics-/industry-informed heuristics used to synthesize targets.
# If you have measured targets, replace these formulas with your real columns.

def synthesize_targets(row: pd.Series) -> Tuple[float,float,float,float,float]:
    co2, size_mw, energy_cost, p_bar = row["CO2_ppm"], row["Plant_Size_MW"], row["Energy_Cost"], row["Pressure_bar"]
    # Baselines by expected tech:
    # Absorption: moderate energy, high water, good efficiency
    # Adsorption: lower water, moderate energy, decent efficiency
    # Membrane: low opex, moderate efficiency, low waste
    # Cryogenic: high energy, highest efficiency at high pressure
    
    # Start with neutral baselines
    base_energy = 120 + 0.03*co2 + 0.02*p_bar                   # kWh/tCO2
    base_eff = 70 + 0.02*(p_bar*co2/1000)                       # %
    base_opex = 60 + 0.15*energy_cost + 0.005*size_mw           # USD/tCO2
    base_waste = 8 + 0.005*size_mw + 0.3*(energy_cost/100)      # kg/tCO2
    base_water = 0.5 + 0.002*size_mw + 0.05*p_bar                # m3/tCO2

    # Adjust by inferred tech from pressure & cost pattern (soft prior)
    if p_bar >= 7.0 and co2 >= 800:
        # Cryogenic favored
        base_energy += 80
        base_eff += 8
        base_opex += 20
        base_waste += 2
        base_water += 0.2
    elif energy_cost <= 60 and 500 <= co2 <= 900:
        # Absorption favored
        base_energy += 25
        base_eff += 5
        base_opex += 15
        base_waste += 4
        base_water += 0.5
    elif p_bar <= 3.5 and energy_cost <= 70:
        # Membrane favored
        base_energy -= 10
        base_eff -= 2
        base_opex -= 8
        base_waste -= 2
        base_water -= 0.1
    else:
        # Adsorption favored
        base_energy -= 5
        base_eff += 1
        base_opex -= 2
        base_waste -= 1
        base_water -= 0.05

    # Clamp realistic ranges
    energy_kwh_per_t = float(np.clip(base_energy + np.random.normal(0, 5), 60, 260))
    capture_eff_pct  = float(np.clip(base_eff   + np.random.normal(0, 2), 50, 95))
    opex_usd_per_t   = float(np.clip(base_opex  + np.random.normal(0, 5), 35, 220))
    waste_kg_per_t   = float(np.clip(base_waste + np.random.normal(0, 1), 3, 40))
    water_m3_per_t   = float(np.clip(base_water + np.random.normal(0, 0.1), 0.2, 6.0))
    return energy_kwh_per_t, capture_eff_pct, opex_usd_per_t, waste_kg_per_t, water_m3_per_t

reg_targets = X.apply(synthesize_targets, axis=1, result_type="expand")
reg_targets.columns = ["Energy_kWh_per_tCO2","Capture_Eff_%","OPEX_USD_per_tCO2","Waste_kg_per_tCO2","Water_m3_per_tCO2"]

# ----------------- Encode target -----------------
le = LabelEncoder()
y_enc = le.fit_transform(y_class)

# ----------------- Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
Yreg_train, Yreg_test = train_test_split(
    reg_targets, test_size=0.2, random_state=42
)

# ----------------- Pipelines -----------------
# Scaler for numeric features
num_transformer = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[("num", num_transformer, num_cols)]
)

# Classifier with regularization to avoid overfitting
clf = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ))
])

# Multi-output regressor for KPIs
regr = Pipeline(steps=[
    ("prep", preprocessor),
    ("model", MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=42
        )
    ))
])

# ----------------- Train -----------------
print("Fitting classifier...")
clf.fit(X_train, y_train)

print("Fitting regressor...")
regr.fit(X_train, Yreg_train)

# ----------------- Evaluation -----------------
print("\n=== Classification Performance ===")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Regression Performance (test) ===")
Yreg_pred = pd.DataFrame(regr.predict(X_test), columns=Yreg_train.columns)
for col in Yreg_train.columns:
    mae = mean_absolute_error(Yreg_test[col], Yreg_pred[col])
    r2  = r2_score(Yreg_test[col], Yreg_pred[col])
    print(f"{col}: MAE={mae:.2f}, R2={r2:.3f}")

# Simple CV to check stability
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
print(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# ----------------- Save -----------------
joblib.dump(clf, os.path.join(MODEL_DIR, "carbon_capture_model.pkl"))
joblib.dump(regr, os.path.join(MODEL_DIR, "carbon_capture_kpi_model.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "carbon_capture_label_encoder.pkl"))
joblib.dump(num_cols, os.path.join(MODEL_DIR, "carbon_capture_feature_names.pkl"))

# Metadata for the app
metadata = {
    "features": num_cols,
    "class_labels": list(le.classes_),
    "kpi_targets": list(Yreg_train.columns),
    "version": "v2.0",
    "notes": "Classifier + KPI regressor with scaling and class weighting."
}
with open(os.path.join(MODEL_DIR, "carbon_capture_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved:")
print(" - carbon_capture_model.pkl (technology classifier)")
print(" - carbon_capture_kpi_model.pkl (multi-output KPI regressor)")
print(" - carbon_capture_label_encoder.pkl")
print(" - carbon_capture_feature_names.pkl")
print(" - carbon_capture_metadata.json")
