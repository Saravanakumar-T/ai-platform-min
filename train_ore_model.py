import os, joblib, pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Training Ore Grade Model with Overfitting Prevention...")

df = pd.read_csv(os.path.join(DATA_DIR, "ore_grade_dataset.csv"))

# ==================== CRITICAL: CHECK ORIGINAL DATA DISTRIBUTION ====================
print("\n" + "="*60)
print("ORIGINAL DATASET ANALYSIS")
print("="*60)

original_grade_dist = df['Grade'].value_counts()
original_grade_pct = (df['Grade'].value_counts(normalize=True) * 100).round(2)

print("Original Grade Distribution:")
for grade in original_grade_dist.index:
    count = original_grade_dist[grade]
    pct = original_grade_pct[grade]
    print(f"  {grade}: {count} samples ({pct}%)")

# Check for severe class imbalance
max_class_pct = original_grade_pct.max()
if max_class_pct > 70:
    print(f"\n⚠️  WARNING: Severe class imbalance detected!")
    print(f"   Dominant class: {original_grade_pct.idxmax()} ({max_class_pct}%)")
    print("   This will cause overfitting toward the majority class!")

# Features and target
X = df.iloc[:, :-1]  # Fe, Cu, Al, Moisture
y_grade = df.iloc[:, -1]  # Grade

print(f"\nDataset size: {len(df)} samples")
print(f"Features: {list(X.columns)}")

# ==================== IMPROVED METRICS CALCULATION ====================
def calculate_realistic_metrics(row):
    fe_content = row['Fe']
    cu_content = row['Cu'] 
    al_content = row['Al']
    moisture = row['Moisture']
    
    # Add more randomness to break deterministic patterns
    noise_factor = np.random.normal(0, 0.1)  # Small random noise
    
    # Quality Score with more complexity
    base_quality = (fe_content * 1.2) + (cu_content * 15) + (al_content * 0.8)
    moisture_penalty = moisture * 3.5
    quality_score = base_quality - moisture_penalty + noise_factor * 10
    quality_score = max(10, min(95, quality_score))
    
    # Ore Usage with grade-dependent logic
    if fe_content > 50:
        base_usage = np.random.normal(150, 40)  # Increased variance
    elif fe_content > 35:
        base_usage = np.random.normal(250, 60) 
    else:
        base_usage = np.random.normal(400, 100)
    
    usage_adjustment = cu_content * 15 + noise_factor * 20
    ore_usage = base_usage - usage_adjustment + (moisture * 10)
    ore_usage = max(80, min(600, ore_usage))
    
    # Waste and Energy with more variance
    base_waste = 25 - (fe_content * 0.25) - (cu_content * 2.5) - (al_content * 0.15)
    moisture_waste = moisture * 1.8
    waste_pct = base_waste + moisture_waste + np.random.normal(0, 4)  # Increased noise
    waste_pct = max(8, min(45, waste_pct))
    
    base_energy = 35 + (ore_usage * 0.08)
    processing_difficulty = (100 - quality_score) * 0.4
    moisture_energy = moisture * 4.2
    energy_usage = base_energy + processing_difficulty + moisture_energy + np.random.normal(0, 8)
    energy_usage = max(15, min(180, energy_usage))
    
    return ore_usage, quality_score, waste_pct, energy_usage

# Set random seed
np.random.seed(42)

# Calculate metrics
print("\nCalculating enhanced metrics...")
metrics_list = []
for idx, row in X.iterrows():
    metrics = calculate_realistic_metrics(row)
    metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list, columns=['Ore_Usage_Tons', 'Quality_Score', 'Waste_Pct', 'Energy_MWh'])

# ==================== ENHANCED MODEL TRAINING ====================
# Prepare grade target
le = LabelEncoder()
y_grade_enc = le.fit_transform(y_grade)

print("\n" + "="*60)
print("MODEL TRAINING WITH OVERFITTING PREVENTION")
print("="*60)

# Stratified split to maintain class balance
X_train, X_test, y_grade_train, y_grade_test = train_test_split(
    X, y_grade_enc, test_size=0.2, random_state=42, stratify=y_grade_enc
)
_, _, metrics_train, metrics_test = train_test_split(
    X, metrics_df, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Check training set distribution
train_dist = pd.Series(y_grade_train).map(lambda x: le.inverse_transform([x])[0]).value_counts()
print(f"Training set distribution: {dict(train_dist)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== REDUCED COMPLEXITY MODEL ====================
# Use simpler model to prevent overfitting
grade_model = RandomForestClassifier(
    n_estimators=30,          # Reduced significantly
    max_depth=6,              # Shallower trees
    min_samples_split=10,     # Higher minimum
    min_samples_leaf=5,       # Higher minimum
    max_features='sqrt',      # Limit features per tree
    random_state=42,
    class_weight='balanced'   # Handle class imbalance
)

print("Training grade classification model...")
grade_model.fit(X_train_scaled, y_grade_train)

# ==================== COMPREHENSIVE EVALUATION ====================
print("\n" + "="*60)
print("OVERFITTING DETECTION & MODEL EVALUATION")
print("="*60)

# Train and test predictions
train_pred = grade_model.predict(X_train_scaled)
test_pred = grade_model.predict(X_test_scaled)

train_acc = accuracy_score(y_grade_train, train_pred)
test_acc = accuracy_score(y_grade_test, test_pred)

print(f"Training Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Accuracy Gap: {train_acc - test_acc:.3f}")

# Overfitting detection
if train_acc - test_acc > 0.15:
    print("🚨 SEVERE OVERFITTING DETECTED!")
elif train_acc - test_acc > 0.05:
    print("⚠️  Mild overfitting detected")
else:
    print("✅ No significant overfitting")

# Cross-validation
cv_scores = cross_val_score(grade_model, X_train_scaled, y_grade_train, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# ==================== PREDICTION DISTRIBUTION ANALYSIS ====================
test_pred_labels = le.inverse_transform(test_pred)
test_pred_dist = pd.Series(test_pred_labels).value_counts()
test_pred_pct = (pd.Series(test_pred_labels).value_counts(normalize=True) * 100).round(2)

print(f"\nTest Predictions Distribution:")
for grade in test_pred_dist.index:
    count = test_pred_dist[grade]
    pct = test_pred_pct[grade]
    print(f"  {grade}: {count} samples ({pct}%)")

# Check for unrealistic prediction distribution
if test_pred_pct.max() > 80:
    print(f"\n🚨 UNREALISTIC PREDICTION DISTRIBUTION!")
    print(f"   Model predicts {test_pred_pct.idxmax()} grade {test_pred_pct.max()}% of the time")
    print("   This indicates overfitting or model bias!")

# Detailed classification report
print(f"\n" + "="*50)
print("DETAILED CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_grade_test, test_pred, target_names=le.classes_))

# ==================== SAVE MODELS ONLY IF VALID ====================
# Validation criteria
valid_model = (
    test_acc > 0.4 and                    # Minimum accuracy
    train_acc - test_acc < 0.2 and        # Not too much overfitting
    test_pred_pct.max() < 85               # Not predicting single class
)

if valid_model:
    print("\n✅ Model validation passed. Saving models...")
    
    # Train metrics model
    metrics_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=30, 
            max_depth=8, 
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5
        )
    )
    metrics_model.fit(X_train_scaled, metrics_train)
    
    # Save all models
    joblib.dump(grade_model, os.path.join(MODEL_DIR, "ore_grade_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "ore_grade_label_encoder.pkl"))
    joblib.dump(metrics_model, os.path.join(MODEL_DIR, "ore_metrics_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "ore_scaler.pkl"))
    
    print("Models saved successfully!")
    
else:
    print("\n❌ Model validation failed. Models NOT saved.")
    print("Reasons:")
    if test_acc <= 0.4:
        print(f"  - Low test accuracy: {test_acc:.3f}")
    if train_acc - test_acc >= 0.2:
        print(f"  - Severe overfitting: gap = {train_acc - test_acc:.3f}")
    if test_pred_pct.max() >= 85:
        print(f"  - Unrealistic predictions: {test_pred_pct.max():.1f}% single class")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
