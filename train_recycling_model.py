import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Training Recycling Model...")

df = pd.read_csv(os.path.join(DATA_DIR, "recycling_dataset.csv"))

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(MODEL_DIR, "recycling_model.pkl"))

print("✅ Recycling model saved in 'models/'")
