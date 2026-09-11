import pandas as pd

# Load clean and API predictions
clean = pd.read_csv("data/production/production_data.csv")
api = pd.read_csv("reports/api_predictions.csv")

# Get baseline model predictions
from joblib import load

model = load("models/baseline_model.joblib")
baseline_pred = model.predict(clean.drop(columns="target"))
baseline_rate = baseline_pred.mean()
api_rate = api["prediction"].mean()
shift = abs(baseline_rate - api_rate)

print("\nPrediction Shift Monitoring")
print("\n")
print(f"Baseline positive rate : {baseline_rate:.4f}")
print(f"API positive rate      : {api_rate:.4f}")
print(f"Prediction shift       : {shift:.4f}")

# Simple alert threshold
if shift >= 0.10:
    print("Status: WARNING")
else:
    print("Status: STABLE")