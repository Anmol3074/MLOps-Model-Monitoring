import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load model and production batches
model = joblib.load("models/baseline_model.joblib")
clean = pd.read_csv("data/production/production_data.csv")
drifted = pd.read_csv("data/production/production_drifted.csv")


def monitor(df):
    X = df.drop(columns="target")
    y = df["target"]

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    # Calculate calibration error
    calibr_err = 0
    for low, high in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)):
        mask = (prob >= low) & (prob < high)
        if mask.any():
            calibr_err += mask.mean() * abs((pred[mask] == y[mask]).mean() - prob[mask].mean())

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1_score": f1_score(y, pred),
        "positive_rate": pred.mean(),
        "mean_probability": prob.mean(),
        "calibration_error": calibr_err
    }

# Compare normal and drifted production
clean_metrics = monitor(clean)
drifted_metrics = monitor(drifted)

print("\nModel Monitoring")
print("\n")

for name in clean_metrics:
    print(f"{name:20}: {clean_metrics[name]:.4f} → {drifted_metrics[name]:.4f}")

# Measure prediction shift
print("\nPrediction Shift")
print("\n")
print(f"Positive rate : {abs(clean_metrics['positive_rate'] - drifted_metrics['positive_rate']):.4f}")
print(f"Probability   : {abs(clean_metrics['mean_probability'] - drifted_metrics['mean_probability']):.4f}")


# Save results
os.makedirs("reports", exist_ok=True)
pd.DataFrame([
    {"batch": "clean", **clean_metrics},
    {"batch": "drifted", **drifted_metrics}
]).to_csv("reports/performance_metrics.csv", index=False)

print("\nReport saved.")