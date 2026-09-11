import pandas as pd
import joblib

from data_quality import check_data_quality
from drift_detection import check_drift
from alerting import classify_drift
from performance_monitor import monitor

# Load data and model
ref = pd.read_csv("data/reference/reference_data.csv")
clean = pd.read_csv("data/production/production_data.csv")
prod = pd.read_csv("data/production/production_drifted.csv")
model = joblib.load("models/baseline_model.joblib")

# Data quality
quality = check_data_quality(prod)
print("\nData Quality:", quality["missing_values"],
      "missing,", quality["duplicate_rows"], "duplicates")

# Feature drift
features = [c for c in ref.columns if c != "target"]
results = []

for feature in features:
    drift = check_drift(ref[feature], prod[feature])
    results.append({"feature": feature, **drift})

results = pd.DataFrame(results)
results["status"] = results.apply(classify_drift, axis=1)
results.to_csv("reports/drift_report.csv", index=False)

# Model performance
clean_metrics = monitor(clean)
prod_metrics = monitor(prod)

print("\nModel Performance")
for metric in ["accuracy", "precision", "recall", "f1_score"]:
    print(f"{metric}: {clean_metrics[metric]:.4f} -> {prod_metrics[metric]:.4f}")

# Prediction shift
clean_pred = model.predict(clean.drop(columns="target"))
prod_pred = model.predict(prod.drop(columns="target"))

print("\nPrediction Shift:",abs(clean_pred.mean() - prod_pred.mean()))
print("\nAlerts:")
print(results["status"].value_counts())
print("\nPipeline completed.")