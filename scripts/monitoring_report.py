import os
import pandas as pd
import matplotlib.pyplot as plt

# Load monitoring results
drift = pd.read_csv("reports/drift_report.csv")
performance = pd.read_csv("reports/performance_metrics.csv")
os.makedirs("reports/plots", exist_ok=True)

# Plot top feature drift
top = drift.sort_values("PSI", ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.bar(top["feature"], top["PSI"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("PSI")
plt.title("Top Feature Drift")
plt.tight_layout()
plt.savefig("reports/plots/feature_drift.png")
plt.close()

# Compare model performance
metrics = ["accuracy", "precision", "recall", "f1_score"]

performance.set_index("batch")[metrics].plot(kind="bar", figsize=(9, 5))
plt.ylabel("Score")
plt.title("Model Performance Monitoring")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("reports/plots/model_performance.png")
plt.close()

print("Monitoring plots generated.")