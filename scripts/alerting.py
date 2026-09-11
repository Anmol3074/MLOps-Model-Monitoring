import pandas as pd
import yaml

# Load monitoring thresholds
with open("config/thresholds.yaml", "r") as file:
    thresholds = yaml.safe_load(file)

def classify_drift(row):
    #Classify a feature based on PSI and KS test.
    psi = row["PSI"]
    pvalue = row["KS_pvalue"]
    if psi >= thresholds["drift"]["psi_critical"]:
        return "CRITICAL"
    if psi >= thresholds["drift"]["psi_warning"] or pvalue < thresholds["drift"]["ks_pvalue"]:
        return "WARNING"
    return "STABLE"

if __name__ == "__main__":
    # Load drift results
    results = pd.read_csv("reports/drift_results.csv")
    results["status"] = results.apply(classify_drift, axis=1)

    print("\nDrift Alerts")
    print("\n")
    print(results[["feature", "PSI", "KS_pvalue", "status"]].to_string(index=False))
    print("\nAlert Summary")
    print("\n")
    print(results["status"].value_counts())