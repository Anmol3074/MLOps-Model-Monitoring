import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

# Calculate Population Stability Index
def calculate_psi(reference, production, bins=10):
    edges = np.histogram_bin_edges(reference, bins=bins)

    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    # Convert counts to proportions
    ref_pct = ref_counts / len(reference)
    prod_pct = prod_counts / len(production)

    # Avoid division by zero
    ref_pct = np.clip(ref_pct, 1e-6, None)
    prod_pct = np.clip(prod_pct, 1e-6, None)

    return np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))


# Run all drift tests for one feature
def check_drift(reference, production):
    psi = calculate_psi(reference, production)
    ks_stat, ks_pvalue = ks_2samp(reference, production)
    wasserstein = wasserstein_distance(reference, production)

    return {
        "PSI": psi,
        "KS_statistic": ks_stat,
        "KS_pvalue": ks_pvalue,
        "Wasserstein": wasserstein
    }


if __name__ == "__main__":
    # Load reference and production data
    ref = pd.read_csv("data/reference/reference_data.csv")
    prod = pd.read_csv("data/production/production_drifted.csv")
    features = [col for col in ref.columns if col != "target"]
    results = []

    # Check every feature
    for feature in features:
        drift = check_drift(ref[feature], prod[feature])
        results.append({"feature": feature,**drift})

    results = pd.DataFrame(results)

    # Save results for the alerting engine
    results.to_csv("reports/drift_results.csv", index=False)
    print("\nDrift Detection Results")
    print("\n")
    print(results.round(4).to_string(index=False))
    print("\nResults saved to reports/drift_results.csv")