import pandas as pd
import numpy as np

def check_data_quality(df):
    #Run basic data-quality checks on an incoming dataset
    # Ignore the target column while checking features
    features = df.drop(columns=["target"], errors="ignore")

    # Missing-value check
    missing = features.isnull().sum()

    # Duplicate records
    duplicates = int(features.duplicated().sum())

    # Infinite values
    infinite = int(np.isinf(features.select_dtypes(include=np.number)).sum().sum())

    # Basic feature statistics
    stats = features.describe().T
    stats["missing"] = missing
    stats["missing_pct"] = (missing / len(features)) * 100

    return {
        "rows": len(df),
        "columns": len(features.columns),
        "duplicate_rows": duplicates,
        "infinite_values": infinite,
        "missing_values": int(missing.sum()),
        "feature_stats": stats
    }


if __name__ == "__main__":
    # Load the production batch
    df = pd.read_csv("data/production/production_data.csv")
    results = check_data_quality(df)

    print("\nData Quality Report")
    print("\n")
    print("Rows:", results["rows"])
    print("Features:", results["columns"])
    print("Missing values:", results["missing_values"])
    print("Duplicate rows:", results["duplicate_rows"])
    print("Infinite values:", results["infinite_values"])
    print("\nFeature statistics:")
    print(results["feature_stats"][["mean", "std", "min", "max", "missing_pct"]])