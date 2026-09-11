import pandas as pd

# Load drift results
df = pd.read_csv("reports/drift_report.csv")

critical = (df["status"] == "CRITICAL").sum()
warning = (df["status"] == "WARNING").sum()

print("\nMLOps Alert Summary")
print("\n")
print("Critical:", critical)
print("Warning :", warning)

if critical > 0:
    print("Overall Status: CRITICAL")
elif warning > 0:
    print("Overall Status: WARNING")
else:
    print("Overall Status: STABLE")