import pandas as pd

# Load normal production data
df = pd.read_csv("data/production/production_data.csv")

# Introduce controlled feature drift
df["mean radius"] *= 1.25
df["mean texture"] += 3
df["worst radius"] *= 1.20

# Save the drifted batch
df.to_csv("data/production/production_drifted.csv", index=False)
print("Drifted production data created.")