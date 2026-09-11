import os
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# Load the same dataset used for training
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Keep the reference data separate from future production data
X_ref, X_prod, y_ref, y_prod = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)


# Add labels so we can monitor model performance later
reference = X_ref.copy()
reference["target"] = y_ref.values

production = X_prod.copy()
production["target"] = y_prod.values


# Create folders if they don't exist
os.makedirs("data/reference", exist_ok=True)
os.makedirs("data/production", exist_ok=True)


# Save both datasets
reference.to_csv("data/reference/reference_data.csv", index=False)
production.to_csv("data/production/production_data.csv", index=False)

print("Reference data:", reference.shape)
print("Production data:", production.shape)
print("Data files created successfully.")
