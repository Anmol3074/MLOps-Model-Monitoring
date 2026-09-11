import os
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target
print("Dataset shape:", X.shape)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features and train the classifier
model = Pipeline([("scaler", StandardScaler()),("classifier", LogisticRegression(max_iter=2000))])
model.fit(X_train, y_train)

# Check model performance
pred = model.predict(X_test)

print("\nBaseline Model Performance")
print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"Precision: {precision_score(y_test, pred):.4f}")
print(f"Recall   : {recall_score(y_test, pred):.4f}")
print(f"F1-score : {f1_score(y_test, pred):.4f}")


# Save the trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/baseline_model.joblib")

print("\nModel saved successfully.")
