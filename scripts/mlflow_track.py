import mlflow
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load data and model
data = load_breast_cancer()
X, y = data.data, data.target
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = joblib.load("models/baseline_model.joblib")

# Track model performance
with mlflow.start_run():
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("features", X.shape[1])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", round(accuracy, 4))
    print("F1-score:", round(f1, 4))
    print("MLflow run logged.")