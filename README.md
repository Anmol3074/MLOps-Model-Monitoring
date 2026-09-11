# MLOps Model Observability

A small-scale MLOps project built to understand how machine learning models can be monitored after deployment. The project covers data quality, data drift, prediction shifts, model performance, alerting, MLflow tracking, and API-based serving.

## Features

* Data quality checks for missing, duplicate, and invalid values.
* Feature drift detection using PSI, KS-test, and Wasserstein distance.
* Model monitoring using accuracy, precision, recall, F1-score, and calibration error.
* Prediction distribution monitoring for deployed API predictions.
* Configurable `WARNING` and `CRITICAL` alert thresholds.
* MLflow-based experiment tracking.
* FastAPI endpoint for model serving.
* Monitoring reports and visualizations.
* GitHub Actions workflow for automated execution.

## Model

The project uses a Logistic Regression model with feature standardization on the Breast Cancer Wisconsin dataset from scikit-learn.

* 569 samples
* 30 input features
* Binary classification
* Stratified train/test split

## Running Locally

Create and activate a virtual environment:
python -m venv .menv
.menv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Train the model and create the datasets:
python scripts/train_model.py
python scripts/create_data.py

Simulate production drift:
python scripts/simulate_drift.py

Run the monitoring pipeline:
python scripts/run_monitoring.py

Reports and plots are generated in:
reports/

## API

Start the model API:
uvicorn scripts.api:app --reload

The API provides:
GET  /
POST /predict

Predictions are logged to:
reports/api_predictions.csv


## MLflow

Run experiment tracking with:
python scripts/mlflow_track.py

The MLflow UI can be started using:
mlflow ui

## Scope

This is a small-scale practice and portfolio project created to gain hands-on experience with MLOps and model observability.

It is not intended to be a research project, production-grade system, or publication-oriented implementation and ideas used may match previously used methods. The focus is on understanding how different MLOps components work together in a simple end-to-end workflow.

## Future Improvements

* Scheduled monitoring and automated notifications.
* Docker-based deployment
