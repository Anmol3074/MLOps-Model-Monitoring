import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI

app = FastAPI()
model = joblib.load("models/baseline_model.joblib")

@app.get("/")
def home():
    return {"status": "Model API is running"}

@app.post("/predict")
def predict(data: dict):
    X = pd.DataFrame([data])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    # Log every prediction
    os.makedirs("reports", exist_ok=True)
    log = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "prediction": int(pred),
        "confidence": float(prob)
    }])

    log.to_csv("reports/api_predictions.csv",mode="a",header=not os.path.exists("reports/api_predictions.csv"),index=False)
    return {"prediction": int(pred),"confidence": round(float(prob), 4)}