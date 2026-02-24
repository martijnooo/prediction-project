from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import boto3
import json
import mlflow.sklearn
import dagshub
import sys
from datetime import datetime
from typing import List, Optional

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from inference.predict import predict
from utils.s3_helper import read_parquet_s3

app = FastAPI(title="Gold Price Prediction API")

# MLflow tracking configuration
if os.environ.get('MLFLOW_TRACKING_URI'):
    print(f"MLflow Tracking URI initialized: {os.environ.get('MLFLOW_TRACKING_URI')}")
else:
    print("Warning: MLFLOW_TRACKING_URI not set.")

BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

class PredictionResponse(BaseModel):
    date: str
    prediction: float
    model_used: str

class HistoricalData(BaseModel):
    date: str
    close: float

class InferenceRequest(BaseModel):
    history: List[HistoricalData]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Gold Price Prediction API. Go to /docs for Swagger UI."}

@app.get("/predict/latest", response_model=PredictionResponse)
def get_latest_prediction():
    """Fetch the latest cached prediction from S3"""
    s3 = boto3.client('s3')
    key = "predictions/latest.json"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        data = json.loads(response['Body'].read())
        return {
            "date": data['date'],
            "prediction": data['prediction'],
            "model_used": data.get('model_used', "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Latest prediction not found: {str(e)}")

@app.post("/predict/on-demand", response_model=PredictionResponse)
def on_demand_inference(request: InferenceRequest):
    """Perform on-demand inference using provided history"""
    if len(request.history) < 5:
        raise HTTPException(status_code=400, detail="History must contain at least 5 days of data.")
    
    # Convert request to DataFrame
    df = pd.DataFrame([h.model_dump() for h in request.history])
    df['date'] = pd.to_datetime(df['date'])
    
    # Load Champion Model
    model_name = os.environ.get('MODEL_NAME', 'GoldPriceRandomForest')
    model_uri = f"models:/{model_name}@champion"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        pred_val = predict(model, df)
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "prediction": float(pred_val),
            "model_used": model_uri
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
