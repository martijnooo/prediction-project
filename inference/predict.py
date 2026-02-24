import os
import pandas as pd
import joblib
import boto3
import mlflow
import mlflow.sklearn
from io import BytesIO

# For now, we load from a local path or S3 artifact path if provided
# In production, we might load from S3 or Model Registry

def load_model(model_uri):
    """Load model from MLflow or local path"""
    print(f"Loading model from {model_uri}...")
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict(model, df, lag_days=5):
    """
    Generate predictions for the latest data.
    Assumes df has the necessary history to create lags.
    """
    # Create features (same logic as training)
    # We need at least lag_days rows
    if len(df) < lag_days:
        print("Not enough data to create lags.")
        return None
        
    df_sorted = df.sort_values('date')
    latest_data = df_sorted.iloc[-lag_days-1:] # Take enough for lags
    
    # We want to predict for the "next" day? 
    # Or for the "today" row if we have the close price?
    # Usually we want to forecast Tomorrow using Today's close.
    # So we take the latest available data as the input for lag_1.
    
    # Construct a single feature vector
    features = {}
    for i in range(1, lag_days + 1):
        features[f'lag_{i}'] = [df_sorted['close'].iloc[-i]]
    
    X_new = pd.DataFrame(features)
    # Ensure column order matches training
    feature_cols = [f'lag_{i}' for i in range(1, lag_days + 1)]
    X_new = X_new[feature_cols]
    
    prediction = model.predict(X_new)
    return prediction[0]

if __name__ == "__main__":
    # Test stub
    pass
