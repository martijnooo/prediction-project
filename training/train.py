import os
import pandas as pd
import numpy as np
import boto3
import mlflow
import mlflow.sklearn
import dagshub
from mlflow.tracking import MlflowClient
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .model import PricePredictor
import argparse
from datetime import datetime

# Setup MLflow
# Ensure MLFLOW_TRACKING_URI is set in environment or .env

def prepare_data(df, target_col='close', lag_days=5):
    """
    Prepare data for time series prediction using lag features.
    Predict 'target_col' using past 'lag_days' values.
    """
    df_processed = df.copy()
    
    # Sort by date
    if 'date' in df_processed.columns:
        df_processed = df_processed.sort_values('date')
        
    # Create Lag Features
    for i in range(1, lag_days + 1):
        df_processed[f'lag_{i}'] = df_processed[target_col].shift(i)
        
    # Drop rows with NaN (initial lags)
    df_processed.dropna(inplace=True)
    
    # Define features (X) and target (y)
    feature_cols = [f'lag_{i}' for i in range(1, lag_days + 1)]
    X = df_processed[feature_cols]
    y = df_processed[target_col]
    
    return X, y, feature_cols

def get_champion_rmse(experiment_name, model_name):
    """
    Get the RMSE of the current champion model from MLflow.
    """
    client = MlflowClient()
    try:
        # Get the version with the 'champion' alias
        mv = client.get_model_version_by_alias(model_name, "champion")
        run = client.get_run(mv.run_id)
        return float(run.data.metrics.get("rmse", float('inf')))
    except Exception:
        # If no champion exists or error occurs
        return float('inf')

def train_flow(data_path, n_estimators, max_depth, experiment_name="Gold Price Prediction", model_name="GoldPriceRandomForest"):
    print(f"Loading data from {data_path}...")
    
    # Read from S3 or local
    if data_path.startswith('s3://'):
        from utils.s3_helper import read_parquet_s3
        df = read_parquet_s3(data_path)
    else:
        df = pd.read_parquet(data_path)
        
    print(f"Data loaded. Shape: {df.shape}")
    
    # Prepare data
    X, y, feature_names = prepare_data(df, lag_days=5)
    
    # 1. Proper Time Series Split for initial evaluation
    # Split into Train (80%) and Test (Holdout - 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Optional: Cross-validation using TimeSeriesSplit on the training set
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    print("Performing Time Series Cross-Validation...")
    for train_index, val_index in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_cv_train, y_cv_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        cv_model = PricePredictor(n_estimators=n_estimators, max_depth=max_depth)
        cv_model.train(X_cv_train, y_cv_train)
        preds = cv_model.predict(X_cv_val)
        cv_scores.append(np.sqrt(mean_squared_error(y_cv_val, preds)))
    
    avg_cv_rmse = np.mean(cv_scores)
    print(f"Average CV RMSE: {avg_cv_rmse}")
    
    # Start MLflow Run
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Enable autologging for the final training ONLY
        mlflow.sklearn.autolog(log_models=False) 
        
        print("Training final model on full dataset (train + holdout)...")
        model_wrapper = PricePredictor(n_estimators=n_estimators, max_depth=max_depth)
        model_wrapper.train(X, y) # Train on EVERYTHING for production
        
        # We still use the holdout RMSE for registering decisions to be fair
        # but the model artifacts now contain full knowledge.
        
        # Predict on Test Holdout
        test_predictions = model_wrapper.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        mae = mean_absolute_error(y_test, test_predictions)
        
        print(f"Test RMSE: {rmse}")
        print(f"Test MAE: {mae}")
        
        # Log to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", data_path)
        mlflow.log_metric("cv_rmse", avg_cv_rmse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Log Model
        mlflow.sklearn.log_model(
            sk_model=model_wrapper.model, 
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Champion logic
        champion_rmse = get_champion_rmse(experiment_name, model_name)
        print(f"Current Champion RMSE: {champion_rmse}")
        
        # Save reference data for drift detection (Force for test)
        BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
        if BUCKET_NAME:
            from io import BytesIO
            s3 = boto3.client('s3')
            ref_buffer = BytesIO()
            X_train.assign(target=y_train).to_parquet(ref_buffer, index=False)
            s3.put_object(Bucket=BUCKET_NAME, Key="monitoring/reference.parquet", Body=ref_buffer.getvalue())
            print("Reference data saved to S3.")

        if rmse < champion_rmse:
            print("New Champion found! Registering alias...")
            client = MlflowClient()
            # Get latest version for this model name
            model_info = client.get_latest_versions(model_name, stages=["None"])[0]
            client.set_registered_model_alias(model_name, "champion", model_info.version)
            print(f"Model version {model_info.version} is now the champion.")
        else:
            print("Current model did not beat the champion.")
            
        print("Run complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet file (S3 or local)")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize DagsHub BEFORE any mlflow calls to prevent 403
    dagshub.init(repo_owner=os.environ.get('DAGSHUB_USER'), repo_name=os.environ.get('DAGSHUB_REPO'), mlflow=True)
    
    # Disable global autologging here; we'll enable it specifically inside train_flow if needed 
    # or just rely on manual logging to keep the DagsHub UI clean.
    # mlflow.sklearn.autolog() 
    
    train_flow(args.data_path, args.n_estimators, args.max_depth)
