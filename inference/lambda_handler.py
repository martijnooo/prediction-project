import os
import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
from io import BytesIO
import mlflow.sklearn
import dagshub

# Add current directory to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .predict import predict
from .drift import check_drift, save_report_to_s3

def get_latest_data(bucket):
    """Fetch the latest processed data from S3"""
    path = f"s3://{bucket}/processed/gold_prices_latest.parquet"
    print(f"Reading data from {path}...")
    from utils.s3_helper import read_parquet_s3
    return read_parquet_s3(path)

def save_prediction(bucket, prediction, date_str):
    """Save prediction to S3"""
    s3 = boto3.client('s3')
    
    result = {
        'date': date_str,
        'prediction': float(prediction),
        'created_at': datetime.now().isoformat()
    }
    
    key = f"predictions/pred_{date_str}.json"
    print(f"Saving prediction to s3://{bucket}/{key}")
    
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(result)
    )
    
    # Update "latest" prediction for dashboard
    s3.put_object(
        Bucket=bucket,
        Key="predictions/latest.json",
        Body=json.dumps(result)
    )

def handler(event, context):
    print("Starting Inference job...")
    
    # Initialize MLflow tracking via environment variables
    # dagshub.init() can be problematic in Lambda due to token storage permissions
    # We rely on MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD being set in env
    print(f"MLflow Tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    
    BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
    MODEL_URI = os.environ.get('MODEL_URI')
    MODEL_NAME = os.environ.get('MODEL_NAME', 'GoldPriceRandomForest')
    
    try:
        df = get_latest_data(BUCKET_NAME)
        
        # Load production model
        if not MODEL_URI:
            MODEL_URI = f"models:/{MODEL_NAME}@champion"
            print(f"Loading champion model: {MODEL_URI}")

        try:
            model = mlflow.sklearn.load_model(MODEL_URI)
            prediction = predict(model, df)
            print(f"Prediction generated using model: {MODEL_URI}")
        except Exception as model_err:
            print(f"Failed to load model from {MODEL_URI}: {model_err}")
            print("Falling back to dummy prediction logic (Moving Average).")
            prediction = df['close'].iloc[-1] * 1.001 # Dummy +0.1%
            
        print(f"Prediction for next day: {prediction}")
        
        save_prediction(BUCKET_NAME, prediction, datetime.now().strftime("%Y-%m-%d"))
        
        # --- DRIFT DETECTION ---
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if project_root not in sys.path:
                sys.path.append(project_root)
                
            # Load reference data
            ref_path = f"s3://{BUCKET_NAME}/monitoring/reference.parquet"
            from utils.s3_helper import read_parquet_s3
            ref_df = read_parquet_s3(ref_path)
            
            # Prepare current data for monitoring
            from training.train import prepare_data
            X_curr, y_curr, _ = prepare_data(df.tail(40)) 
            curr_df = X_curr.assign(target=y_curr).tail(30)
            
            print(f"Reference shape: {ref_df.shape}, Current shape: {curr_df.shape}")
            
            html_report = check_drift(ref_df, curr_df)
            save_report_to_s3(html_report, BUCKET_NAME, "latest_drift_report.html")
            print("Drift report updated.")
        except Exception as drift_err:
            print(f"Drift detection failed with error: {drift_err}")
            import traceback
            traceback.print_exc()
        # ------------------------
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Inference Complete',
                'prediction': prediction,
                'model_used': MODEL_URI
            })
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Inference failed: {str(e)}")
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Mock event/context
    print(handler({}, None))
