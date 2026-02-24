import pandas as pd
import numpy as np
import os
import boto3
import mlflow.sklearn
from datetime import datetime
from io import BytesIO

# Fix imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.train import prepare_data
from inference.predict import predict

def run_backtest(bucket, model_uri, start_date=None):
    """
    Run backtesting on historical data.
    """
    print(f"Starting backtest using model: {model_uri}")
    
    # 1. Load Data
    from utils.s3_helper import read_parquet_s3
    path = f"s3://{bucket}/processed/gold_prices_latest.parquet"
    df = read_parquet_s3(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 2. Load Model
    import dagshub
    dagshub.init(repo_owner=os.environ.get('DAGSHUB_USER'), repo_name=os.environ.get('DAGSHUB_REPO'), mlflow=True)
    model = mlflow.sklearn.load_model(model_uri)
    
    # 3. Iterate and Predict
    # We need at least 5 days (lags) to predict
    results = []
    
    # Let's backtest the last 30 days of data
    if start_date:
        test_df = df[df['date'] >= pd.to_datetime(start_date)]
    else:
        test_df = df.tail(30)
        
    for i in range(len(test_df)):
        current_date = test_df.iloc[i]['date']
        actual_price = test_df.iloc[i]['close']
        
        # Get historical data up to the day BEFORE current_date
        hist_df = df[df['date'] < current_date]
        
        if len(hist_df) < 10: # Minimum data needed for lags
            continue
            
        try:
            # Prepare features for the "latest" window in hist_df
            # This will predict the NEXT value (which is our current_date price)
            X, y, _ = prepare_data(hist_df.tail(15))
            
            # Use the latest feature row to predict
            prediction = predict(model, hist_df)
            
            results.append({
                'date': current_date,
                'actual': actual_price,
                'prediction': prediction,
                'error': actual_price - prediction
            })
        except Exception as e:
            print(f"Error predicting for {current_date}: {e}")
            
    backtest_df = pd.DataFrame(results)
    
    # 4. Save to S3
    if not backtest_df.empty:
        out_buffer = BytesIO()
        backtest_df.to_parquet(out_buffer, index=False)
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=bucket,
            Key="predictions/backtest_results.parquet",
            Body=out_buffer.getvalue()
        )
        print(f"Backtest complete. Saved {len(backtest_df)} predictions to S3.")
        return backtest_df
    else:
        print("No backtest results generated.")
        return None

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    BUCKET = os.environ.get('AWS_BUCKET_NAME')
    MODEL_NAME = os.environ.get('MODEL_NAME', 'GoldPriceRandomForest')
    MODEL_URI = f"models:/{MODEL_NAME}@champion"
    
    run_backtest(BUCKET, MODEL_URI)
