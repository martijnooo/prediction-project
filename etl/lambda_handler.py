import os
import boto3
import sys
import json
from datetime import datetime
from io import BytesIO

# Add current directory to path so imports work in Lambda
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract import fetch_data
from transform import transform_data

def save_to_s3(df, bucket, key):
    """Save DataFrame to S3 as Parquet"""
    if df is None:
        return False
        
    s3 = boto3.client('s3')
    try:
        # Use parquet for efficiency
        out_buffer = BytesIO()
        df.to_parquet(out_buffer, index=False)
        
        print(f"Saving to s3://{bucket}/{key}")
        s3.put_object(Bucket=bucket, Key=key, Body=out_buffer.getvalue())
        return True
    except Exception as e:
        print(f"Error saving to S3: {e}")
        return False

def handler(event, context):
    """
    AWS Lambda Handler for ETL Pipeline.
    Triggered by EventBridge (Scheduler).
    """
    print("Starting ETL job...")
    
    # Configuration
    BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')
    TICKER = os.environ.get('TICKER', 'GC=F')
    
    if not BUCKET_NAME:
        print("Error: AWS_BUCKET_NAME environment variable not set.")
        return {
            'statusCode': 500,
            'body': json.dumps('Configuration error: AWS_BUCKET_NAME missing')
        }

    # 1. Extract
    df_raw = fetch_data(ticker=TICKER)
    if df_raw is None:
        return {'statusCode': 500, 'body': 'Extraction failed'}

    # 2. Transform
    df_clean = transform_data(df_raw)
    if df_clean is None:
        return {'statusCode': 500, 'body': 'Transformation failed'}
        
    # 3. Load (Save to S3)
    # Save generic "latest" dataset and a timestamped version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Raw (optional, good for data lake)
    # save_to_s3(df_raw, BUCKET_NAME, f"raw/gold_prices_{timestamp}.parquet")
    
    # Save Processed
    key_latest = "processed/gold_prices_latest.parquet"
    key_hist = f"processed/history/gold_prices_{timestamp}.parquet"
    
    success = save_to_s3(df_clean, BUCKET_NAME, key_latest)
    
    if success:
        return {
            'statusCode': 200,
            'body': json.dumps(f'ETL Complete. Rows: {len(df_clean)}')
        }
    else:
        return {
            'statusCode': 500,
            'body': json.dumps('Failed to save to S3')
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if os.environ.get('AWS_BUCKET_NAME'):
        resp = handler({}, None)
        print(resp)
