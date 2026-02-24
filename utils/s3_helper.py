import boto3
import pandas as pd
from io import BytesIO

def read_parquet_s3(path):
    """
    Read parquet file from S3 using boto3.
    Path format: s3://bucket/key or just bucket/key
    """
    if path.startswith("s3://"):
        path = path[5:]
    
    parts = path.split('/', 1)
    bucket = parts[0]
    key = parts[1]
    
    print(f"Reading from s3://{bucket}/{key} using boto3...")
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(obj['Body'].read()))
