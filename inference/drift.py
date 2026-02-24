import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
import os
import boto3
from datetime import datetime

def check_drift(reference_df, current_df, column_mapping=None):
    """
    Generate an Evidently drift report comparing reference and current data.
    
    Args:
        reference_df (pd.DataFrame): Data the model was trained on
        current_df (pd.DataFrame): Latest data seen during inference
        column_mapping: Evidently ColumnMapping object (optional)
        
    Returns:
        str: HTML report as a string
    """
    print("Generating drift report...")
    
    # Standardize column names for both to ensure comparison works
    # We already do this in transform.py, but good to be safe
    
    drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    snapshot = drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # Get HTML as string
    html_report = snapshot.get_html_str(as_iframe=False)
    print(f"Generated drift report: {len(html_report)} bytes")
    return html_report

def save_report_to_s3(html_report, bucket, filename):
    """Save the HTML report to S3"""
    s3 = boto3.client('s3')
    key = f"monitoring/{filename}"
    
    try:
        print(f"Uploading drift report to s3://{bucket}/{key}")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=html_report,
            ContentType='text/html'
        )
        return True
    except Exception as e:
        print(f"Error saving report to S3: {e}")
        return False
