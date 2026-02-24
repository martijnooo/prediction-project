import gradio as gr
import pandas as pd
import boto3
import json
import plotly.graph_objects as go
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

# --- Data Loading Functions ---
def load_data():
    """Load historical data from S3"""
    if not BUCKET_NAME:
        return None
    path = f"s3://{BUCKET_NAME}/processed/gold_prices_latest.parquet"
    try:
        from utils.s3_helper import read_parquet_s3
        return read_parquet_s3(path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_latest_prediction():
    """Load latest prediction from S3"""
    if not BUCKET_NAME:
        return None
    s3 = boto3.client('s3')
    key = "predictions/latest.json"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(response['Body'].read())
    except Exception:
        return None

def load_drift_report():
    """Load latest drift report HTML"""
    if not BUCKET_NAME:
        return "AWS_BUCKET_NAME not set."
    s3 = boto3.client('s3')
    key = "monitoring/latest_drift_report.html"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        return f"<h1>No drift report found</h1><p>{str(e)}</p>"

# --- Dashboard Logic ---
def get_dashboard_metrics():
    df = load_data()
    pred = load_latest_prediction()
    
    current_price = "N/A"
    date_str = ""
    if df is not None and not df.empty:
        df_sorted = df.sort_values('date')
        current_price = f"${df_sorted['close'].iloc[-1]:.2f}"
        date_str = df_sorted['date'].iloc[-1].strftime("%Y-%m-%d")
    
    forecast = "N/A"
    delta_text = ""
    model_info = "Unknown"
    
    if pred:
        val = pred['prediction']
        forecast = f"${val:.2f}"
        model_info = pred.get('model_used', 'Unknown Model').split('/')[-1]
        
        # Calculate delta if possible
        if current_price != "N/A":
             curr_val = float(current_price.replace('$',''))
             delta = val - curr_val
             sign = "+" if delta > 0 else ""
             delta_text = f"({sign}{delta:.2f})"

    # Markdown for metrics
    metrics_md = f"""
    ### 💰 Market Status
    **Current Price:** {current_price} ({date_str})  
    **Tomorrow's Forecast:** {forecast} {delta_text}  
    **Active Model:** {model_info}
    """
    return metrics_md

def get_price_chart():
    df = load_data()
    pred = load_latest_prediction()
    
    fig = go.Figure()
    
    if df is not None:
        subset = df.sort_values('date').tail(90)
        
        # Actuals
        fig.add_trace(go.Scatter(
            x=subset['date'], 
            y=subset['close'], 
            mode='lines', 
            name='Actual Price',
            line=dict(color='#0369A1', width=2)
        ))
        
        # Forecast Point
        if pred:
            last_date = subset['date'].max()
            forecast_date = last_date + timedelta(days=1)
            
            # Link Line
            fig.add_trace(go.Scatter(
                x=[last_date, forecast_date], 
                y=[subset['close'].iloc[-1], pred['prediction']], 
                mode='lines', 
                name='Forecast Path', 
                line=dict(color='#F72585', dash='dot'), 
                showlegend=False
            ))
            
            # Marker
            fig.add_trace(go.Scatter(
                x=[forecast_date], 
                y=[pred['prediction']], 
                mode='markers+text', 
                name='Next Forecast', 
                text=[f"${pred['prediction']:.2f}"], 
                textposition="top right", 
                marker=dict(size=14, color='#F72585', symbol='diamond')
            ))
            
    fig.update_layout(
        title="Gold Price History & Prediction",
        template="plotly_white",
        height=500
    )
    return fig

def predict_custom(history_df):
    # This roughly mimics the 'On-Demand' logic
    # In a real app we'd call the prediction model function or the deployed API
    try:
        # Just a dummy return for the demo if model loading is complex in Gradio process
        # Ideally we import predict() from inference/predict.py if dependencies allow
        return f"Prediction logic triggered for {len(history_df)} rows. (Mock Result: $2500.00)"
    except Exception as e:
        return f"Error: {e}"

# --- Build Blocks ---
with gr.Blocks(title="GoldSight AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 GoldSight AI Dashboard")
    gr.Markdown("Next-Gen Financial Forecasting powered by AWS Lambda & MLflow.")
    
    with gr.Tab("Live Dashboard"):
        with gr.Row():
            metrics_output = gr.Markdown(get_dashboard_metrics)
            refresh_btn = gr.Button("🔄 Refresh Data")
        
        chart_output = gr.Plot(get_price_chart)
        
        refresh_btn.click(get_dashboard_metrics, outputs=metrics_output)
        refresh_btn.click(get_price_chart, outputs=chart_output)

    with gr.Tab("Model Monitoring"):
        gr.Markdown("### 🛡️ Evidently Data Drift Report")
        drift_output = gr.HTML(load_drift_report)
        refresh_drift = gr.Button("Reload Report")
        refresh_drift.click(load_drift_report, outputs=drift_output)

    with gr.Tab("About"):
        gr.Markdown("""
        ### Architecture
        - **ETL**: AWS Lambda (Daily) -> S3 Parquet
        - **Training**: AWS Lambda (Weekly) -> MLflow
        - **Inference**: AWS Lambda (Daily) -> S3 JSON
        - **Hosting**: AWS App Runner (Dockerized Gradio)
        """)

# Launch
if __name__ == "__main__":
    # App Runner expects listening on 0.0.0.0 and a specific port (we stick to 8080)
    demo.launch(server_name="0.0.0.0", server_port=8080)
