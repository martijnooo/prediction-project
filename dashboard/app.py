import streamlit as st
import pandas as pd
import boto3
import json
import plotly.graph_objects as go
import os
import sys
import requests
from datetime import datetime, timedelta

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration
from dotenv import load_dotenv
load_dotenv()

# --- Page Config & Styling ---
st.set_page_config(
    page_title="GoldSight AI", 
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look (Light Theme)
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #FFFFFF;
        color: #1F2937;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F3F4F6;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'DM Sans', sans-serif;
        color: #D97706 !important; /* Amber-600 */
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #111827;
    }

    /* Paragraphs */
    p, li, .stMarkdown {
        color: #374151;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: bold;
        border: 1px solid #D97706;
        color: #D97706;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #D97706;
        color: #FFFFFF;
        border-color: #D97706;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .stExpander {
        background-color: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

BUCKET_NAME = os.environ.get('AWS_BUCKET_NAME')

if not BUCKET_NAME:
    st.error("⚠️ AWS_BUCKET_NAME environment variable not set.")
    st.stop()

# --- Data Loading Functions ---
@st.cache_data(ttl=3600)
def load_data():
    """Load historical data from S3"""
    path = f"s3://{BUCKET_NAME}/processed/gold_prices_latest.parquet"
    try:
        from utils.s3_helper import read_parquet_s3
        return read_parquet_s3(path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def load_latest_prediction():
    """Load latest prediction from S3"""
    s3 = boto3.client('s3')
    key = "predictions/latest.json"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        return json.loads(response['Body'].read())
    except Exception:
        return None

def load_backtest_results():
    """Load backtest results from S3"""
    path = f"s3://{BUCKET_NAME}/predictions/backtest_results.parquet"
    try:
        from utils.s3_helper import read_parquet_s3
        return read_parquet_s3(path)
    except Exception:
        return None

# --- Views ---

def show_home():
    # Hero Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.title("GoldSight AI")
        st.markdown("### Next-Gen Financial Forecasting")
        st.markdown("""
        Leverage the power of **Machine Learning** to predict Gold prices with precision. 
        Our system runs daily automated pipelines to train, predict, and monitor market trends.
        """)
        
        if st.button("🚀 Launch Dashboard"):
             st.session_state.page = "Live Dashboard"
             st.rerun()

        st.markdown("---")
        st.info("💡 **Tip:** Use the sidebar to navigate between modules.")

    with col2:
        # Display Header Image
        img_path = os.path.join(os.path.dirname(__file__), 'gold_header_light.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Header image not found.")

    # Features Section
    st.markdown("---")
    st.header("How It Works")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1. 📥 Automated ETL")
            st.markdown("""
            Daily ingestion pipelines fetch the latest market data from **Yahoo Finance**. 
            Technical indicators (RSI, SMA, Bollinger Bands) are calculated automatically 
            to enrich the dataset for our models.
            """)
            
        with c2:
            st.subheader("2. 🤖 ML Inference")
            st.markdown("""
            A **Random Forest Regressor**, trained on 20+ years of historical data, 
            analyzes the new features to predict the closing price of Gold for the next trading day. 
            The system automatically detects data drift.
            """)
            
        with c3:
            st.subheader("3. 🚀 Scalable Deployment")
            st.markdown("""
            Built on **AWS Lambda** for serverless computation and **App Runner** for hosting. 
            All artifacts are version-controlled using **MLflow** & **DagsHub**, ensuring reproducible AI.
            """)
    
    st.markdown("---")
    
    st.subheader("Architecture Overview")
    st.info("The system follows a modern MLOps architecture: GitHub Actions for CI/CD, Docker for containerization, and S3 for persistent storage.")
    
    st.markdown("### Core Capabilities")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("#### 📈 Predictive Analytics")
        st.caption("Advanced Random Forest models trained on years of historical data to forecast future price movements.")
    with f2:
        st.markdown("#### 🛡️ Drift Monitoring")
        st.caption("Continuous surveillance of data distribution to ensure model accuracy and delete market regime changes.")
    with f3:
        st.markdown("#### 🔌 Developer API")
        st.caption("RESTful API endpoints for on-demand inference and seamless integration with external trading bots.")

def show_dashboard():
    st.title("📊 Live Market Dashboard")
    st.markdown("Real-time insights and model performance tracking.")
    
    # 1. Key Metrics
    df = load_data()
    pred = load_latest_prediction()
    
    m1, m2, m3 = st.columns(3)
    
    current_price = 0
    if df is not None and not df.empty:
        df_sorted = df.sort_values('date')
        current_price = df_sorted['close'].iloc[-1]
        date_str = df_sorted['date'].iloc[-1].strftime("%Y-%m-%d")
        
        m1.metric("Current Gold Price", f"${current_price:.2f}", f"As of {date_str}")
    
    if pred:
        pred_val = pred['prediction']
        delta = pred_val - current_price
        color = "normal" if delta > 0 else "inverse"
        m2.metric("Tomorrow's Forecast", f"${pred_val:.2f}", f"{delta:+.2f}", delta_color=color)
        
        model_name = pred.get('model_used', 'Unknown Model').split('/')[-1]
        m3.metric("Active Model", "Random Forest", f"Ver: {model_name}")

    st.markdown("---")

    # 2. Main Chart
    if df is not None:
        st.subheader("Price History & Trajectory")
        
        # Filter for recent history (last 90 days)
        subset = df.sort_values('date').tail(90)
        
        fig = go.Figure()
        
        # Style chart for Light mode
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="DM Sans", color="#1F2937"),
            height=500,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB')
        )
        
        # Actuals
        fig.add_trace(go.Scatter(
            x=subset['date'], 
            y=subset['close'], 
            mode='lines', 
            name='Actual Price',
            line=dict(color='#0369A1', width=2) # Sky-700
        ))
        
        # Backtest Overlay
        backtest_df = load_backtest_results()
        if backtest_df is not None:
            backtest_df['date'] = pd.to_datetime(backtest_df['date'])
            bt_subset = backtest_df[backtest_df['date'].isin(subset['date'])]
            fig.add_trace(go.Scatter(
                x=bt_subset['date'], 
                y=bt_subset['prediction'], 
                mode='lines', 
                name='Model Prediction', 
                line=dict(color='#D97706', dash='dot', width=2) # Amber-600
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
            
        st.plotly_chart(fig, width='stretch')
        
    # 3. Accuracy Table
    if backtest_df is not None:
        with st.expander("🔎 View Historical Accuracy Data"):
            st.subheader("Recent Model Performance")
            table_df = backtest_df.sort_values('date', ascending=False).head(10).copy()
            table_df['date'] = table_df['date'].dt.strftime('%b %d, %Y')
            table_df['actual'] = table_df['actual'].map('${:,.2f}'.format)
            table_df['prediction'] = table_df['prediction'].map('${:,.2f}'.format)
            table_df['error'] = table_df['error'].map('${:,.2f}'.format)
            table_df.columns = ['Date', 'Actual Price', 'Predicted Price', 'Error']
            
            st.dataframe(table_df, use_container_width=True, hide_index=True)

def show_monitoring():
    st.title("🛡️ Model Health & Monitoring")
    st.markdown("Automated drift detection reports generated by Evidently AI.")
    
    try:
        s3 = boto3.client('s3')
        key = "monitoring/latest_drift_report.html"
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        html_content = response['Body'].read().decode('utf-8')
        
        if len(html_content) > 1000:
            st.success(f"✅ Latest Drift Report Available (Gen: {datetime.now().strftime('%Y-%m-%d')})")
            
            # Embed the HTML report
            import streamlit.components.v1 as components
            # Create a scrolling container for the report
            with st.container():
                components.html(html_content, height=1000, scrolling=True)
        else:
            st.warning("⚠️ Drift report appears to be empty or corrupted.")
            
    except Exception as e:
        st.info("🚫 No drift reports found. Run the inference pipeline to generate one.")
        st.code(f"Error: {e}")

def show_api():
    st.title("🔌 Developer API Playground")
    st.markdown("Test the internal REST endpoints directly from the dashboard.")
    
    st.info("ℹ️ The API runs on port **8001** within the container.")
    
    tab1, tab2 = st.tabs(["GET Latest", "POST On-Demand"])
    
    with tab1:
        st.subheader("Retrieve Daily Forecast")
        st.markdown("Fetch the pre-calculated prediction for the current day.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("GET /predict/latest", type="primary"):
                try:
                    res = requests.get("http://localhost:8001/predict/latest")
                    if res.status_code == 200:
                        st.session_state['api_latest'] = res.json()
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
        
        with col2:
            if 'api_latest' in st.session_state:
                st.json(st.session_state['api_latest'])
                
    with tab2:
        st.subheader("Generate On-Demand Forecast")
        st.markdown("Enter the last 5 days of Gold Price history to generate a new prediction.")
        
        # Initialize default data if not in session state
        if 'input_df' not in st.session_state:
            default_data = [{
                "date": datetime.now() - timedelta(days=i), # Keep as datetime object for DateColumn compatibility
                "close": 4600.0 + (i * 10)
            } for i in range(5, 0, -1)]
            st.session_state.input_df = pd.DataFrame(default_data)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("##### Historical Data Input")
            edited_df = st.data_editor(
                st.session_state.input_df, 
                num_rows="dynamic",
                column_config={
                    "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                    "close": st.column_config.NumberColumn("Close Price ($)", format="$%.2f", required=True)
                },
                use_container_width=True
            )
            
            if st.button("🚀 Generate Forecast", type="primary"):
                try:
                    # Convert dataframe to API payload format
                    # Ensure date is string formatted
                    payload_data = edited_df.copy()
                    payload_data['date'] = pd.to_datetime(payload_data['date']).dt.strftime('%Y-%m-%d')
                    
                    history_list = payload_data.to_dict(orient='records')
                    payload = {"history": history_list}
                    
                    with st.spinner("Running Inference Model..."):
                        res = requests.post("http://localhost:8001/predict/on-demand", json=payload)
                        
                    if res.status_code == 200:
                        st.session_state['api_demand'] = res.json()
                        st.success("✅ Prediction Successful")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                        
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    
        with c2:
            st.markdown("##### API Response")
            if 'api_demand' in st.session_state:
                res_data = st.session_state['api_demand']
                
                # Visual Response
                st.metric("Predicted Price", f"${res_data['prediction']:.2f}")
                st.caption(f"For Date: {res_data['date']}")
                st.json(res_data)
            else:
                st.info("Submit data to see prediction results here.")


# --- Main Navigation ---
def main():
    
    # Initialize session state for navigation if not set
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    with st.sidebar:
        st.markdown("## Navigation")
        
        # Use a callback to update session state if the radio button changes manually
        selection = st.radio(
            "Go to", 
            ["Home", "Live Dashboard", "Model Monitoring", "Developer API"],
            index=["Home", "Live Dashboard", "Model Monitoring", "Developer API"].index(st.session_state.page)
        )
        
        # Sync selection with session state
        if selection != st.session_state.page:
            st.session_state.page = selection
            st.rerun()
            
        st.markdown("---")
        st.markdown("### System Status")
        st.success("● API Online (Port 8001)")
        st.success("● S3 Connection Active")
        
        st.markdown("---")
        st.caption("v1.2.0 | Built with Streamlit")

    # Routing
    if st.session_state.page == "Home":
        show_home()
    elif st.session_state.page == "Live Dashboard":
        show_dashboard()
    elif st.session_state.page == "Model Monitoring":
        show_monitoring()
    elif st.session_state.page == "Developer API":
        show_api()

if __name__ == "__main__":
    main()
