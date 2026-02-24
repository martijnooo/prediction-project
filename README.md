# 🏆 GoldSight AI: Autonomous Gold Price Prediction & Monitoring

GoldSight AI is a production-grade MLOps platform designed to predict, track, and monitor the price of Gold. It leverages a serverless architecture on AWS to automate the entire machine learning lifecycle—from daily data ingestion to model serving and drift detection.

## 🚀 Core Features

*   **Automated ETL Pipeline**: Daily extraction from Yahoo Finance, technical indicator calculation, and storage in S3 as Parquet.
*   **Versioned Model Registry**: Full experiment tracking and model versioning integrated with **MLflow** and **DagsHub**.
*   **Predictive Inference**: Serverless model execution via AWS Lambda providing daily "Next Day" forecasts.
*   **Real-time Monitoring**: Automated data drift detection using **Evidently AI** to ensure model reliability.
*   **Unified Dashboard**: High-fidelity Gradio interface for market visualization and prediction insights.
*   **Developer API**: FastAPI service offering REST endpoints for on-demand predictions.

## 🏗️ Architecture Overview

The system is built on a modern MLOps stack for scalability and cost-efficiency:

1.  **Ingestion & Processing (ETL)**: AWS Lambda triggered by EventBridge fetches data and saves processed Parquet files to Amazon S3.
2.  **Training Pipeline**: retrains models including Classical (Random Forest) and potentially LSTM/ARIMA, registering the best performer as the "Champion" in MLflow.
3.  **Inference Layer**: Runs daily to generate forecasts. It performs drift analysis between the training distribution and incoming live data.
4.  **Serving Layer**: A Dockerized container running on AWS App Runner that hosts both the Gradio Dashboard and the FastAPI backend.

## 🛠️ Technology Stack

*   **Core**: Python 3.12 (Pandas, Scikit-learn, MLflow, FastAPI)
*   **UI/UX**: Gradio, Plotly
*   **Cloud (AWS)**: Lambda, S3, ECR, App Runner, IAM
*   **Tooling**: Docker, Evidently AI, DagsHub

## 📂 Project Structure

```text
├── api/                # FastAPI application code
├── dashboard/          # Gradio & Streamlit dashboard implementations
├── etl/                # Data extraction & transformation logic (Lambda)
├── inference/          # Daily prediction & drift detection scripts
├── training/           # Model training & registry logic
├── utils/              # Shared helper functions (S3, MLflow)
├── Dockerfile          # Container config for App Runner (Dashboard/API)
├── Dockerfile.lambda   # Container config for AWS Lambda functions
├── start.sh            # Production startup script
└── requirements.txt    # Project dependencies
```

## ⚙️ Local Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/martijnooo/prediction-project.git
    cd prediction-project
    ```

2.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\\Scripts\\activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file with your AWS and DagsHub credentials.

4.  **Run Dashboard Locally**:
    ```bash
    python dashboard/app_gradio.py
    ```

## ☁️ Deployment

- **Lambdas**: Use `Dockerfile.lambda` to build the image and push to ECR. Update Lambda functions to use the latest image.
- **App Runner**: Use `Dockerfile` to build the image and push to ECR. Create an App Runner service pointing to this image on Port 8080.

---
*Developed as part of a modern financial MLOps implementation.*