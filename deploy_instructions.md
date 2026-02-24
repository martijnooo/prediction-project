# AWS Project Deployment Guide

This guide describes how to deploy the Gold Price Prediction project to AWS.

## 1. Prerequisites
- AWS CLI installed and configured.
- Docker installed and running.
- ECR Repositories created: `gold-prediction-lambda` and `gold-prediction-dashboard`.

## 2. Deploy ETL & Inference (Lambda)

### Build and Push Image
```bash
# Auth
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com

# Build
docker build -t gold-prediction-lambda -f Dockerfile.lambda .

# Tag
docker tag gold-prediction-lambda:latest <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/gold-prediction-lambda:latest

# Push
docker push <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/gold-prediction-lambda:latest
```

### Create/Update Lambda Functions
1. **ETL Lambda**:
   - Create function from ECR image.
   - Override CMD: `etl.lambda_handler.handler`.
2. **Inference Lambda**:
   - Create function from ECR image.
   - Override CMD: `inference.lambda_handler.handler`.

## 3. Deploy Dashboard (App Runner)

### Build and Push Image
```bash
# Build
docker build -t gold-prediction-dashboard -f Dockerfile .

# Tag
docker tag gold-prediction-dashboard:latest <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/gold-prediction-dashboard:latest

# Push
docker push <aws_account_id>.dkr.ecr.<your-region>.amazonaws.com/gold-prediction-dashboard:latest
```

### Create App Runner Service
1. Create a "Service" in AWS App Runner.
2. Select "Container registry" -> "Amazon ECR".
3. Select the `gold-prediction-dashboard` image.
4. Set Environment Variables:
   - `AWS_BUCKET_NAME`
   - `DAGSHUB_USER` / `DAGSHUB_REPO`
   - `MLFLOW_TRACKING_URI` / `MLFLOW_TRACKING_PASSWORD`
5. **Port Configuration**:
   - Set the port to `8501` (Streamlit) for health checks.
   - The API will be available on port `8001`.

## 4. API Endpoints
Once deployed, your App Runner URL will support:
- `GET /`: API Root.
- `GET /predict/latest`: Returns the most recent daily prediction.
- `POST /predict/on-demand`: Perform a custom forecast by sending history.
- `GET /docs`: Interactive Swagger documentation.

## 5. Automation (EventBridge)
1. Setup a "Rule" to trigger the ETL Lambda (e.g., Daily at 6 AM).
2. (Optional) Setup another rule or use Lambda Destinations to trigger the Inference Lambda immediately after ETL success.
