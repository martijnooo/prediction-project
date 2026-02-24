#!/bin/bash

# Start FastAPI in the background
echo "Starting FastAPI on port 8001..."
# uvicorn api.main:app --host 0.0.0.0 --port 8001 &

# Start Gradio App
echo "Starting Gradio Dashboard on port 8080..."
python dashboard/app_gradio.py
