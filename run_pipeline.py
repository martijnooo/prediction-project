import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(1)
    print("Success.")

def get_python_executable():
    """Returns the path to the venv python executable"""
    if os.name == "nt": # Windows
        venv_python = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")
    else: # Linux/Mac
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python")
        
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def main():
    print("=== Starting Full Pipeline Test ===")
    python_exe = get_python_executable()
    
    # 1. ETL
    print("\n--- Phase 1: ETL ---")
    run_command(f"{python_exe} -m etl.lambda_handler")
    
    # 2. Training
    print("\n--- Phase 2: Training ---")
    # Need to get S3 path dynamically or assume consistent naming
    # Default bucket from env
    from dotenv import load_dotenv
    load_dotenv()
    bucket = os.environ.get('AWS_BUCKET_NAME')
    if not bucket:
        print("AWS_BUCKET_NAME not set in .env")
        sys.exit(1)
        
    data_path = f"s3://{bucket}/processed/gold_prices_latest.parquet"
    run_command(f"{python_exe} -m training.train --data_path {data_path}")
    
    # 3. Inference
    print("\n--- Phase 3: Inference ---")
    run_command(f"{python_exe} -m inference.lambda_handler")
    
    print("\n=== Pipeline Complete ===")
    print("Run `streamlit run dashboard/app.py` to view the dashboard.")

if __name__ == "__main__":
    main()
