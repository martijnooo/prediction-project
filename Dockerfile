FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY dashboard/ dashboard/
COPY inference/ inference/
COPY training/ training/
COPY utils/ utils/

COPY start.sh .

# Make startup script executable
RUN chmod +x start.sh

# Expose ports
EXPOSE 8001
EXPOSE 8080

# Command to run the app
CMD ["./start.sh"]
