# Standard Dockerfile for GCP (Cloud Run / App Engine Flexible / GKE)
# ====================================================================

# Use Python 3.11 slim image as base (smaller size, faster builds)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
# Uncomment if you need additional system packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port (Cloud Run uses PORT env var, default to 8080)
# For Cloud Run, the PORT environment variable is set automatically
EXPOSE 8080

# Set the command to run your application
# For Cloud Run, ensure your app listens on the PORT environment variable
# Example for Flask: CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
# Example for FastAPI: CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
# Example for simple Python script: CMD python main.py

# Default command (modify based on your application)
CMD ["python", "main.py"]

# ====================================================================
# GCP-Specific Notes:
# ====================================================================
# 
# Cloud Run:
#   - Must listen on PORT environment variable (set by Cloud Run)
#   - Health checks: /healthz or / endpoint should return 200
#   - Timeout: Default 300s, max 3600s
#   - Memory: 128MB to 8GB
#   - CPU: 1-8 vCPUs
#
# App Engine Flexible:
#   - Similar to Cloud Run but uses app.yaml configuration
#   - Port 8080 is standard
#
# GKE (Google Kubernetes Engine):
#   - Standard Kubernetes deployment
#   - Use ConfigMaps/Secrets for environment variables
#
# Build and Deploy:
#   gcloud builds submit --tag gcr.io/PROJECT-ID/SERVICE-NAME
#   gcloud run deploy SERVICE-NAME --image gcr.io/PROJECT-ID/SERVICE-NAME
#
# ====================================================================

