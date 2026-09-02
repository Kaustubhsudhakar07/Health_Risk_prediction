# Use an official lightweight Python runtime
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables for non-interactive and unbuffered outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Install system dependencies (needed for OpenMP / LightGBM / XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Generate sample benchmark data and train baseline model if not already present
RUN python run_pipeline.py --benchmark || true

# Expose Streamlit default port
EXPOSE 8501

# Health check to ensure the container is responsive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
