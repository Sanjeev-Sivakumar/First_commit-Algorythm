# ============================================================
# KineticGuard - Production Container Dockerfile
# Python 3.11 Base for Google Cloud Run (Region: asia-south1)
# ============================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HOST=0.0.0.0

# Install system dependencies for OpenCV, MediaPipe C-bindings, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . /app

# Ensure runtime directories exist
RUN mkdir -p /app/output/incidents/keyframes \
    /app/output/opensearch_data \
    /app/output/reports \
    /app/data

# Expose default Cloud Run port
EXPOSE 8080

# Health check endpoint probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start the KineticGuard production dashboard server
CMD ["python", "dashboard.py", "--host", "0.0.0.0", "--port", "8080"]
