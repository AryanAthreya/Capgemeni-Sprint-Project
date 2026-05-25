# Use a slim, secure Python base image
FROM python:3.11-slim

# Create a non-root user for security (Industry Standard)
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Install system dependencies required for ML packages (gcc, g++)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies optimally
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.2.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY agents/ ./agents/
COPY ml/ ./ml/
COPY models/ ./models/
COPY data/ ./data/
COPY config.py .

# Give ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user for execution
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Health check to ensure container is running correctly
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application using Uvicorn (Detected automatically from FastAPI architecture)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
