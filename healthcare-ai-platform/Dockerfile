FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full application code
COPY . .

# Create required directories
RUN mkdir -p data/raw data/staged data/curated data/pdfs models exports logs

# Run the data pipeline and train the model at build time
# (Training.csv must be present in data/raw/ before building)
# If Training.csv is absent, the build skips training gracefully
RUN if [ -f data/raw/Training.csv ]; then \
        python ml/data_pipeline.py && python ml/train.py; \
    else \
        echo "WARNING: data/raw/Training.csv not found. Skipping model training."; \
    fi

# Generate synthetic patient records if patients.csv is absent
RUN if [ ! -f data/raw/patients.csv ]; then \
        python pipeline/generate_patients.py; \
    fi

# Expose FastAPI port
EXPOSE 8000

# Default command: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
