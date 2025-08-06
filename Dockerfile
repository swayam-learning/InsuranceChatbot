<<<<<<< HEAD
# Use a lightweight Python base image with Python 3.9
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080
ENV HOST 0.0.0.0
ENV HF_HOME=/tmp/hf_cache
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /tmp/Parsed_text /tmp/faiss_index /tmp/hf_cache
=======
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080
ENV HOST 0.0.0.0
ENV HF_HOME=/app/cache
ENV TOKENIZERS_PARALLELISM=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create cache directory
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create other directories
RUN mkdir -p /app/Parsed_text /app/faiss_index
>>>>>>> 614d1978b3935932050b0459fb1f19da2d2ae7d0

# Copy application code
COPY . .

<<<<<<< HEAD
# Expose the port the app runs on
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
=======
# Set ownership
RUN chown -R 1000:1000 /app

USER 1000

EXPOSE $PORT

# Pre-download the smaller model
CMD ["sh", "-c", "python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L5-v2', cache_folder='/app/cache')\" && uvicorn main:app --host 0.0.0.0 --port 8080"]
>>>>>>> 614d1978b3935932050b0459fb1f19da2d2ae7d0
