# Use a lightweight Python base image with Python 3.9
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

# Create and set working directory
WORKDIR /app

# Create cache directory with proper permissions
RUN mkdir -p /app/cache && chmod 777 /app/cache

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies including python-multipart
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create other necessary directories
RUN mkdir -p /app/Parsed_text /app/faiss_index

# Copy application code
COPY . .

# Set ownership of all files to the app user
RUN chown -R 1000:1000 /app

# Switch to non-root user
USER 1000

# Expose the port the app runs on
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
