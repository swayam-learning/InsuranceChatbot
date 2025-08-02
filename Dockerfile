FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies needed for faiss, building wheels, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Disable HuggingFace caching to avoid file lock/permission issues
ENV HF_HOME=""
ENV TRANSFORMERS_CACHE=""

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Run uvicorn on all interfaces, port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
