# Use a lightweight official Python image (debian-slim variant)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS dependencies for faiss, pypdf, sentence-transformers, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies without cache (reduce image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose the port your FastAPI app will listen on
EXPOSE 8000

# Start the app with uvicorn on the correct host and port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
