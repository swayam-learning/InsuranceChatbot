# # --- Stage 1: The Build Environment ---
# # This stage installs dependencies and downloads the model
# FROM python:3.10 AS builder

# # Set the working directory
# WORKDIR /app

# # Set the Hugging Face cache directory to a predictable location
# # Your Python code will automatically respect this environment variable
# ENV HF_HOME /app/.cache/huggingface

# # Copy the requirements file first to leverage Docker's cache
# COPY requirements.txt .

# # Install dependencies, ensuring the cache is not in the final image
# RUN pip install --no-cache-dir -r requirements.txt

# # Create the cache directory explicitly
# RUN mkdir -p ${HF_HOME}/hub

# # Download the sentence-transformer model to the new, predictable location
# # The `startup` event in your code will then load it from here.
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# # --- Stage 2: The Final, Slim Runtime Image ---
# # This stage creates a new, much smaller image for running the app
# FROM python:3.10-slim

# # Set the working directory
# WORKDIR /app

# # Set the same Hugging Face cache directory for the runtime
# ENV HF_HOME /app/.cache/huggingface

# # Copy the installed packages from the builder stage to the new image
# COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin

# # Copy the entire Hugging Face model hub from the builder stage
# # This ensures the downloaded model is available at runtime
# COPY --from=builder ${HF_HOME}/hub ${HF_HOME}/hub

# # Copy your application code
# COPY . .

# # Expose the port the app will run on
# EXPOSE 8000

# # Set the command to run the application, using the dynamic PORT env var
# # This is a robust command that works with your __main__ block
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Use the smallest compatible Python base (slim, NOT alpine for faiss/ML)
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Install OS dependencies for faiss, pypdf, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1-dev \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .

# Install Python dependencies, no pip cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only your code (no test data, no PDFs, no models)
COPY . .

# Expose the port your app runs on (Railway auto-detects 8000)
EXPOSE 8000

# Default command for Railway (adjust file if needed)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
