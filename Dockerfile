# --- Stage 1: The Build Environment ---
# This stage installs all dependencies and downloads the model.
# It uses a slightly larger base image with build tools.
FROM python:3.10 AS builder

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt .

# Install dependencies, but don't cache them in the final image
RUN pip install --no-cache-dir -r requirements.txt

# Download the sentence-transformer model during the build
# This ensures it's part of the image and not downloaded at runtime.
# This model will be in ~/.cache/torch/sentence_transformers/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# --- Stage 2: The Final, Slim Runtime Image ---
# This stage creates a new, much smaller image.
# We'll only copy what's absolutely necessary from the builder stage.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the installed packages from the builder stage to the new image
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the pre-downloaded sentence-transformer model cache
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy your application code
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]