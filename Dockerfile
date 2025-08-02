# --- Stage 1: The Build Environment ---
FROM python:3.10 AS builder

# Set the working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker's cache
COPY requirements.txt .

# Install dependencies, but don't cache them in the final image
RUN pip install --no-cache-dir -r requirements.txt

# Download the sentence-transformer model during the build
# This command stores the model in /root/.cache/torch/sentence_transformers/
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


# --- Stage 2: The Final, Slim Runtime Image ---
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the installed packages from the builder stage to the new image
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Corrected COPY command:
# Copy the specific model cache directory from the builder stage
# The path is now correct and will be found.
COPY --from=builder /root/.cache/torch/sentence_transformers /root/.cache/torch/sentence_transformers

# Copy your application code
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]