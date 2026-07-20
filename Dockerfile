# Use Python slim image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install build tools and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y build-essential gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of your application code
COPY . .

# Use runtime environment variables for ClearML secrets
CMD ["python", "src/train.py"]
