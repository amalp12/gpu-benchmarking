# Lightweight CUDA base image (no heavy libs)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set Python version
ENV DEBIAN_FRONTEND=noninteractive

# Install only required packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy only requirements first (layer caching)
COPY requirements.txt /app/requirements.txt

# Install Python deps (CPU-friendly, GPU features via CUDA runtime)
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy repo contents
COPY . /app/

# Default command (can be overridden)
CMD ["python3"]()
