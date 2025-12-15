# GTP/GAT - Graph Transformer & Graph Attention Network for Pathology
# Prostate Cancer Grading (PANDA Challenge)

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="Manuel Morteo <mmorteo@bu.edu>"
LABEL description="Graph-based models for prostate cancer grading on PANDA dataset"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    openslide-tools \
    python3-openslide \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project structure
COPY models/ ./models/
COPY utils/ ./utils/
COPY scripts/ ./scripts/
COPY helper.py .
COPY main.py .
COPY main_gat.py .
COPY evaluate.py .
COPY evaluate_gat.py .
COPY inference.py .

# Create directories for data and outputs
RUN mkdir -p /data/input \
             /data/output \
             /data/graphs \
             /app/weights \
             /app/graph_transformer/saved_models \
             /app/graph_transformer/runs \
             /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command - show help
CMD ["python", "inference.py", "--help"]