FROM ubuntu:24.04

# Install Python, R, and basic tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    r-base \
    r-base-dev \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy input data files into container
COPY data/4118e166-34f5-4c1f-9eed-c64b90a3dace.h5ad /app/data/
COPY data/type1_isg_signature.txt /app/data/
WORKDIR /app
