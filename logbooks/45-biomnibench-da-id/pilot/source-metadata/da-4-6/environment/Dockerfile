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
COPY data/mmc5.csv /app/data/
WORKDIR /app
