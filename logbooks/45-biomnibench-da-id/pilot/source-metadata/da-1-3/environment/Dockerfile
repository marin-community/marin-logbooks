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
COPY data/GSE236581_CRC-ICB_metadata.txt /app/data/
COPY data/GSE236581_barcodes.tsv /app/data/
COPY data/GSE236581_counts.mtx /app/data/
COPY data/GSE236581_features.tsv /app/data/
COPY data/clinical.xlsx /app/data/
WORKDIR /app
