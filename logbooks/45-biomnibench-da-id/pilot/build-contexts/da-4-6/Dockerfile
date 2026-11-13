FROM ubuntu:24.04

# Install Python, R, and basic tools
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Options::=--force-confnew install -y \
    python3 \
    python3-pip \
    python3-venv \
    r-base \
    r-base-dev \
    curl \
    wget \
    git \
    ca-certificates \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

# Copy input data files into container
COPY data/mmc5.csv /app/data/
WORKDIR /app

# Explicit development harness pin; record installed package versions.
COPY --from=node:22-bookworm-slim /usr/local/ /usr/local/
RUN npm install -g @earendil-works/pi-coding-agent@0.87.0 && pi --version
RUN python3 -m venv /opt/biomnibench-judge && /opt/biomnibench-judge/bin/pip install --no-cache-dir openai==2.24.0
