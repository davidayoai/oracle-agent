FROM python:3.12-slim@sha256:1a84f37bc4439fa3487431697dd60a6c23062319efae7fc52e7960fc5d4c82c6

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20230311 \
    procps=2:3.3.17-9 \
    ripgrep=13.0.0-3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.lock /tmp/requirements.lock
COPY task_inputs/ /app/environment/task_inputs/
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

COPY . /app