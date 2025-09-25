# Python slim + non-root, siap Cloud Run
FROM python:3.11-slim

# Sysdeps minimal utk pandas/grpc + tini buat signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tini git && \
    rm -rf /var/lib/apt/lists/*

# Default ENV (bisa dioverride di Cloud Run)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    DATASETS_ROOT=/tmp/datasets \
    CHARTS_ROOT=/tmp/charts

# Non-root user
RUN useradd -m -u 1001 -s /bin/bash appuser
WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY . .

# Siapkan staging lokal (ephemeral)
RUN mkdir -p /tmp/datasets /tmp/charts && \
    chown -R appuser:appuser /app /tmp/datasets /tmp/charts
USER appuser

EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["gunicorn","--bind","0.0.0.0:8080","--workers","2","--threads","8","--timeout","240","main:app"]