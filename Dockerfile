# Python slim + non-root, siap untuk Cloud Run (port 8080)
FROM python:3.11-slim

# System deps yang umum dipakai Pandas/Plotly
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl tini && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

# Buat user non-root
RUN useradd -m appuser
WORKDIR /app

# (Opsional) kalau kamu punya requirements.txt, ini lebih cepat & reproducible:
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# # Tanpa requirements.txt, install paket yang dipakai di app.py:
# RUN pip install --upgrade pip && pip install \
#     gunicorn \
#     flask \
#     flask-cors \
#     pandas \
#     python-dotenv \
#     litellm \
#     pandasai \
#     pandasai-litellm \
#     duckdb \
#     plotly

# Copy source code
COPY . .

# Pastikan folder lokal (untuk mode LOCAL DEV) tersedia; di Cloud Run ini ephemeral.
RUN mkdir -p datasets charts && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Gunicorn entrypoint (Flask app = app:app di app.py)
# Tini membantu menangani signal (graceful shutdown)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "8", "--timeout", "300", "app:app"]
