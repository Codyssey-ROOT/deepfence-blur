FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/deepfence \
    HF_HOME=/models/huggingface \
    YOLO_CONFIG_DIR=/models/ultralytics \
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates fonts-nanum libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

RUN useradd -m -u 10001 deepfence \
    && mkdir -p /models /app \
    && chown -R deepfence:deepfence /models /app /home/deepfence
COPY --chown=deepfence:deepfence app ./app
COPY --chown=deepfence:deepfence scripts ./scripts
USER deepfence

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
