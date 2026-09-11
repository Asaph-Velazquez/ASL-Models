# Imagen reproducible del pipeline ASL hotelero (entrena en GPU NVIDIA; en Mac solo smoke en CPU).
FROM python:3.10-slim

# Dependencias de sistema para MediaPipe / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Usuario no-root; los directorios montados se crean con permisos abiertos
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/checkpoints /app/reports /app/outputs /app/external /app/dataset \
    && chown -R appuser:appuser /app

ENV PYTHONPATH=/app \
    HF_HOME=/app/external/hf_cache \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONUNBUFFERED=1

USER appuser
