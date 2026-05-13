FROM python:3.11-slim

# espeak-ng is required by kokoro's English phonemiser
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (CPU-only torch keeps the image ~1.5 GB instead of ~5 GB)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spacy English model (needed by kokoro's misaki phonemiser)
RUN python -m spacy download en_core_web_sm

COPY app.py .

# HuggingFace model cache — mount a volume here to avoid re-downloading
ENV HF_HOME=/cache/huggingface
ENV PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
