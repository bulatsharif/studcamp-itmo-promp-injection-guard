# syntax=docker/dockerfile:1.4
FROM python:3.10-slim AS base

WORKDIR /app

# 1) Install system dependencies (e.g. build tools, ca-certs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2) Copy & install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3) Pre-download HF models into a BuildKit cache mount, then bake them into /opt/hf_models
#    We set ALL of the Hugging Face cache env-vars so nothing ever lands in /root/.cache:
ENV HUGGINGFACE_HUB_CACHE=${MODEL_CACHE_DIR:-/opt/hf_models}
ENV HF_HOME=${MODEL_CACHE_DIR:-/opt/hf_models}
ENV XDG_CACHE_HOME=${MODEL_CACHE_DIR:-/opt/hf_models}

RUN --mount=type=cache,target=/hf-cache \
    bash -euxo pipefail <<'BASH'
# verify our cache mount is writable
df -h /hf-cache

# Download models via a tiny inline Python script,
# explicitly telling HF to use /hf-cache for everything
python - <<'PYCODE'
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

cache_dir = "/hf-cache"
print(f"Downloading all models into {cache_dir} …")

pipeline(
    "text-classification",
    model="protectai/deberta-v3-base-prompt-injection-v2",
    cache_dir=cache_dir
)
AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny-toxicity",
    cache_dir=cache_dir
)
AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny-toxicity",
    cache_dir=cache_dir
)
pipeline(
    "text-classification",
    model="minuva/MiniLMv2-toxic-jigsaw",
    cache_dir=cache_dir,
    verbose=False
)
pipeline(
    "text-classification",
    model="RUSpam/spam_deberta_v4",
    cache_dir=cache_dir,
    verbose=False
)
pipeline(
    "text-classification",
    model="mariagrandury/roberta-base-finetuned-sms-spam-detection",
    cache_dir=cache_dir,
    verbose=False
)

print("All models downloaded.")
PYCODE

# Copy the fully-populated cache into an image layer
mkdir -p /opt/hf_models
cp -a /hf-cache/. /opt/hf_models/
echo "✅ Models baked"
BASH

# 4) Copy any extra model files from your context
COPY ru-bert-prompt-injection ./ru-bert-prompt-injection

# 5) Prepare metrics dir
RUN mkdir -p /tmp/prom_metrics

# 6) Copy application code
COPY backend ./backend

EXPOSE 8000

# 7) Runtime env vars
ENV OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
ENV MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# 8) Launch with Uvicorn
CMD ["sh", "-c", "uvicorn backend.basic:app --workers ${WORKERS:-4} --loop uvloop --http h11 --host 0.0.0.0 --port 8000"]
