# syntax=docker/dockerfile:1.4
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies first (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from the root directory
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache HuggingFace models in a separate layer
RUN python - <<'PYTHON'
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
print("Downloading models...")
# Prompt injection detection model
pipeline("text-classification", model="protectai/deberta-v3-base-prompt-injection-v2")
# Russian toxicity detection model
AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")
AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-toxicity")
# English toxicity detection model
pipeline(task="text-classification", model="minuva/MiniLMv2-toxic-jigsaw", verbose=False)
# Russian spam detection model
AutoTokenizer.from_pretrained("RUSpam/spam_deberta_v4")
AutoModelForSequenceClassification.from_pretrained("RUSpam/spam_deberta_v4")
# English smap detection model
pipeline("text-classification", model="mariagrandury/roberta-base-finetuned-sms-spam-detection", verbose=False)
print("HuggingFace models downloaded and cached successfully.")
PYTHON

# Copy the Russian BERT prompt-injection model from the project root
COPY ru-bert-prompt-injection ./ru-bert-prompt-injection

# Create multiprocess metrics directory
RUN mkdir -p /tmp/prom_metrics

# Copy backend application code last (changes most frequently)
COPY backend ./backend

EXPOSE 8000

# Use exec form for better signal handling
CMD ["sh", "-c", "gunicorn backend.basic:app --workers ${WORKERS:-4} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"] 