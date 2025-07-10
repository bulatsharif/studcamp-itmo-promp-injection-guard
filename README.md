# Prompt Injection Guard

Multi-layered defense system for LLM applications that detects prompt injection, toxicity, and spam in user messages. Built for Yandex Studcamp at ITMO 2025.

## Features

- **Prompt Injection Detection**: ML-based detection using BERT models + regex patterns
- **Toxicity Detection**: Support for English and Russian languages
- **Spam Detection**: SMS spam and general content spam filtering
- **Human-in-the-Loop Controls**: Flagging high-risk messages for review
- **Prometheus Metrics**: Real-time monitoring and alerting
- **Grafana Dashboard**: Visualization of defense metrics

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/bulatsharif/studcamp-itmo-promp-injection-guard.git
cd studcamp-itmo-promp-injection-guard
```

### 2. Environment Configuration

Create `.env` file in project root:

```env
# Model Configuration
TOXICITY_THRESHOLD=0.5
SPAM_THRESHOLD=0.5
PROMPT_INJECTION_THRESHOLD=0.5

# OpenRouter API (optional, for fallback)
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=qwen/qwen3-30b-a3b:free

# Prometheus Settings
PROMETHEUS_MULTIPROC_DIR=/tmp/prom_metrics

# Performance Tuning
WORKERS=4
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
```

### 3. Build and Run
DOCKER_BUILDKIT=1 docker compose up --build
```

### 4. Access Services

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## API Usage

### Defense Endpoint

```bash
curl -X POST "http://localhost:8000/defend" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your message here"}'
```

Response:
```json
{
  "is_safe": true,
  "reason": null,
  "scores": {
    "prompt_injection": 0.03,
    "toxicity": 0.02,
    "spam": 0.01
  }
}
```

### Fallback Analysis Endpoint

```bash
curl -X POST "http://localhost:8000/fallback-analyze" \
  -H "Content-Type: application/json" \
  -d '{"user_prompt": "Your message here"}'
```

Response:
```json
{
  "result": "Analysis result from LLM",
  "error": null
}
```

## Hardware Requirements

### Minimum
- **RAM**: 4GB
- **CPU**: 2 cores
- **Storage**: 3GB free space
- **Network**: Internet access for model downloads

### Recommended
- **RAM**: 8GB+
- **CPU**: 4+ cores
- **Storage**: 5GB+ free space
- **GPU**: Optional, for faster inference

### Monitoring

Visit Grafana dashboard at http://localhost:3000 to monitor:
- Request metrics and latency
- Detection rates by type
- Model inference performance
- System resource usage

