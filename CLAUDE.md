# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a prompt injection and fraud detection system for LLM applications, developed as part of Yandex Studcamp at ITMO 2025. The system implements multi-layered defense against prompt injection attacks, toxicity detection, and fraudulent content using both rule-based and ML-based approaches.

## Architecture

### Core Components

- **backend/basic.py**: Main FastAPI application with unified defense system
  - `UnifiedMessageDefense`: Main defense orchestrator combining all protection layers
  - `PromptInjectionFilter`: Detects prompt injection via regex patterns and ML models
  - `ToxicityDetector`: Multilingual toxicity detection (English/Russian)
  - `OutputValidator`: Validates and filters LLM outputs
  - `HITLController`: Human-in-the-loop controls for high-risk content

- **backend/fallback.py**: Fallback analysis service using OpenRouter API
- **backend/metrics.py**: Prometheus metrics collection and monitoring
- **ru-bert-prompt-injection/**: Fine-tuned Russian BERT model for prompt injection detection

### Language Support

The system supports both English and Russian languages:
- **English**: Uses protectai/deberta-v3-base-prompt-injection-v2 for prompt injection
- **Russian**: Uses custom fine-tuned ru-bert-prompt-injection model
- **Toxicity**: cointegrated/rubert-tiny-toxicity (Russian), minuva/MiniLMv2-toxic-jigsaw (English)

## Development Commands

### Docker (Recommended)
```bash
# Build and run the complete stack
DOCKER_BUILDKIT=1 docker compose up --build

# Run in detached mode
DOCKER_BUILDKIT=1 docker compose up --build -d

# Stop services
docker compose down
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the backend server
uvicorn backend.basic:app --reload --host 0.0.0.0 --port 8000

# Run with multiple workers (production-like)
uvicorn backend.basic:app --workers 4 --loop uvloop --http h11 --host 0.0.0.0 --port 8000
```

### Testing
The system currently uses pytest for testing. Run tests with:
```bash
pytest
```

## API Endpoints

### Main Defense Endpoint
- `POST /defend`: Main content filtering endpoint
  - Input: `{"text": "user message", "language": "en|ru"}`
  - Returns comprehensive safety analysis

### Fallback Analysis
- `POST /fallback-analyze`: LLM-based fallback analysis
  - Input: `{"user_prompt": "text to analyze"}`
  - Uses OpenRouter API for additional analysis

### Monitoring
- `GET /metrics`: Prometheus metrics endpoint
- Grafana dashboard available at `http://localhost:3000` (admin/admin)
- Prometheus at `http://localhost:9090`

## Model Optimization

The system uses several performance optimizations:
- **Torch Compilation**: PyTorch 2.0+ graph compilation with `torch.compile()`
- **BetterTransformer**: Optimized transformer implementations
- **Fast Tokenizers**: Rust-backed tokenizers for better throughput
- **GPU Support**: Automatic CUDA detection and model placement
- **Multi-process Metrics**: Prometheus metrics work across multiple workers

## Environment Variables

Key environment variables for configuration:
- `PROMETHEUS_MULTIPROC_DIR`: Directory for multi-process metrics
- `MODEL_CACHE_DIR`: HuggingFace model cache directory
- `WORKERS`: Number of Uvicorn workers
- `OMP_NUM_THREADS`: OpenMP thread count
- `MKL_NUM_THREADS`: Intel MKL thread count
- `OPEN_ROUTER_API_KEY`: API key for fallback analysis

## Security Features

### Multi-layered Defense
1. **Regex-based Detection**: Pattern matching for common injection attempts
2. **ML-based Detection**: Transformer models for sophisticated attacks
3. **Toxicity Filtering**: Content toxicity assessment
4. **Output Validation**: LLM response sanitization
5. **Human-in-the-Loop**: High-risk content routing

### Monitoring and Metrics
- Request latency tracking
- Detection method performance
- Language-specific metrics
- Safety score distributions
- Blocked content categorization