from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
# --- Multi-process Prometheus setup ---
import os
from prometheus_client import REGISTRY, CollectorRegistry, multiprocess


# If multiprocess mode is enabled (e.g., under Gunicorn), collect from all workers
if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
    _registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(_registry)
else:
    _registry = REGISTRY

from fastapi import Response, APIRouter

REQUESTS_TOTAL = Counter(
    "defense_requests_total",
    "Total number of /defend requests processed",
    labelnames=["status"],
)

REQUEST_LATENCY = Histogram(
    "defense_request_processing_seconds",
    "Latency of /defend endpoint in seconds",
)

SAFE_MESSAGES_TOTAL = Counter("defense_messages_safe_total", "Messages marked safe")

REJECTED_MESSAGES_TOTAL = Counter(
    "defense_messages_rejected_total",
    "Messages rejected by reason",
    labelnames=["reason"],
)

HUMAN_REVIEW_TOTAL = Counter(
    "defense_human_review_required_total", "Messages routed for human review"
)

PROMPT_INJECTION_DETECTIONS_TOTAL = Counter(
    "prompt_injection_detections_total",
    "Prompt-injection detections by method",
    labelnames=["method"],
)

TOXICITY_DETECTIONS_TOTAL = Counter(
    "toxicity_detections_total",
    "Toxic messages detected by language",
    labelnames=["language"],
)

TOXICITY_SCORE_HIST = Histogram(
    "toxicity_score_histogram",
    "Distribution of toxicity scores",
    buckets=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

INJECTION_ML_LATENCY = Histogram(
    "prompt_injection_ml_inference_seconds",
    "Latency of prompt-injection ML inference in seconds",
    labelnames=["language"],
)

TOXICITY_ML_LATENCY = Histogram(
    "toxicity_ml_inference_seconds",
    "Latency of toxicity ML inference in seconds",
    labelnames=["language"],
)

# New: Blocked messages by label (category)
BLOCKED_MESSAGES_BY_LABEL = Counter(
    "blocked_messages_by_label_total",
    "Total blocked messages by label/category (e.g., prompt_injection, toxicity, spam)",
    labelnames=["label"],
)

# New: All requests per language
REQUESTS_BY_LANGUAGE = Counter(
    "requests_by_language_total",
    "Total requests processed per language",
    labelnames=["language"],
)

# New: Blocked requests per language
BLOCKED_REQUESTS_BY_LANGUAGE = Counter(
    "blocked_requests_by_language_total",
    "Total blocked requests per language",
    labelnames=["language"],
)

# New: Safe/unsafe pie
SAFE_UNSAFE_MESSAGES_TOTAL = Counter(
    "safe_unsafe_messages_total",
    "Total messages by safety verdict",
    labelnames=["verdict"],
)

router = APIRouter()


@router.get("/metrics", tags=["metrics"])
def metrics():
    return Response(generate_latest(_registry), media_type=CONTENT_TYPE_LATEST)
