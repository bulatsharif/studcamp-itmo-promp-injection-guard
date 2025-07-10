import time
from fastapi import APIRouter, Depends, Request
from backend.schemas import MessageRequest, DefenseResponse
from backend.dependencies import get_defense_system
from backend.services.defense import UnifiedMessageDefense
from backend.services.language import language_detector
import backend.metrics as metrics


router = APIRouter()


@router.post("/defend", response_model=DefenseResponse, tags=["defense"])
def defend_message(
    message_request: MessageRequest,
    request: Request,
    defense_system: UnifiedMessageDefense = Depends(get_defense_system)
):
    """
    Analyze message for safety using multi-layered defense system.
    
    This endpoint processes text through:
    - Prompt injection detection (regex + ML)
    - Toxicity detection (multilingual)
    - Spam detection (multilingual)
    - Human-in-the-loop controls
    """
    start_t = time.perf_counter()
    
    # Detect language for the request
    detected_language = language_detector.detect_language(message_request.text)
    
    # Process message through defense system
    result = defense_system.process_message(message_request.text, language=detected_language)
    
    # Track processing time
    elapsed = time.perf_counter() - start_t
    metrics.REQUEST_LATENCY.labels(detected_language).observe(elapsed)
    metrics.REQUESTS_BY_LANGUAGE.labels(detected_language).inc()
    
    # Create simplified response
    response = DefenseResponse(
        is_safe=result["is_safe"],
        reason=result["rejection_reason"],
        scores=result["safety_scores"]
    )
    
    # Track metrics based on result
    if result["is_safe"]:
        status = "success"
        metrics.SAFE_MESSAGES_TOTAL.inc()
        metrics.SAFE_UNSAFE_MESSAGES_TOTAL.labels("safe").inc()
    else:
        status = "rejected"
        reason = result.get("rejection_reason", "unknown").lower()
        
        # Categorize rejection reason
        if "prompt" in reason:
            reason_label = "prompt_injection"
        elif "toxic" in reason:
            reason_label = "toxicity"
        elif "spam" in reason:
            reason_label = "spam"
        else:
            reason_label = "other"
        
        metrics.REJECTED_MESSAGES_TOTAL.labels(reason_label).inc()
        metrics.SAFE_UNSAFE_MESSAGES_TOTAL.labels("unsafe").inc()
        metrics.BLOCKED_MESSAGES_BY_LABEL.labels(reason_label).inc()
        metrics.BLOCKED_REQUESTS_BY_LANGUAGE.labels(detected_language).inc()
    
    metrics.REQUESTS_TOTAL.labels(status).inc()
    
    return response