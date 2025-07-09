from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import re
from typing import Optional
import time
# Remove asyncio/threadpool imports - relying on FastAPI default thread pool
import os
import backend.metrics as metrics

# FastAPI will offload sync endpoints to its own ThreadPoolExecutor, so no custom pool is needed.


class PromptInjectionFilter:
    def __init__(self):
        self.dangerous_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
            r"system\s+override",
            r"reveal\s+prompt",
        ]
        self.fuzzy_patterns = [
            "ignore",
            "bypass",
            "override",
            "reveal",
            "delete",
            "system",
        ]
        self._init_ml_model()

    def _init_ml_model(self):
        try:
            self.ml_classifier = pipeline(
                "text-classification",
                model="protectai/deberta-v3-base-prompt-injection-v2",
            )
            self.ml_model_available = True
        except Exception as e:
            print(f"Warning: Could not load ML prompt injection model: {e}")
            print("Falling back to regex-only detection")
            self.ml_model_available = False
            self.ml_classifier = None

    def detect_injection_regex(self, text: str) -> dict:
        result = {"detected": False, "patterns_matched": [], "fuzzy_matches": []}
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["detected"] = True
                result["patterns_matched"].append(pattern)
                metrics.PROMPT_INJECTION_DETECTIONS_TOTAL.labels("regex").inc()
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            for pattern in self.fuzzy_patterns:
                if self._is_similar_word(word, pattern):
                    result["detected"] = True
                    result["fuzzy_matches"].append({"word": word, "pattern": pattern})
                    metrics.PROMPT_INJECTION_DETECTIONS_TOTAL.labels("regex").inc()
        return result

    def detect_injection_ml(self, text: str) -> dict:
        start_t = time.perf_counter()
        if not self.ml_model_available:
            return {
                "detected": False,
                "confidence": 0.0,
                "label": "SAFE",
                "error": "ML model not available",
            }
        # -------------------------------------------------------------------
        # Record inference latency regardless of outcome
        # -------------------------------------------------------------------
        try:
            result = self.ml_classifier(text)
            prediction = result[0] if isinstance(result, list) else result
            label = prediction["label"]
            confidence = prediction["score"]
            is_injection = label in ["INJECTION", "MALICIOUS", "1"] or (
                label == "LABEL_1" and confidence > 0.5
            )
            # Metric: total detections per method
            if is_injection:
                metrics.PROMPT_INJECTION_DETECTIONS_TOTAL.labels("ml").inc()
            return {
                "detected": is_injection,
                "confidence": confidence,
                "label": label,
                "raw_prediction": prediction,
            }
        except Exception as e:
            return {
                "detected": False,
                "confidence": 0.0,
                "label": "ERROR",
                "error": str(e),
            }
        finally:
            metrics.INJECTION_ML_LATENCY.observe(time.perf_counter() - start_t)

    def detect_injection(self, text: str) -> dict:
        regex_result = self.detect_injection_regex(text)
        ml_result = self.detect_injection_ml(text)
        combined_result = {
            "text": text,
            "detected": regex_result["detected"] or ml_result["detected"],
            "regex_detection": regex_result,
            "ml_detection": ml_result,
            "detection_methods": [],
        }
        if regex_result["detected"]:
            combined_result["detection_methods"].append("regex")
        if ml_result["detected"]:
            combined_result["detection_methods"].append("ml")
        return combined_result

    def _is_similar_word(self, word: str, target: str) -> bool:
        if len(word) != len(target) or len(word) < 3:
            return False
        return (
            word[0] == target[0]
            and word[-1] == target[-1]
            and sorted(word[1:-1]) == sorted(target[1:-1])
        )

    def sanitize_input(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(.)\1{3,}", r"\1", text)
        for pattern in self.dangerous_patterns:
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
        return text[:10000]


class OutputValidator:
    def __init__(self):
        self.suspicious_patterns = [
            r"SYSTEM\s*[:]\s*You\s+are",
            r"API[_\s]KEY[:=]\s*\w+",
            r"instructions?[:]\s*\d+\.",
        ]

    def validate_output(self, output: str) -> bool:
        return not any(
            re.search(pattern, output, re.IGNORECASE)
            for pattern in self.suspicious_patterns
        )

    def filter_response(self, response: str) -> str:
        if not self.validate_output(response) or len(response) > 5000:
            return "I cannot provide that information for security reasons."
        return response


class HITLController:
    def __init__(self):
        self.high_risk_keywords = [
            "password",
            "api_key",
            "admin",
            "system",
            "bypass",
            "override",
        ]

    def requires_approval(self, user_input: str) -> bool:
        risk_score = sum(
            1 for keyword in self.high_risk_keywords if keyword in user_input.lower()
        )
        injection_patterns = ["ignore instructions", "developer mode", "reveal prompt"]
        risk_score += sum(
            2 for pattern in injection_patterns if pattern in user_input.lower()
        )
        return risk_score >= 3


class ToxicityDetector:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._load_models()

    def _load_models(self):
        model_checkpoint_ru = "cointegrated/rubert-tiny-toxicity"
        self.tokenizers["ru"] = AutoTokenizer.from_pretrained(model_checkpoint_ru)
        self.models["ru"] = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint_ru
        )
        model_checkpoint_en = "minuva/MiniLMv2-toxic-jigsaw"
        self.tokenizers["en"] = None
        self.models["en"] = pipeline(
            model="minuva/MiniLMv2-toxic-jigsaw",
            task="text-classification",
            verbose=False,
        )
        if torch.cuda.is_available():
            for model in self.models.values():
                model.cuda()

    def detect_language(self, text):
        cyrillic_pattern = re.compile(r"[а-яё]", re.IGNORECASE)
        if cyrillic_pattern.search(text):
            return "ru"
        return "en"

    def text2toxicity_ru(self, text, aggregate=True):
        text = text.lower()
        with torch.no_grad():
            inputs = self.tokenizers["ru"](
                text, return_tensors="pt", truncation=True, padding=True
            ).to(self.models["ru"].device)
            proba = torch.sigmoid(self.models["ru"](**inputs).logits).cpu().numpy()
        if isinstance(text, str):
            proba = proba[0]
        if aggregate:
            return 1 - proba.T[0] * (1 - proba.T[-1])
        return proba

    def text2toxicity_en(self, text, aggregate=True):
        with torch.no_grad():
            text = text.lower()
            pipe_result = self.models["en"](text)
        return pipe_result[0]["score"]

    def predict_toxicity(self, text, language=None):
        if language is None:
            language = self.detect_language(text)
        start_t = time.perf_counter()
        if language == "ru":
            toxicity_score = float(self.text2toxicity_ru(text))
        elif language == "en":
            toxicity_score = float(self.text2toxicity_en(text))
        else:
            raise ValueError(f"Unsupported language: {language}")
        metrics.TOXICITY_ML_LATENCY.labels(language).observe(
            time.perf_counter() - start_t
        )
        threshold = 0.5
        is_toxic = toxicity_score > threshold
        if is_toxic:
            metrics.TOXICITY_DETECTIONS_TOTAL.labels(language).inc()
        metrics.TOXICITY_SCORE_HIST.observe(toxicity_score)
        return {
            "text": text,
            "language": language,
            "toxicity_score": toxicity_score,
            "is_toxic": is_toxic,
        }

    def batch_predict(self, texts, language=None):
        results = []
        for text in texts:
            try:
                result = self.predict_toxicity(text, language)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "text": text,
                        "error": str(e),
                        "toxicity_score": None,
                        "is_toxic": None,
                    }
                )
        return results


class UnifiedMessageDefense:
    def __init__(self):
        self.prompt_injection_filter = PromptInjectionFilter()
        self.toxicity_detector = ToxicityDetector()
        self.hitl_controller = HITLController()
        self.toxicity_threshold = 0.5
        self.enable_prompt_injection_detection = True
        self.enable_toxicity_detection = True
        self.enable_hitl_control = True

    def process_message(self, message: str, language: Optional[str] = None) -> dict:
        result = {
            "original_message": message,
            "filtered_message": message,
            "is_safe": True,
            "rejection_reason": None,
            "safety_scores": {},
            "requires_human_review": False,
        }
        if self.enable_prompt_injection_detection:
            injection_result = self.prompt_injection_filter.detect_injection(message)
            result["safety_scores"]["prompt_injection"] = injection_result
            if injection_result["detected"]:
                result["is_safe"] = False
                detection_methods = ", ".join(injection_result["detection_methods"])
                result["rejection_reason"] = (
                    f"Prompt injection detected via {detection_methods}"
                )
                return result
            result["filtered_message"] = self.prompt_injection_filter.sanitize_input(
                message
            )
        if self.enable_toxicity_detection:
            try:
                toxicity_result = self.toxicity_detector.predict_toxicity(
                    result["filtered_message"], language
                )
                print(toxicity_result)
                result["safety_scores"]["toxicity"] = toxicity_result["toxicity_score"]
                result["safety_scores"]["language"] = toxicity_result["language"]
                if toxicity_result["is_toxic"]:
                    result["is_safe"] = False
                    result["rejection_reason"] = (
                        f"Toxic content detected (score: {toxicity_result['toxicity_score']:.3f})"
                    )
                    return result
            except Exception as e:
                result["safety_scores"]["toxicity_error"] = str(e)
        if self.enable_hitl_control:
            requires_approval = self.hitl_controller.requires_approval(
                result["filtered_message"]
            )
            result["requires_human_review"] = requires_approval
            if requires_approval:
                result["is_safe"] = False
                result["rejection_reason"] = "High-risk content requires human review"
                return result
        return result


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
app.include_router(metrics.router)

defense_system = UnifiedMessageDefense()


class MessageRequest(BaseModel):
    text: str
    language: Optional[str] = None


@app.post("/defend", tags=["defense"])
def defend_message(request: MessageRequest):
    start_t = time.perf_counter()
    result = defense_system.process_message(request.text)
    elapsed = time.perf_counter() - start_t

    metrics.REQUEST_LATENCY.observe(elapsed)

    # Track language for all requests (from result or request)
    language = None
    if "safety_scores" in result and "language" in result["safety_scores"]:
        language = result["safety_scores"]["language"]
    elif hasattr(request, "language") and request.language:
        language = request.language
    else:
        language = "unknown"
    metrics.REQUESTS_BY_LANGUAGE.labels(language).inc()

    if result["is_safe"]:
        STATUS = "success"
        metrics.SAFE_MESSAGES_TOTAL.inc()
        metrics.SAFE_UNSAFE_MESSAGES_TOTAL.labels("safe").inc()
    else:
        STATUS = "rejected"
        reason = result.get("rejection_reason", "unknown").lower()
        if "prompt" in reason:
            REASON_LABEL = "prompt_injection"
        elif "toxic" in reason:
            REASON_LABEL = "toxicity"
        elif "human review" in reason:
            REASON_LABEL = "human_review"
        else:
            REASON_LABEL = "other"
        metrics.REJECTED_MESSAGES_TOTAL.labels(REASON_LABEL).inc()
        metrics.SAFE_UNSAFE_MESSAGES_TOTAL.labels("unsafe").inc()
        metrics.BLOCKED_MESSAGES_BY_LABEL.labels(REASON_LABEL).inc()
        metrics.BLOCKED_REQUESTS_BY_LANGUAGE.labels(language).inc()

    metrics.REQUESTS_TOTAL.labels(STATUS).inc()

    if result.get("requires_human_review"):
        metrics.HUMAN_REVIEW_TOTAL.inc()

    return result
