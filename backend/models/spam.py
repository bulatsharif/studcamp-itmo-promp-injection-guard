import time
import torch
from typing import Any, Optional
from backend.models.base import BaseDetector
from backend.config import settings
import backend.metrics as metrics


class SpamDetector(BaseDetector):
    """Spam detection for both Russian and English text."""
    
    def _init_models(self):
        """Initialize spam detection models for both languages."""
        # Russian spam detection model
        try:
            self.ru_pipeline = self._load_pipeline(
                settings.ru_spam_model,
                task="text-classification",
                return_all_scores=True
            )
            self.models["ru"] = self.ru_pipeline
            print("Russian spam model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Russian spam model: {e}")
            self.models["ru"] = None
        
        # English spam detection model
        try:
            self.en_pipeline = self._load_pipeline(
                settings.en_spam_model,
                task="text-classification",
                return_all_scores=True
            )
            self.models["en"] = self.en_pipeline
            print("English spam model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load English spam model: {e}")
            self.models["en"] = None
    
    def text2spam_ru(self, text: str) -> float:
        """Predict spam probability for Russian text."""
        text = text.lower()
        
        with torch.no_grad():
            pipe_result = self.models["ru"](text)
        
        # Get spam probability (assuming index 1 is spam class)
        proba_spam = pipe_result[0][1]["score"]
        return proba_spam
    
    def text2spam_en(self, text: str) -> float:
        """Predict spam probability for English text."""
        text = text.lower()
        
        with torch.no_grad():
            pipe_result = self.models["en"](text)
        
        # Get spam probability (assuming index 1 is spam class)
        proba_spam = pipe_result[0][1]["score"]
        return proba_spam
    
    def predict(self, text: str, language: Optional[str] = None, 
                spam_threshold: Optional[float] = None, **kwargs) -> dict[str, Any]:
        """Predict spam for given text."""
        if language is None:
            language = self.detect_language(text)
        
        if spam_threshold is None:
            spam_threshold = settings.spam_threshold
        
        start_t = time.perf_counter()
        
        try:
            if language == "ru" and self.models["ru"] is not None:
                spam_score = float(self.text2spam_ru(text))
            elif language == "en" and self.models["en"] is not None:
                spam_score = float(self.text2spam_en(text))
            else:
                return self._create_error_result(
                    text, 
                    f"Unsupported language or model not available: {language}",
                    f"spam-{language}"
                )
            
            # Track metrics
            metrics.SPAM_ML_LATENCY.labels(language).observe(time.perf_counter() - start_t)
            
            is_spam = spam_score > spam_threshold
            if is_spam:
                metrics.SPAM_DETECTIONS_TOTAL.labels(language).inc()
            
            return {
                "text": text,
                "language": language,
                "spam_score": spam_score,
                "is_spam": is_spam,
                "detected": is_spam,
                "confidence": spam_score,
                "model_used": f"spam-{language}"
            }
            
        except Exception as e:
            metrics.SPAM_ML_LATENCY.labels(language).observe(time.perf_counter() - start_t)
            return self._create_error_result(text, str(e), f"spam-{language}")
    
    def predict_spam(self, text: str, language: Optional[str] = None, 
                    spam_threshold: Optional[float] = None) -> dict[str, Any]:
        """Alias for predict method for backward compatibility."""
        return self.predict(text, language=language, spam_threshold=spam_threshold)