import time
import torch
from typing import Any, Optional
from backend.models.base import BaseDetector
from backend.config import settings
import backend.metrics as metrics


class ToxicityDetector(BaseDetector):
    """Toxicity detection for both Russian and English text."""
    
    def _init_models(self):
        """Initialize toxicity models for both languages."""
        # Russian toxicity model
        try:
            self.ru_model, self.ru_tokenizer = self._load_model_and_tokenizer(
                settings.ru_toxicity_model, use_cuda=True
            )
            self.models["ru"] = self.ru_model
            self.tokenizers["ru"] = self.ru_tokenizer
            print("Russian toxicity model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Russian toxicity model: {e}")
            self.models["ru"] = None
            self.tokenizers["ru"] = None
        
        # English toxicity model (pipeline)
        try:
            self.en_pipeline = self._load_pipeline(
                settings.en_toxicity_model,
                task="text-classification"
            )
            self.models["en"] = self.en_pipeline
            print("English toxicity model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load English toxicity model: {e}")
            self.models["en"] = None
    
    def text2toxicity_ru(self, text: str, aggregate: bool = True) -> float:
        """Predict toxicity for Russian text."""
        text = text.lower()
        
        with torch.no_grad():
            inputs = self.tokenizers["ru"](
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(self.models["ru"].device)
            
            proba = torch.sigmoid(self.models["ru"](**inputs).logits).cpu().numpy()
        
        if isinstance(text, str):
            proba = proba[0]
        
        if aggregate:
            return 1 - proba.T[0] * (1 - proba.T[-1])
        return proba
    
    def text2toxicity_en(self, text: str, aggregate: bool = True) -> float:
        """Predict toxicity for English text."""
        with torch.no_grad():
            text = text.lower()
            pipe_result = self.models["en"](text)
        
        return pipe_result[0]["score"]
    
    def predict(self, text: str, language: Optional[str] = None, **kwargs) -> dict[str, Any]:
        """Predict toxicity for given text."""
        if language is None:
            language = self.detect_language(text)
        
        start_t = time.perf_counter()
        
        try:
            if language == "ru" and self.models["ru"] is not None:
                toxicity_score = float(self.text2toxicity_ru(text))
            elif language == "en" and self.models["en"] is not None:
                toxicity_score = float(self.text2toxicity_en(text))
            else:
                return self._create_error_result(
                    text, 
                    f"Unsupported language or model not available: {language}",
                    f"toxicity-{language}"
                )
            
            # Track metrics
            metrics.TOXICITY_ML_LATENCY.labels(language).observe(time.perf_counter() - start_t)
            
            is_toxic = toxicity_score > settings.toxicity_threshold
            if is_toxic:
                metrics.TOXICITY_DETECTIONS_TOTAL.labels(language).inc()
            
            metrics.TOXICITY_SCORE_HIST.observe(toxicity_score)
            
            return {
                "text": text,
                "language": language,
                "toxicity_score": toxicity_score,
                "is_toxic": is_toxic,
                "detected": is_toxic,
                "confidence": toxicity_score,
                "model_used": f"toxicity-{language}"
            }
            
        except Exception as e:
            metrics.TOXICITY_ML_LATENCY.labels(language).observe(time.perf_counter() - start_t)
            return self._create_error_result(text, str(e), f"toxicity-{language}")
    
    def predict_toxicity(self, text: str, language: Optional[str] = None) -> dict[str, Any]:
        """Alias for predict method for backward compatibility."""
        return self.predict(text, language=language)