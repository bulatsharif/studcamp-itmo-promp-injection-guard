import re
import os
import time
import torch
from typing import Any
from backend.models.base import BaseDetector
from backend.config import settings
import backend.metrics as metrics


class PromptInjectionFilter(BaseDetector):
    """Prompt injection detection using both regex patterns and ML models."""
    
    def _init_models(self):
        """Initialize both English and Russian prompt injection models."""
        self.en_model_available = False
        self.ru_model_available = False
        
        # Initialize English model
        try:
            self.en_classifier = self._load_pipeline(
                settings.en_prompt_injection_model,
                task="text-classification"
            )
            self.en_model_available = True
            print("English prompt injection model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load English prompt injection model: {e}")
            self.en_classifier = None
        
        # Initialize Russian model
        try:
            if os.path.exists(settings.ru_bert_model_path):
                self.ru_model, self.ru_tokenizer = self._load_model_and_tokenizer(
                    settings.ru_bert_model_path, use_cuda=True
                )
                self.ru_model_available = True
                print("Russian prompt injection model loaded successfully")
            else:
                raise FileNotFoundError(f"Russian model not found at {settings.ru_bert_model_path}")
        except Exception as e:
            print(f"Warning: Could not load Russian prompt injection model: {e}")
            print("Falling back to English model for Russian text")
            self.ru_model = None
            self.ru_tokenizer = None
        
        # set overall model availability
        self.ml_model_available = self.en_model_available or self.ru_model_available
        
        if not self.ml_model_available:
            print("Warning: No ML models available, falling back to regex-only detection")
    
    def predict_russian_injection(self, text: str) -> dict[str, Any]:
        """Predict prompt injection using the fine-tuned Russian BERT model."""
        try:
            # Tokenize input
            inputs = self.ru_tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Move to same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.ru_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
            # Convert to CPU for processing
            probabilities = probabilities.cpu().numpy()[0]
            
            # Assuming binary classification: [SAFE, INJECTION]
            injection_probability = float(probabilities[1])  # Index 1 for injection class
            safe_probability = float(probabilities[0])       # Index 0 for safe class
            
            # Determine prediction
            is_injection = injection_probability > settings.prompt_injection_threshold
            confidence = injection_probability if is_injection else safe_probability
            label = "INJECTION" if is_injection else "SAFE"
            
            return {
                'detected': is_injection,
                'confidence': confidence,
                'label': label,
                'injection_probability': injection_probability,
                'safe_probability': safe_probability,
                'model_used': 'ru-bert-fine-tuned'
            }
            
        except Exception as e:
            return self._create_error_result(text, f"Russian model prediction error: {str(e)}", 'ru-bert-fine-tuned')
    
    def detect_injection_ml(self, text: str) -> dict[str, Any]:
        """Detect prompt injection using the appropriate language model."""
        language = self.detect_language(text)
        start_t = time.perf_counter()
        
        try:
            if not self.ml_model_available:
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "label": "SAFE",
                    "error": "No ML models available",
                    "model_used": "none",
                    "detected_language": language,
                }
            
            # Russian model branch
            if language == "ru" and self.ru_model_available:
                result = self.predict_russian_injection(text)
                result["detected_language"] = language
            
            # English / fallback branch
            elif self.en_model_available:
                try:
                    pipe_out = self.en_classifier(text)
                    prediction = pipe_out[0] if isinstance(pipe_out, list) else pipe_out
                    label = prediction["label"]
                    confidence = prediction["score"]
                    is_injection = label in ["INJECTION", "MALICIOUS", "1"] or (label == "LABEL_1" and confidence > settings.prompt_injection_threshold)
                    
                    result = {
                        "detected": is_injection,
                        "confidence": confidence,
                        "label": label,
                        "raw_prediction": prediction,
                        "model_used": "en-deberta-v3" if language == "en" else "en-deberta-v3-fallback",
                        "detected_language": language,
                    }
                except Exception as e:
                    result = self._create_error_result(text, f"English model prediction error: {str(e)}", "en-deberta-v3")
                    result["detected_language"] = language
            else:
                result = {
                    "detected": False,
                    "confidence": 0.0,
                    "label": "SAFE",
                    "error": f"No suitable model available for language: {language}",
                    "model_used": "none",
                    "detected_language": language,
                }
            
            # Track metrics
            if result.get("detected"):
                metrics.PROMPT_INJECTION_DETECTIONS_TOTAL.labels("ml").inc()
            
            return result
            
        finally:
            metrics.INJECTION_ML_LATENCY.labels(language).observe(time.perf_counter() - start_t)
    
    def detect_injection_regex(self, text: str) -> dict[str, Any]:
        """Detect prompt injection using regex patterns."""
        result = {"detected": False, "patterns_matched": [], "fuzzy_matches": []}
        
        # Check dangerous patterns
        for pattern in settings.dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["detected"] = True
                result["patterns_matched"].append(pattern)
                metrics.PROMPT_INJECTION_DETECTIONS_TOTAL.labels("regex").inc()
        
        # Check fuzzy patterns
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            for pattern in settings.fuzzy_patterns:
                if self._is_similar_word(word, pattern):
                    result["detected"] = True
                    result["fuzzy_matches"].append({"word": word, "pattern": pattern})
                    metrics.PROMPT_INJECTION_DETECTIONS_TOTAL.labels("regex").inc()
        
        return result
    
    def predict(self, text: str, **kwargs) -> dict[str, Any]:
        """Main prediction method combining regex and ML detection."""
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
        """Check if word is similar to target using fuzzy matching."""
        if len(word) != len(target) or len(word) < 3:
            return False
        return (
            word[0] == target[0]
            and word[-1] == target[-1]
            and sorted(word[1:-1]) == sorted(target[1:-1])
        )
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize input text by removing dangerous patterns."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove excessive character repetition
        text = re.sub(r"(.)\1{3,}", r"\1", text)
        
        # Filter out dangerous patterns
        for pattern in settings.dangerous_patterns:
            text = re.sub(pattern, "[FILTERED]", text, flags=re.IGNORECASE)
        
        # Limit length to 10000 characters
        return text[:10000]