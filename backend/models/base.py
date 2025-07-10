from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import contextlib
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from optimum.bettertransformer import BetterTransformer
from backend.services.language import LanguageDetector


class BaseDetector(ABC):
    """Base class for all ML-based detectors providing common functionality."""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.language_detector = LanguageDetector()
        self._init_models()

    @abstractmethod
    def _init_models(self):
        """Initialize models - must be implemented by subclasses."""
        pass

    def _load_pipeline(
        self,
        model_name: str,
        task: str = "text-classification",
        device: Optional[int] = None,
        **kwargs,
    ) -> pipeline:
        """Load a HuggingFace pipeline with optimizations."""
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        pipe = pipeline(
            model=model_name, task=task, device=device, verbose=False, **kwargs
        )

        # Apply optimizations (always enabled)
        if hasattr(pipe, "model") and isinstance(pipe.model, torch.nn.Module):
            try:
                pipe.model.eval()

                # Apply torch.compile optimization
                with contextlib.suppress(Exception):
                    pipe.model = torch.compile(pipe.model, mode="max_autotune")
                    pipe.model = BetterTransformer.transform(pipe.model)

            except Exception as e:
                print(f"Warning: Failed to apply optimizations to {model_name}: {e}")

        return pipe

    def _load_model_and_tokenizer(
        self, model_path: str, use_cuda: bool = True
    ) -> tuple:
        """Load model and tokenizer with optimizations."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,  # Always use fast tokenizers
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Move to GPU if available
        if torch.cuda.is_available() and use_cuda:
            model = model.cuda()

        # Apply torch.compile optimization (always enabled)
        with contextlib.suppress(Exception):
            model = torch.compile(model, mode="max_autotune")
            model = BetterTransformer.transform(model)

        # Set to evaluation mode
        model.eval()

        return model, tokenizer

    def detect_language(self, text: str) -> str:
        """Detect language using centralized language detector."""
        return self.language_detector.detect_language(text)

    @abstractmethod
    def predict(self, text: str, **kwargs) -> dict[str, Any]:
        """Make a prediction - must be implemented by subclasses."""
        pass

    def batch_predict(self, texts: list, **kwargs) -> list:
        """Make predictions on a batch of texts."""
        results = []
        for text in texts:
            try:
                result = self.predict(text, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "text": text,
                        "error": str(e),
                        "detected": False,
                        "confidence": 0.0,
                    }
                )
        return results

    def _create_error_result(
        self, text: str, error: str, model_name: str = "unknown"
    ) -> dict[str, Any]:
        """Create a standardized error result."""
        return {
            "text": text,
            "detected": False,
            "confidence": 0.0,
            "error": error,
            "model_used": model_name,
        }
