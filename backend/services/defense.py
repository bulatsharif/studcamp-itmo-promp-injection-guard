from typing import Any, Optional
from backend.schemas import SafetyScores


class UnifiedMessageDefense:
    """Unified defense system orchestrating all safety checks."""

    def __init__(
        self, prompt_injection_filter=None, toxicity_detector=None, spam_detector=None
    ):
        # Detectors are injected via dependency injection
        self.prompt_injection_filter = prompt_injection_filter
        self.toxicity_detector = toxicity_detector
        self.spam_detector = spam_detector

    def process_message(
        self, message: str, language: Optional[str] = None
    ) -> dict[str, Any]:
        """Process message through all safety checks."""
        result = {
            "original_message": message,
            "filtered_message": message,
            "is_safe": True,
            "rejection_reason": None,
            "safety_scores": SafetyScores(),
        }

        # Step 1: Prompt injection detection (always enabled)
        if self.prompt_injection_filter:
            injection_result = self.prompt_injection_filter.predict(message)
            # Extract confidence from ML detection regardless of detection status
            ml_detection = injection_result.get("ml_detection", {})
            result["safety_scores"].prompt_injection = ml_detection.get(
                "confidence", 0.0
            )

            if injection_result["detected"]:
                result["is_safe"] = False
                detection_methods = ", ".join(injection_result["detection_methods"])
                result["rejection_reason"] = (
                    f"Prompt injection detected via {detection_methods}"
                )
                return result

            # Sanitize input if not blocked
            result["filtered_message"] = self.prompt_injection_filter.sanitize_input(
                message
            )

        # Step 2: Toxicity detection (always enabled)
        if self.toxicity_detector:
            try:
                toxicity_result = self.toxicity_detector.predict(
                    result["filtered_message"], language=language
                )
                result["safety_scores"].toxicity = toxicity_result["toxicity_score"]

                if toxicity_result["is_toxic"]:
                    result["is_safe"] = False
                    result["rejection_reason"] = (
                        f"Toxic content detected (score: {toxicity_result['toxicity_score']:.3f})"
                    )
                    return result
            except Exception as e:
                print(f"Toxicity detection error: {e}")

        # Step 3: Spam detection (always enabled)
        if self.spam_detector:
            try:
                spam_result = self.spam_detector.predict(
                    result["filtered_message"], language=language
                )

                result["safety_scores"].spam = spam_result["spam_score"]

                if spam_result["is_spam"]:
                    result["is_safe"] = False
                    result["rejection_reason"] = (
                        f"Spam content detected (score: {spam_result['spam_score']:.3f})"
                    )
                    return result
            except Exception as e:
                print(f"Spam detection error: {e}")

        return result
