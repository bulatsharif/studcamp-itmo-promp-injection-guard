import re
from typing import Literal


class LanguageDetector:
    """Centralized language detection service for consistent language identification across all models."""
    
    def __init__(self):
        self.cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
    
    def detect_language(self, text: str) -> Literal['ru', 'en']:
        """
        Detect if text is Russian or English based on Cyrillic characters.
        
        Args:
            text: Input text to analyze
            
        Returns:
            'ru' if Russian text detected, 'en' otherwise
        """
        if not text:
            return 'en'
        
        if self.cyrillic_pattern.search(text):
            return 'ru'
        return 'en'
    
    def is_russian(self, text: str) -> bool:
        """Check if text is Russian."""
        return self.detect_language(text) == 'ru'
    
    def is_english(self, text: str) -> bool:
        """Check if text is English."""
        return self.detect_language(text) == 'en'


# Global language detector instance
language_detector = LanguageDetector()