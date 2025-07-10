from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Any


class Settings(BaseSettings):
    # Model paths and names
    ru_bert_model_path: str = Field(default="./ru-bert-prompt-injection", description="Path to Russian BERT model")
    en_prompt_injection_model: str = Field(default="protectai/deberta-v3-base-prompt-injection-v2", description="English prompt injection model")
    ru_toxicity_model: str = Field(default="cointegrated/rubert-tiny-toxicity", description="Russian toxicity model")
    en_toxicity_model: str = Field(default="minuva/MiniLMv2-toxic-jigsaw", description="English toxicity model")
    ru_spam_model: str = Field(default="RUSpam/spam_deberta_v4", description="Russian spam model")
    en_spam_model: str = Field(default="mariagrandury/roberta-base-finetuned-sms-spam-detection", description="English spam model")
    
    # Detection thresholds
    toxicity_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Toxicity detection threshold")
    spam_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Spam detection threshold")
    prompt_injection_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Prompt injection detection threshold")
    
    # Prompt injection regex patterns
    dangerous_patterns: list[str] = Field(
        default=[
            r"ignore\s+(all\s+)?previous\s+instructions?",
            r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
            r"system\s+override",
            r"reveal\s+prompt"
        ],
        description="Dangerous regex patterns for prompt injection"
    )
    
    fuzzy_patterns: list[str] = Field(
        default=["ignore", "bypass", "override", "reveal", "delete", "system"],
        description="Fuzzy matching patterns for prompt injection"
    )
    
    # OpenRouter API settings for fallback
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    openrouter_model: str = Field(default="qwen/qwen3-30b-a3b:free", description="OpenRouter model to use")
    
    # Prometheus settings
    prometheus_multiproc_dir: Optional[str] = Field(default=None, description="Prometheus multiprocess directory")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }
    
    @classmethod
    def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
        if field_name in ["dangerous_patterns", "fuzzy_patterns"]:
            # Parse comma-separated lists from environment variables
            return [item.strip() for item in raw_val.split(",") if item.strip()]
        return raw_val


# Global settings instance
settings = Settings()