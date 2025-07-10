from pydantic import BaseModel, Field, field_validator
from typing import Optional


class MessageRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=50000, description="Text to analyze for safety"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or v.isspace():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class SafetyScores(BaseModel):
    prompt_injection: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Prompt injection confidence score"
    )
    toxicity: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Toxicity confidence score"
    )
    spam: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Spam confidence score"
    )


class DefenseResponse(BaseModel):
    is_safe: bool = Field(..., description="Whether the message is safe")
    reason: Optional[str] = Field(None, description="Reason for rejection if not safe")
    scores: SafetyScores = Field(..., description="Safety scores from all models")


class FallbackRequest(BaseModel):
    user_prompt: str = Field(
        ..., min_length=1, max_length=50000, description="User prompt to analyze"
    )

    @field_validator("user_prompt")
    @classmethod
    def validate_user_prompt(cls, v):
        if not v or v.isspace():
            raise ValueError("User prompt cannot be empty or only whitespace")
        return v.strip()


class FallbackResponse(BaseModel):
    result: str = Field(..., description="LLM analysis result")
    error: Optional[str] = Field(None, description="Error message if analysis failed")
