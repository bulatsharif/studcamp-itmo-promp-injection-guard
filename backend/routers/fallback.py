import os
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from backend.schemas import FallbackRequest, FallbackResponse
from backend.config import settings


router = APIRouter()


def load_fallback_prompt() -> str:
    """Load the fallback analysis prompt from file."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "prompts", "fallback.txt"
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


@router.post("/fallback-analyze", response_model=FallbackResponse, tags=["fallback"])
async def fallback_analyze(request: FallbackRequest):
    """
    Perform fallback analysis using external LLM via OpenRouter API.

    This endpoint provides an additional layer of analysis using a large language model
    when the primary ML models may not be sufficient or available.
    """
    # Check if API key is configured
    if not settings.openrouter_api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    try:
        # Initialize OpenAI client with OpenRouter
        client = OpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )

        # Load prompt template
        prompt_template = load_fallback_prompt()
        prompt = prompt_template.format(user_prompt=request.user_prompt)

        # Make API call
        completion = client.chat.completions.create(
            model=settings.openrouter_model,
            messages=[{"role": "user", "content": prompt}],
        )

        return FallbackResponse(result=completion.choices[0].message.content)

    except Exception as e:
        return FallbackResponse(result="", error=f"Fallback analysis failed: {str(e)}")
