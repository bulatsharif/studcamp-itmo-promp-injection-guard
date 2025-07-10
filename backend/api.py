from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.models.prompt_injection import PromptInjectionFilter
from backend.models.toxicity import ToxicityDetector
from backend.models.spam import SpamDetector
from backend.services.defense import UnifiedMessageDefense
from backend.routers import defense, fallback
import backend.metrics as metrics


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Manage application lifespan and create shared resources."""
    # Startup
    print("Starting application...")

    # Initialize prompt injection filter (always enabled)
    print("Loading prompt injection filter...")
    prompt_injection_filter = PromptInjectionFilter()
    print("✓ Prompt injection filter loaded")

    # Initialize toxicity detector (always enabled)
    print("Loading toxicity detector...")
    toxicity_detector = ToxicityDetector()
    print("✓ Toxicity detector loaded")

    # Initialize spam detector (always enabled)
    print("Loading spam detector...")
    spam_detector = SpamDetector()
    print("✓ Spam detector loaded")

    # Initialize unified defense system
    print("Initializing unified defense system...")
    defense_system = UnifiedMessageDefense(
        prompt_injection_filter=prompt_injection_filter,
        toxicity_detector=toxicity_detector,
        spam_detector=spam_detector,
    )
    print("✓ Unified defense system initialized")

    print("Model initialization complete!")

    yield {"defense_system": defense_system}

    # Shutdown
    print("Shutting down application...")
    # Models will be garbage collected automatically
    print("Model cleanup complete!")


# Create FastAPI application
app = FastAPI(
    title="Prompt Injection Guard",
    description="Multi-layered defense system for LLM applications",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(defense.router)
app.include_router(fallback.router)
app.include_router(metrics.router)


@app.get("/", tags=["health"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "prompt-injection-guard"}
