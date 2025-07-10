from fastapi import Request
from backend.services.defense import UnifiedMessageDefense


def get_defense_system(request: Request) -> UnifiedMessageDefense:
    """Dependency to get unified defense system from request state."""
    return request.state.defense_system
