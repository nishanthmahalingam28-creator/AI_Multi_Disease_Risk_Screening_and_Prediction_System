"""API route modules."""

from .heart import router as heart_router
from .diabetes import router as diabetes_router
from .stroke import router as stroke_router

__all__ = ["heart_router", "diabetes_router", "stroke_router"]
