"""Service layer for executing registered disease prediction functions."""

from typing import Any

from fastapi import HTTPException

from backend.core.model_registry import get_prediction_function


def predict_disease(disease: str, features: dict[str, Any]) -> Any:
    """Run the registered prediction function for a disease."""
    try:
        predict_function = get_prediction_function(disease)
        return predict_function(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
