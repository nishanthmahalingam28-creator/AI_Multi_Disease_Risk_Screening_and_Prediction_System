"""Service layer for executing disease prediction functions."""

from collections.abc import Callable
from typing import Any

from fastapi import HTTPException


PredictionFunction = Callable[[dict[str, Any]], Any]


def handle_prediction(
    predict_function: PredictionFunction,
    features: dict[str, Any],
) -> Any:
    """Run a prediction function and convert validation errors to HTTP 400."""
    try:
        return predict_function(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
