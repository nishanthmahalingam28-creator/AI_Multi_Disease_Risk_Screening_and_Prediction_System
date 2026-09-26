"""Central registry for prediction functions used by the API."""

from collections.abc import Callable
from typing import Any

from ml.member1.src.predict_clinical import (
    predict_diabetes,
    predict_heart,
    predict_stroke,
)

PredictionFunction = Callable[[dict[str, Any]], Any]


PREDICTION_FUNCTIONS: dict[str, PredictionFunction] = {
    "heart": predict_heart,
    "diabetes": predict_diabetes,
    "stroke": predict_stroke,
}


def get_prediction_function(disease: str) -> PredictionFunction:
    """Return the registered prediction function for a disease."""
    try:
        return PREDICTION_FUNCTIONS[disease.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported disease: {disease}") from exc


def list_supported_diseases() -> list[str]:
    """Return the diseases currently registered with the backend."""
    return sorted(PREDICTION_FUNCTIONS)
