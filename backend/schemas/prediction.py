"""Shared request schema for disease prediction endpoints."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    """Validated payload accepted by prediction endpoints."""

    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any]
