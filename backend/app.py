from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import Any

from ml.member1.src.predict_clinical import (
    predict_heart,
    predict_diabetes,
    predict_stroke,
)

app = FastAPI(
    title="AI Multi-Disease Risk Screening API",
    description="Screening-only API for heart disease, diabetes, and stroke.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, Any]


@app.get("/")
def root():
    return {
        "project": "AI Multi-Disease Risk Screening System",
        "status": "running",
        "screening_only": True,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "module": "member1",
    }


def handle_prediction(predict_function, features):
    try:
        return predict_function(features)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/api/predict/heart/clinical")
def heart_clinical(request: PredictionRequest):
    return handle_prediction(predict_heart, request.features)


@app.post("/api/predict/diabetes/clinical")
def diabetes_clinical(request: PredictionRequest):
    return handle_prediction(predict_diabetes, request.features)


@app.post("/api/predict/stroke/clinical")
def stroke_clinical(request: PredictionRequest):
    return handle_prediction(predict_stroke, request.features)