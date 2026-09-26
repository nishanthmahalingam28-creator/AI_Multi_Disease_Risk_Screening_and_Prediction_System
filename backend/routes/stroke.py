"""Stroke prediction API route."""

from fastapi import APIRouter

from backend.schemas.prediction import PredictionRequest
from backend.services.prediction_service import handle_prediction
from ml.member1.src.predict_clinical import predict_stroke

router = APIRouter(prefix="/api/predict/stroke", tags=["Stroke"])


@router.post("/clinical")
def stroke_clinical(request: PredictionRequest):
    return handle_prediction(predict_stroke, request.features)
