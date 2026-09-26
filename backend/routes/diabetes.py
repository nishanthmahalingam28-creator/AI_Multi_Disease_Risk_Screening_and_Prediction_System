"""Diabetes prediction API route."""

from fastapi import APIRouter

from backend.schemas.prediction import PredictionRequest
from backend.services.prediction_service import handle_prediction
from ml.member1.src.predict_clinical import predict_diabetes

router = APIRouter(prefix="/api/predict/diabetes", tags=["Diabetes"])


@router.post("/clinical")
def diabetes_clinical(request: PredictionRequest):
    return handle_prediction(predict_diabetes, request.features)
