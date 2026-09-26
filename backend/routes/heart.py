"""Heart disease prediction routes."""

from fastapi import APIRouter

from backend.schemas.prediction import PredictionRequest
from backend.services.prediction_service import handle_prediction
from ml.member1.src.predict_clinical import predict_heart

router = APIRouter(prefix="/api/predict/heart", tags=["Heart Disease"])


@router.post("/clinical")
def heart_clinical(request: PredictionRequest):
    """Run the clinical heart disease prediction."""
    return handle_prediction(predict_heart, request.features)
