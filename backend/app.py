from fastapi import FastAPI

from backend.routes import heart_router
from backend.services.prediction_service import handle_prediction
from backend.schemas.prediction import PredictionRequest
from ml.member1.src.predict_clinical import predict_diabetes, predict_stroke

app = FastAPI(
    title="AI Multi-Disease Risk Screening API",
    description="Screening-only API for heart disease, diabetes, and stroke.",
    version="1.0.0",
)

app.include_router(heart_router)


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


@app.post("/api/predict/diabetes/clinical")
def diabetes_clinical(request: PredictionRequest):
    return handle_prediction(predict_diabetes, request.features)


@app.post("/api/predict/stroke/clinical")
def stroke_clinical(request: PredictionRequest):
    return handle_prediction(predict_stroke, request.features)
