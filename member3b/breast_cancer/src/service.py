"""
Breast Cancer Prediction Module - Service Layer.

Provides the primary business logic and decoupled service interfaces
for Breast Cancer risk screening and clinical evaluation.
Delegates model inference and feature scaling strictly to predict.py.

Medical Safety Disclaimer:
This Breast Cancer prediction service is a machine-learning screening/risk
prediction component based on the trained dataset model. It is NOT a medical
diagnosis and must not replace evaluation by a qualified healthcare professional.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json

from .predict import (
    predict_user as _predict_user_core,
    predict_clinical as _predict_clinical_core,
    load_artifacts as _load_artifacts_core,
    EXPECTED_PREDICTOR_FEATURES,
    DEFAULT_CLASS_MAPPING,
    DEFAULT_MODEL_NAME,
    DEFAULT_THRESHOLD,
)

DISEASE_KEY = "breast_cancer"


class BreastCancerPredictionService:
    """
    Unified, decoupled service class managing Breast Cancer risk screening
    and clinical diagnostic predictions.
    """

    def __init__(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initializes the Breast Cancer prediction service.

        Args:
            models_dir: Optional custom path to model artifacts directory.
        """
        self.models_dir = Path(models_dir) if models_dir is not None else None
        # Verify readiness upon initialization
        self._preprocessor = None
        self._model = None
        self._class_mapping = None
        self._metadata = None

    def _ensure_artifacts(self) -> None:
        """Loads and caches artifacts via predict.py if not already loaded."""
        if self._model is None or self._preprocessor is None:
            (
                self._preprocessor,
                self._model,
                self._class_mapping,
                self._metadata,
            ) = _load_artifacts_core(models_dir=self.models_dir)

    def health_check(self) -> Dict[str, Any]:
        """
        Internal Python capability check verifying inference artifacts are available and loadable.
        Does NOT expose an HTTP endpoint.

        Returns:
            Dict[str, Any]: Status dictionary.
        """
        try:
            self._ensure_artifacts()
            model_loaded = hasattr(self._model, "predict")
            preprocessor_loaded = hasattr(self._preprocessor, "transform")
            return {
                "status": "ready" if (model_loaded and preprocessor_loaded) else "unhealthy",
                "disease": DISEASE_KEY,
                "model_loaded": model_loaded,
                "preprocessor_loaded": preprocessor_loaded,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "disease": DISEASE_KEY,
                "model_loaded": False,
                "preprocessor_loaded": False,
                "message": str(e),
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieves authentic metadata describing the active model configuration.

        Returns:
            Dict[str, Any]: Model information dictionary.
        """
        self._ensure_artifacts()

        model_name = self._metadata.get("selected_model_name", DEFAULT_MODEL_NAME)
        threshold = self._metadata.get("decision_threshold", DEFAULT_THRESHOLD)
        feature_count = self._metadata.get("feature_count", len(EXPECTED_PREDICTOR_FEATURES))
        class_mapping = self._class_mapping if self._class_mapping else DEFAULT_CLASS_MAPPING

        return {
            "disease": DISEASE_KEY,
            "model": model_name,
            "threshold": threshold,
            "feature_count": feature_count,
            "class_mapping": class_mapping,
            "features": list(EXPECTED_PREDICTOR_FEATURES),
        }

    def predict_user(self, input_data: Any) -> Dict[str, Any]:
        """
        Executes user-facing risk screening prediction.
        Accepts 30 cytological features (flat dict or grouped) and returns
        patient-accessible risk assessment information.

        Args:
            input_data: 30-feature input dictionary, DataFrame, or Series.

        Returns:
            Dict[str, Any]: Consistent service response dictionary.
        """
        try:
            raw_res = _predict_user_core(input_data=input_data, models_dir=self.models_dir)
            return self._format_service_response(raw_res)
        except Exception:
            return {
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "service_error",
                "message": "An internal error occurred while evaluating the prediction.",
            }

    def predict_clinical(self, input_data: Any) -> Dict[str, Any]:
        """
        Executes clinical diagnostic prediction.
        Enforces strict 30-parameter fine needle aspirate (FNA) measurements
        and returns calibrated malignant probability.

        Args:
            input_data: 30-feature input dictionary, DataFrame, or Series.

        Returns:
            Dict[str, Any]: Consistent service response dictionary.
        """
        try:
            raw_res = _predict_clinical_core(input_data=input_data, models_dir=self.models_dir)
            return self._format_service_response(raw_res)
        except Exception:
            return {
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": "service_error",
                "message": "An internal error occurred while evaluating the prediction.",
            }

    def _format_service_response(self, raw_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforces consistent response contract across all service calls.
        """
        if raw_res.get("status") == "success":
            return {
                "status": "success",
                "disease": DISEASE_KEY,
                "prediction": raw_res["prediction"],
                "prediction_class": raw_res["prediction_class"],
                "probability": raw_res["probability"],
                "risk_percentage": raw_res["risk_percentage"],
                "model": raw_res.get("model", DEFAULT_MODEL_NAME),
                "threshold": raw_res.get("threshold", DEFAULT_THRESHOLD),
            }
        else:
            return {
                "status": "error",
                "disease": DISEASE_KEY,
                "error_type": raw_res.get("error_type", "validation_error"),
                "message": raw_res.get("message", "Validation failed"),
            }
