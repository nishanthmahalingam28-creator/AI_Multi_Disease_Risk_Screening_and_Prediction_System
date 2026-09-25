"""
Lung Cancer Prediction Module - Service Layer.

Provides the primary business logic and decoupled service interfaces
for Lung Cancer risk screening and clinical evaluation.
Delegates model inference and feature scaling strictly to predict.py.

Medical Safety Disclaimer:
This system provides a machine-learning-based lung cancer risk/screening prediction
from the supplied survey features. It is not a medical diagnosis and must not be used
as a substitute for professional medical evaluation. The model was developed and
evaluated on the supplied dataset and has not been established here as a clinically
validated diagnostic tool.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import re

from .predict import (
    predict_user as _predict_user_core,
    predict_clinical as _predict_clinical_core,
    load_artifacts as _load_artifacts_core,
    EXPECTED_PREDICTOR_FEATURES,
    DEFAULT_CLASS_MAPPING,
    DEFAULT_MODEL_NAME,
    DEFAULT_THRESHOLD,
    LungCancerValidationError,
    LungCancerModelError,
)

DISEASE_KEY = "lung_cancer"
SERVICE_NAME = "lung-cancer-prediction"


def _sanitize_error_message(msg: str) -> str:
    """Removes any accidental filesystem paths or stack traces from error messages."""
    clean = re.sub(r"[A-Za-z]:\\[^ \t\n\r]+", "[redacted_path]", str(msg))
    clean = re.sub(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "[redacted_path]", clean)
    return clean


class LungCancerPredictionService:
    """
    Unified, decoupled service class managing Lung Cancer risk screening
    and clinical evaluation predictions.
    Delegates all model inference and preprocessing to predict.py.
    """

    def __init__(self, models_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initializes the Lung Cancer prediction service.

        Args:
            models_dir: Optional custom path to model artifacts directory.
        """
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self._preprocessor = None
        self._model = None
        self._metadata = None

    def _ensure_artifacts(self) -> None:
        """Loads and caches artifacts via predict.py if not already loaded."""
        if self._model is None or self._preprocessor is None:
            (
                self._preprocessor,
                self._model,
                self._metadata,
            ) = _load_artifacts_core(
                models_dir=self.models_dir,
                force_reload=bool(self.models_dir is not None),
            )

    def health_check(self) -> Dict[str, Any]:
        """
        Capability check verifying inference artifacts are available and loadable.
        Does NOT expose internal filesystem paths or stack traces.

        Returns:
            Dict[str, Any]: Service health status dictionary.
        """
        try:
            self._ensure_artifacts()
            model_loaded = hasattr(self._model, "predict")
            preprocessor_loaded = hasattr(self._preprocessor, "transform")
            is_ready = bool(model_loaded and preprocessor_loaded)

            return {
                "status": "ready" if is_ready else "unavailable",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "model_loaded": model_loaded,
                "preprocessor_loaded": preprocessor_loaded,
            }
        except Exception:
            return {
                "status": "unavailable",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "model_loaded": False,
                "preprocessor_loaded": False,
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieves authentic metadata describing the active model configuration.
        Does NOT expose internal filesystem paths or raw stack traces.

        Returns:
            Dict[str, Any]: Safe model information dictionary.
        """
        try:
            self._ensure_artifacts()

            model_name = self._metadata.get("model_type", DEFAULT_MODEL_NAME)
            threshold = self._metadata.get("decision_threshold", DEFAULT_THRESHOLD)
            feature_count = self._metadata.get("feature_count", len(EXPECTED_PREDICTOR_FEATURES))
            target_mapping = self._metadata.get("target_mapping", {"NO": 0, "YES": 1})
            class_mapping = DEFAULT_CLASS_MAPPING

            return {
                "status": "success",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "model": model_name,
                "threshold": threshold,
                "feature_count": feature_count,
                "target_mapping": target_mapping,
                "class_mapping": class_mapping,
                "features": list(EXPECTED_PREDICTOR_FEATURES),
            }
        except Exception:
            return {
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "service_error",
                "message": "Lung cancer prediction service is currently unavailable.",
            }

    def predict_user(self, input_data: Any) -> Dict[str, Any]:
        """
        Executes user-facing lung cancer risk screening prediction.
        Accepts 15 survey features and delegates prediction to predict.py.

        Args:
            input_data: 15-feature input dictionary, DataFrame, or Series.

        Returns:
            Dict[str, Any]: Standardized service response dictionary.
        """
        try:
            raw_res = _predict_user_core(input_data=input_data, models_dir=self.models_dir)
            return self._format_service_response(raw_res)
        except Exception as e:
            return {
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "service_error",
                "message": "Lung cancer prediction service is currently unavailable.",
            }

    def predict_clinical(self, input_data: Any) -> Dict[str, Any]:
        """
        Executes clinical screening prediction.
        Enforces identical inference logic as predict_user() by delegating to predict.py.

        Args:
            input_data: 15-feature input dictionary, DataFrame, or Series.

        Returns:
            Dict[str, Any]: Standardized service response dictionary.
        """
        try:
            raw_res = _predict_clinical_core(input_data=input_data, models_dir=self.models_dir)
            return self._format_service_response(raw_res)
        except Exception as e:
            return {
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": "service_error",
                "message": "Lung cancer prediction service is currently unavailable.",
            }

    def _format_service_response(self, raw_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforces a clean, safe, consistent service-level response contract.
        """
        if raw_res.get("status") == "success":
            return {
                "status": "success",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "prediction": raw_res["prediction"],
                "prediction_class": raw_res["prediction_class"],
                "probability": raw_res["probability"],
                "risk_percentage": raw_res["risk_percentage"],
                "model": raw_res.get("model", DEFAULT_MODEL_NAME),
                "threshold": raw_res.get("threshold", DEFAULT_THRESHOLD),
                "screening_interpretation": raw_res.get("screening_interpretation", "Screening result generated"),
            }
        else:
            raw_msg = raw_res.get("message", "Validation failed")
            clean_msg = _sanitize_error_message(raw_msg)
            err_type = raw_res.get("error_type", "validation_error")
            if err_type in ("artifact_error", "transformation_error", "inference_error"):
                err_type = "service_error"
                clean_msg = "Lung cancer prediction service is currently unavailable."
            return {
                "status": "error",
                "service": SERVICE_NAME,
                "disease": DISEASE_KEY,
                "error_type": err_type,
                "message": clean_msg,
            }
