"""
Member 3B Unified Cancer Prediction Module - Service Layer.

Coordinates multi-disease risk screening prediction across:
- Breast Cancer Risk Screening (member3b.breast_cancer)
- Lung Cancer Risk Screening (member3b.lung_cancer)

Medical Safety Disclaimer:
This system provides machine-learning-based cancer risk screening predictions
and risk estimates. It is not a medical diagnostic tool and must not replace
professional medical evaluation by qualified healthcare providers.
"""

from typing import Dict, Any, Optional, Set
import re

from member3b.breast_cancer.src.service import BreastCancerPredictionService
from member3b.lung_cancer.src.service import LungCancerPredictionService

UNIFIED_SERVICE_NAME = "member3b-cancer-prediction"
SUPPORTED_DISEASES: Set[str] = {"breast_cancer", "lung_cancer"}


def _sanitize_error_message(msg: str) -> str:
    """Removes any accidental filesystem paths or stack traces from error messages."""
    clean = re.sub(r"[A-Za-z]:\\[^ \t\n\r]+", "[redacted_path]", str(msg))
    clean = re.sub(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", "[redacted_path]", clean)
    return clean


class CancerPredictionService:
    """
    Unified cancer prediction service coordinating disease-specific services.
    Acts as a thin orchestration layer delegating feature validation and
    model inference strictly to the underlying disease services.
    """

    def __init__(
        self,
        breast_service: Optional[BreastCancerPredictionService] = None,
        lung_service: Optional[LungCancerPredictionService] = None,
    ) -> None:
        """
        Initializes the unified prediction service.

        Args:
            breast_service: Optional injected BreastCancerPredictionService instance.
            lung_service: Optional injected LungCancerPredictionService instance.
        """
        self.breast_service = (
            breast_service if breast_service is not None else BreastCancerPredictionService()
        )
        self.lung_service = (
            lung_service if lung_service is not None else LungCancerPredictionService()
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Aggregates health status across all configured cancer prediction services.

        Returns:
            Dict[str, Any]: Unified health report detailing service readiness.
        """
        try:
            breast_health = self.breast_service.health_check()
            lung_health = self.lung_service.health_check()

            breast_ready = breast_health.get("status") == "ready"
            lung_ready = lung_health.get("status") == "ready"

            overall_status = "ready" if (breast_ready and lung_ready) else "unavailable"

            return {
                "status": overall_status,
                "service": UNIFIED_SERVICE_NAME,
                "diseases": {
                    "breast_cancer": breast_health,
                    "lung_cancer": lung_health,
                },
            }
        except Exception:
            return {
                "status": "unavailable",
                "service": UNIFIED_SERVICE_NAME,
                "diseases": {
                    "breast_cancer": {"status": "unavailable"},
                    "lung_cancer": {"status": "unavailable"},
                },
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retrieves safe model metadata from all supported disease modules.
        Exposes public metadata only; withholds coefficients, intercepts, and filesystem paths.

        Returns:
            Dict[str, Any]: Unified model metadata dictionary.
        """
        try:
            breast_info = self.breast_service.get_model_info()
            lung_info = self.lung_service.get_model_info()

            breast_ok = isinstance(breast_info, dict) and breast_info.get("status") != "error"
            lung_ok = isinstance(lung_info, dict) and lung_info.get("status") != "error"

            overall_status = "success" if (breast_ok and lung_ok) else "unavailable"

            return {
                "status": overall_status,
                "service": UNIFIED_SERVICE_NAME,
                "models": {
                    "breast_cancer": breast_info,
                    "lung_cancer": lung_info,
                },
            }
        except Exception:
            return {
                "status": "unavailable",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "service_error",
                "message": "Unified cancer prediction service is currently unavailable.",
            }

    def predict(
        self,
        disease: Any,
        data: Any,
        mode: str = "user",
    ) -> Dict[str, Any]:
        """
        Routes prediction request to the designated cancer prediction service.
        Validates ONLY the routing envelope (disease identifier and data container);
        delegates feature validation strictly to the underlying disease service.

        Args:
            disease: Target disease identifier ('breast_cancer' or 'lung_cancer').
            data: Disease feature payload (dict).
            mode: Prediction mode ('user' or 'clinical'). Defaults to 'user'.

        Returns:
            Dict[str, Any]: Prediction result or validation error dictionary.
        """
        # 1. Envelope validation: Disease
        if disease is None:
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "validation_error",
                "message": "Field 'disease' is required and cannot be null.",
            }

        if not isinstance(disease, str) or not disease.strip():
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "validation_error",
                "message": f"Invalid disease identifier: '{disease}'. Must be a non-empty string.",
            }

        norm_disease = disease.strip().lower()
        if norm_disease not in SUPPORTED_DISEASES:
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "error_type": "validation_error",
                "message": f"Unsupported disease '{disease}'. Supported diseases: {sorted(SUPPORTED_DISEASES)}.",
            }

        # 2. Envelope validation: Data
        if data is None:
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "disease": norm_disease,
                "error_type": "validation_error",
                "message": "Field 'data' is required and cannot be null.",
            }

        if not isinstance(data, dict):
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "disease": norm_disease,
                "error_type": "validation_error",
                "message": f"Field 'data' must be a valid JSON object. Received {type(data).__name__}.",
            }

        if len(data) == 0:
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "disease": norm_disease,
                "error_type": "validation_error",
                "message": "Field 'data' cannot be an empty object.",
            }

        # 3. Envelope validation: Mode
        norm_mode = mode.strip().lower() if isinstance(mode, str) else "user"
        if norm_mode not in ("user", "clinical"):
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "disease": norm_disease,
                "error_type": "validation_error",
                "message": f"Invalid mode '{mode}'. Supported modes: ['user', 'clinical'].",
            }

        # 4. Route to existing disease service
        try:
            if norm_disease == "breast_cancer":
                if norm_mode == "clinical":
                    return self.breast_service.predict_clinical(data)
                return self.breast_service.predict_user(data)
            elif norm_disease == "lung_cancer":
                if norm_mode == "clinical":
                    return self.lung_service.predict_clinical(data)
                return self.lung_service.predict_user(data)
            else:
                return {
                    "status": "error",
                    "service": UNIFIED_SERVICE_NAME,
                    "error_type": "validation_error",
                    "message": f"Routing failure for disease: '{norm_disease}'.",
                }
        except Exception as e:
            clean_msg = _sanitize_error_message(str(e))
            return {
                "status": "error",
                "service": UNIFIED_SERVICE_NAME,
                "disease": norm_disease,
                "error_type": "service_error",
                "message": "An internal error occurred in the disease prediction service.",
            }
