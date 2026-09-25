"""
Member 3B Common Package - Unified Cancer Prediction Architecture.

Exposes unified prediction service and API coordinating:
- Breast Cancer Risk Screening (member3b.breast_cancer)
- Lung Cancer Risk Screening (member3b.lung_cancer)
"""

from .service import CancerPredictionService, UNIFIED_SERVICE_NAME, SUPPORTED_DISEASES
from .api import create_app

__all__ = [
    "CancerPredictionService",
    "UNIFIED_SERVICE_NAME",
    "SUPPORTED_DISEASES",
    "create_app",
]
