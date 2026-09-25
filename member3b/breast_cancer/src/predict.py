"""
Breast Cancer Prediction Module - Inference Service.

Provides reusable prediction interfaces (User and Clinical) using the trained
Wisconsin Diagnostic Breast Cancer Logistic Regression model and StandardScaler pipeline.
Enforces strict 30-feature validation, deterministic ordering, and leakage-safe transformations.

Medical Safety Disclaimer:
This Breast Cancer prediction system is a machine-learning screening/risk prediction
tool based on the trained dataset model. It is not a medical diagnosis and must
not replace evaluation by a qualified healthcare professional.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import math
import json
import joblib
import numpy as np
import pandas as pd

# Expected 30 predictor features in strict training order
EXPECTED_PREDICTOR_FEATURES: List[str] = [
    # Mean features (10)
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
    "symmetry_mean", "fractal_dimension_mean",
    # Standard error features (10)
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se",
    # Worst / Largest features (10)
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst"
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "mean_features": EXPECTED_PREDICTOR_FEATURES[0:10],
    "se_features": EXPECTED_PREDICTOR_FEATURES[10:20],
    "worst_features": EXPECTED_PREDICTOR_FEATURES[20:30],
}

DEFAULT_CLASS_MAPPING: Dict[int, str] = {0: "Benign", 1: "Malignant"}
DEFAULT_MODEL_NAME: str = "Logistic Regression"
DEFAULT_THRESHOLD: float = 0.50

# Cache for loaded artifacts to avoid redundant disk I/O
_CACHED_PREPROCESSOR = None
_CACHED_MODEL = None
_CACHED_CLASS_MAPPING = None
_CACHED_METADATA = None


def get_default_models_dir() -> Path:
    """Returns the default directory housing trained model artifacts."""
    # File is at: member3b/breast_cancer/src/predict.py
    # parents: 0=src, 1=breast_cancer, 2=member3b, 3=repo_root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "member3b" / "breast_cancer" / "models"


def load_artifacts(
    models_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Tuple[Any, Any, Dict[int, str], Dict[str, Any]]:
    """
    Loads and caches the preprocessor pipeline, trained classifier,
    class mapping, and model metadata.

    Args:
        models_dir: Optional custom directory. Defaults to member3b/breast_cancer/models.
        force_reload: If True, forces reload from disk.

    Returns:
        Tuple[preprocessor, model, class_mapping, metadata]
    """
    global _CACHED_PREPROCESSOR, _CACHED_MODEL, _CACHED_CLASS_MAPPING, _CACHED_METADATA

    if not force_reload and _CACHED_PREPROCESSOR is not None and _CACHED_MODEL is not None:
        return _CACHED_PREPROCESSOR, _CACHED_MODEL, _CACHED_CLASS_MAPPING, _CACHED_METADATA

    base_dir = Path(models_dir) if models_dir is not None else get_default_models_dir()

    prep_path = base_dir / "breast_cancer_preprocessor.joblib"
    model_path = base_dir / "breast_cancer_model.joblib"
    mapping_path = base_dir / "class_mapping.json"
    meta_path = base_dir / "model_metadata.json"

    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    preprocessor = joblib.load(prep_path)
    model = joblib.load(model_path)

    # Load class mapping
    class_mapping = DEFAULT_CLASS_MAPPING
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
                class_mapping = {int(k): str(v) for k, v in raw_map.items()}
        except Exception:
            class_mapping = DEFAULT_CLASS_MAPPING

    # Load metadata
    metadata = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    _CACHED_PREPROCESSOR = preprocessor
    _CACHED_MODEL = model
    _CACHED_CLASS_MAPPING = class_mapping
    _CACHED_METADATA = metadata

    return preprocessor, model, class_mapping, metadata


def validate_input(
    input_data: Any,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, float]]]:
    """
    Validates input features for presence, completeness, strict naming, and numeric validity.
    Accepts flat dictionaries, grouped dictionaries, pandas Series, or single-row DataFrames.

    Returns:
        Tuple[is_valid, error_dict, clean_flat_dict]
    """
    if input_data is None:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": "Input data cannot be None.",
        }, None

    raw_dict: Dict[str, Any] = {}

    # Handle pandas DataFrame / Series
    if isinstance(input_data, pd.DataFrame):
        if len(input_data) == 0:
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": "Input DataFrame cannot be empty.",
            }, None
        raw_dict = input_data.iloc[0].to_dict()
    elif isinstance(input_data, pd.Series):
        raw_dict = input_data.to_dict()
    elif isinstance(input_data, dict):
        # Check if grouped format is supplied: {"mean_features": {...}, ...}
        if any(k in input_data for k in ["mean_features", "se_features", "worst_features"]):
            for group_key in ["mean_features", "se_features", "worst_features"]:
                group_val = input_data.get(group_key, {})
                if isinstance(group_val, dict):
                    raw_dict.update(group_val)
                elif isinstance(group_val, list):
                    # List of feature dicts with name & value
                    for item in group_val:
                        if isinstance(item, dict) and "name" in item and "value" in item:
                            raw_dict[item["name"]] = item["value"]
            # Also capture any flat keys alongside groups
            for k, v in input_data.items():
                if k not in ["mean_features", "se_features", "worst_features"]:
                    raw_dict[k] = v
        else:
            raw_dict = dict(input_data)
    else:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Unsupported input data type: {type(input_data).__name__}. Expected dict or pandas Series/DataFrame.",
        }, None

    # 1. Check for unexpected / extra features
    extra_features = [k for k in raw_dict.keys() if k not in EXPECTED_PREDICTOR_FEATURES]
    if extra_features:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Unexpected feature encountered: {extra_features[0]}",
        }, None

    # 2. Check for missing required features
    missing_features = [k for k in EXPECTED_PREDICTOR_FEATURES if k not in raw_dict]
    if missing_features:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Missing required feature: {missing_features[0]}",
        }, None

    # 3. Check value validity (numeric, not NaN, not Inf)
    clean_dict: Dict[str, float] = {}
    for feat in EXPECTED_PREDICTOR_FEATURES:
        val = raw_dict[feat]

        if val is None:
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Feature '{feat}' cannot be None.",
            }, None

        # Check numeric conversion
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Invalid non-numeric value for feature '{feat}': {val}",
            }, None

        # Check NaN
        if math.isnan(val_float) or np.isnan(val_float):
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Feature '{feat}' cannot be NaN.",
            }, None

        # Check Inf
        if math.isinf(val_float) or np.isinf(val_float):
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Feature '{feat}' cannot be infinite.",
            }, None

        clean_dict[feat] = val_float

    return True, None, clean_dict


def predict_breast_cancer(
    input_data: Any,
    raise_on_error: bool = False,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Core prediction engine. Validates input, scales features with the Step 3
    preprocessor, and evaluates with the Step 4 trained Logistic Regression model.

    Args:
        input_data: 30-feature input representation (dict, Series, DataFrame).
        raise_on_error: If True, raises ValueError on validation error instead of returning error dict.
        models_dir: Optional path to model artifacts directory.

    Returns:
        Dict[str, Any]: Structured prediction output or validation error dictionary.
    """
    # 1. Input validation
    is_valid, error_dict, clean_dict = validate_input(input_data)
    if not is_valid:
        if raise_on_error:
            raise ValueError(error_dict["message"])
        return error_dict

    # 2. Load model & preprocessor artifacts
    try:
        preprocessor, model, class_mapping, metadata = load_artifacts(models_dir=models_dir)
    except Exception as e:
        err = {
            "status": "error",
            "error_type": "artifact_error",
            "message": f"Failed to load prediction artifacts: {str(e)}",
        }
        if raise_on_error:
            raise RuntimeError(err["message"])
        return err

    # 3. Construct DataFrame enforcing exact feature ordering
    ordered_features = [{feat: clean_dict[feat] for feat in EXPECTED_PREDICTOR_FEATURES}]
    df_ordered = pd.DataFrame(ordered_features, columns=EXPECTED_PREDICTOR_FEATURES)

    # 4. Transform features with preprocessor
    X_scaled = preprocessor.transform(df_ordered)

    # 5. Model inference
    pred_class = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]

    # Probability of class 1 (Malignant)
    prob_malignant = float(probs[1])
    risk_percentage = round(prob_malignant * 100.0, 2)

    # Class label mapping: 0 -> Benign, 1 -> Malignant
    prediction_label = class_mapping.get(pred_class, DEFAULT_CLASS_MAPPING.get(pred_class, "Unknown"))

    model_name = metadata.get("selected_model_name", DEFAULT_MODEL_NAME)
    threshold = metadata.get("decision_threshold", DEFAULT_THRESHOLD)

    return {
        "prediction": prediction_label,
        "prediction_class": pred_class,
        "probability": round(prob_malignant, 4),
        "risk_percentage": risk_percentage,
        "model": model_name,
        "threshold": threshold,
        "status": "success",
    }


def predict_user(
    input_data: Any,
    raise_on_error: bool = False,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    User-facing prediction interface.
    Accepts 30 cytological features (flat dict or grouped by mean, se, worst)
    and returns accessible risk assessment information.
    """
    return predict_breast_cancer(
        input_data=input_data,
        raise_on_error=raise_on_error,
        models_dir=models_dir,
    )


def predict_clinical(
    input_data: Any,
    raise_on_error: bool = False,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Clinical diagnostic prediction interface.
    Enforces strict fine needle aspirate (FNA) 30-parameter feature input
    and outputs calibrated malignant probability and risk stratification.
    """
    return predict_breast_cancer(
        input_data=input_data,
        raise_on_error=raise_on_error,
        models_dir=models_dir,
    )
