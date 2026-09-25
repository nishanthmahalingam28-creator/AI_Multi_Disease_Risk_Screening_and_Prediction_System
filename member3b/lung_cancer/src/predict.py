"""
Lung Cancer Prediction Module - Inference Service.

Provides reusable prediction interfaces (User and Clinical) using the trained
Survey Lung Cancer Logistic Regression model and StandardScaler pipeline.
Enforces strict 15-feature validation, canonical ordering, and leakage-safe transformations.

Medical Safety Disclaimer:
This system provides a machine-learning-based lung cancer risk/screening prediction
from the supplied survey features. It is not a medical diagnosis and must not be used
as a substitute for professional medical evaluation. The model was developed and
evaluated on the supplied dataset and has not been established here as a clinically
validated diagnostic tool.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import math
import json
import joblib
import numpy as np
import pandas as pd

# Expected 15 predictor features in strict canonical training order
EXPECTED_PREDICTOR_FEATURES: List[str] = [
    "gender",
    "age",
    "smoking",
    "yellow_fingers",
    "anxiety",
    "peer_pressure",
    "chronic_disease",
    "fatigue",
    "allergy",
    "wheezing",
    "alcohol_consuming",
    "coughing",
    "shortness_of_breath",
    "swallowing_difficulty",
    "chest_pain",
]

# 13 survey symptom/behavior features
SYMPTOM_FEATURES: List[str] = [
    "smoking",
    "yellow_fingers",
    "anxiety",
    "peer_pressure",
    "chronic_disease",
    "fatigue",
    "allergy",
    "wheezing",
    "alcohol_consuming",
    "coughing",
    "shortness_of_breath",
    "swallowing_difficulty",
    "chest_pain",
]

DEFAULT_CLASS_MAPPING: Dict[int, str] = {0: "NO", 1: "YES"}
DEFAULT_MODEL_NAME: str = "LogisticRegression"
DEFAULT_THRESHOLD: float = 0.50

MIN_AGE: float = 1.0
MAX_AGE: float = 120.0

# Cache for loaded artifacts to prevent redundant disk I/O
_CACHED_PREPROCESSOR = None
_CACHED_MODEL = None
_CACHED_METADATA = None


class LungCancerValidationError(ValueError):
    """Raised when user or clinical input fails domain/contract validation."""
    pass


class LungCancerModelError(RuntimeError):
    """Raised when prediction artifacts fail to load or execute."""
    pass


def get_default_models_dir() -> Path:
    """Returns the default directory housing trained model artifacts."""
    # File is at: member3b/lung_cancer/src/predict.py
    # parents: 0=src, 1=lung_cancer, 2=member3b, 3=repo_root
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "member3b" / "lung_cancer" / "models"


def load_artifacts(
    models_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Loads and caches the preprocessor pipeline, trained classifier, and model metadata.

    Args:
        models_dir: Optional custom directory. Defaults to member3b/lung_cancer/models.
        force_reload: If True, reloads artifacts fresh from disk.

    Returns:
        Tuple[preprocessor, model, metadata]

    Raises:
        LungCancerModelError: If model or preprocessor artifacts cannot be located or loaded.
    """
    global _CACHED_PREPROCESSOR, _CACHED_MODEL, _CACHED_METADATA

    if not force_reload and models_dir is None and _CACHED_PREPROCESSOR is not None and _CACHED_MODEL is not None:
        return _CACHED_PREPROCESSOR, _CACHED_MODEL, _CACHED_METADATA

    base_dir = Path(models_dir) if models_dir is not None else get_default_models_dir()

    prep_path = base_dir / "lung_cancer_preprocessor.joblib"
    model_path = base_dir / "lung_cancer_model.joblib"
    meta_path = base_dir / "model_metadata.json"

    if not prep_path.exists():
        raise LungCancerModelError(f"Preprocessor artifact not found at resolved location: {prep_path.name}")
    if not model_path.exists():
        raise LungCancerModelError(f"Model artifact not found at resolved location: {model_path.name}")

    try:
        preprocessor = joblib.load(prep_path)
    except Exception as e:
        raise LungCancerModelError(f"Failed to load preprocessor artifact: {str(e)}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise LungCancerModelError(f"Failed to load model artifact: {str(e)}")

    metadata = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    if models_dir is None:
        _CACHED_PREPROCESSOR = preprocessor
        _CACHED_MODEL = model
        _CACHED_METADATA = metadata

    return preprocessor, model, metadata


def validate_input(
    input_data: Any,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Validates input features for presence, completeness, strict naming, and domain validity:
    - Exactly 15 required features.
    - Rejects missing, extra, None, NaN, infinite, or malformed fields.
    - Normalizes gender ('MALE'/'FEMALE' case-insensitive or {0, 1}) -> 0/1 integer.
    - Validates age within human physiological range [1, 120].
    - Normalizes 13 binary symptoms ({0, 1} or 'YES'/'NO' case-insensitive) -> 0/1 integer.

    Returns:
        Tuple[is_valid, error_dict, clean_dict]
    """
    if input_data is None:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": "Input data cannot be None.",
        }, None

    raw_dict: Dict[str, Any] = {}

    # Extract dict from supported containers
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
        raw_dict = dict(input_data)
    else:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Unsupported input data type: {type(input_data).__name__}. Expected dict or pandas Series/DataFrame.",
        }, None

    # 1. Check for unexpected / extra fields
    extra_features = [k for k in raw_dict.keys() if k not in EXPECTED_PREDICTOR_FEATURES]
    if extra_features:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Unexpected feature encountered: {extra_features[0]}",
        }, None

    # 2. Check for missing required fields
    missing_features = [k for k in EXPECTED_PREDICTOR_FEATURES if k not in raw_dict]
    if missing_features:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Missing required feature: {missing_features[0]}",
        }, None

    clean_dict: Dict[str, Any] = {}

    # 3. Validate Gender
    gender_val = raw_dict["gender"]
    if gender_val is None:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": "Feature 'gender' cannot be None.",
        }, None

    # Python bool is an int subclass; reject True/False for gender
    if isinstance(gender_val, bool):
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Invalid value for gender: {gender_val}. Boolean values are not accepted.",
        }, None

    if isinstance(gender_val, str):
        g_clean = gender_val.strip().upper()
        if g_clean in ["MALE", "1"]:
            clean_dict["gender"] = 1
        elif g_clean in ["FEMALE", "0"]:
            clean_dict["gender"] = 0
        else:
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Invalid value for gender: '{gender_val}'. Expected 'MALE' or 'FEMALE'.",
            }, None
    elif isinstance(gender_val, (int, float)):
        if gender_val in [1, 1.0]:
            clean_dict["gender"] = 1
        elif gender_val in [0, 0.0]:
            clean_dict["gender"] = 0
        else:
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Invalid numeric value for gender: {gender_val}. Expected 0 (FEMALE) or 1 (MALE).",
            }, None
    else:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Invalid value for gender: {gender_val}. Expected 'MALE' or 'FEMALE'.",
        }, None

    # 4. Validate Age
    age_val = raw_dict["age"]
    if age_val is None:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": "Feature 'age' cannot be None.",
        }, None

    if isinstance(age_val, bool):
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Invalid value for age: {age_val}. Boolean values are not accepted.",
        }, None

    try:
        age_float = float(age_val)
    except (ValueError, TypeError):
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Invalid non-numeric value for age: '{age_val}'.",
        }, None

    if math.isnan(age_float) or np.isnan(age_float):
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": "Feature 'age' cannot be NaN.",
        }, None

    if math.isinf(age_float) or np.isinf(age_float):
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": "Feature 'age' cannot be infinite.",
        }, None

    if age_float <= 0:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Age must be positive (greater than 0). Received: {age_val}",
        }, None

    if age_float > MAX_AGE:
        return False, {
            "status": "error",
            "error_type": "validation_error",
            "message": f"Age exceeds realistic human limit ({int(MAX_AGE)}). Received: {age_val}",
        }, None

    clean_dict["age"] = float(age_float)

    # 5. Validate the 13 binary symptom/behavioral features
    for symptom in SYMPTOM_FEATURES:
        sym_val = raw_dict[symptom]

        if sym_val is None:
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Feature '{symptom}' cannot be None.",
            }, None

        # Disallow boolean literals (True/False) per specification
        if isinstance(sym_val, bool):
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Invalid value for {symptom}: {sym_val}. Boolean values (True/False) are not accepted. Expected 0, 1, YES, or NO.",
            }, None

        if isinstance(sym_val, str):
            s_clean = sym_val.strip().upper()
            if s_clean in ["YES", "1"]:
                clean_dict[symptom] = 1
            elif s_clean in ["NO", "0"]:
                clean_dict[symptom] = 0
            else:
                return False, {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": f"Invalid value for {symptom}: '{sym_val}'. Expected 0, 1, YES, or NO.",
                }, None
        elif isinstance(sym_val, (int, float)):
            if math.isnan(sym_val) or np.isnan(sym_val):
                return False, {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": f"Feature '{symptom}' cannot be NaN.",
                }, None
            if math.isinf(sym_val) or np.isinf(sym_val):
                return False, {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": f"Feature '{symptom}' cannot be infinite.",
                }, None

            if sym_val in [1, 1.0]:
                clean_dict[symptom] = 1
            elif sym_val in [0, 0.0]:
                clean_dict[symptom] = 0
            else:
                return False, {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": f"Invalid value for {symptom}: {sym_val}. Expected 0, 1, YES, or NO.",
                }, None
        else:
            return False, {
                "status": "error",
                "error_type": "validation_error",
                "message": f"Invalid value for {symptom}: {sym_val}. Expected 0, 1, YES, or NO.",
            }, None

    return True, None, clean_dict


def predict_lung_cancer(
    input_data: Any,
    raise_on_error: bool = False,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Core prediction engine for Lung Cancer Risk Screening.
    Validates input, scales features with Step 10 preprocessor,
    and performs model inference with the Step 11 trained Logistic Regression classifier.

    Args:
        input_data: 15-feature input representation (dict, Series, DataFrame).
        raise_on_error: If True, raises LungCancerValidationError or LungCancerModelError.
        models_dir: Optional custom path to model artifacts directory.

    Returns:
        Dict[str, Any]: Structured prediction output or controlled validation error dictionary.
    """
    # 1. Input validation
    is_valid, error_dict, clean_dict = validate_input(input_data)
    if not is_valid:
        if raise_on_error:
            raise LungCancerValidationError(error_dict["message"])
        return error_dict

    # 2. Load model & preprocessor artifacts
    try:
        preprocessor, model, metadata = load_artifacts(models_dir=models_dir)
    except LungCancerModelError as e:
        if raise_on_error:
            raise e
        return {
            "status": "error",
            "error_type": "artifact_error",
            "message": str(e),
        }
    except Exception as e:
        err_msg = f"Failed to load prediction artifacts: {str(e)}"
        if raise_on_error:
            raise LungCancerModelError(err_msg)
        return {
            "status": "error",
            "error_type": "artifact_error",
            "message": err_msg,
        }

    # 3. Construct DataFrame enforcing canonical feature order
    ordered_features = [{feat: clean_dict[feat] for feat in EXPECTED_PREDICTOR_FEATURES}]
    df_ordered = pd.DataFrame(ordered_features, columns=EXPECTED_PREDICTOR_FEATURES)

    # 4. Transform features with preprocessor (StandardScaler applied strictly to age)
    try:
        X_scaled = preprocessor.transform(df_ordered)
    except Exception as e:
        err_msg = f"Feature transformation failed: {str(e)}"
        if raise_on_error:
            raise LungCancerModelError(err_msg)
        return {
            "status": "error",
            "error_type": "transformation_error",
            "message": err_msg,
        }

    # 5. Model inference
    try:
        pred_class = int(model.predict(X_scaled)[0])
        probs = model.predict_proba(X_scaled)[0]
    except Exception as e:
        err_msg = f"Model inference execution failed: {str(e)}"
        if raise_on_error:
            raise LungCancerModelError(err_msg)
        return {
            "status": "error",
            "error_type": "inference_error",
            "message": err_msg,
        }

    # Class 1 is YES (Lung Cancer Positive)
    prob_yes = float(probs[1])
    risk_percentage = round(prob_yes * 100.0, 2)
    prediction_class = DEFAULT_CLASS_MAPPING.get(pred_class, "YES" if pred_class == 1 else "NO")

    model_name = metadata.get("model_type", DEFAULT_MODEL_NAME)
    threshold = DEFAULT_THRESHOLD

    # Neutral medical screening interpretation
    screening_interpretation = (
        "Higher predicted risk" if pred_class == 1 else "Lower predicted risk"
    )

    return {
        "status": "success",
        "prediction": pred_class,
        "prediction_class": prediction_class,
        "probability": round(prob_yes, 4),
        "risk_percentage": risk_percentage,
        "model": model_name,
        "threshold": threshold,
        "screening_interpretation": screening_interpretation,
    }


def predict_user(
    input_data: Any,
    raise_on_error: bool = False,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    User-facing prediction interface for screening applications.
    Accepts 15 survey features and returns risk assessment details.
    """
    return predict_lung_cancer(
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
    Clinical screening interface.
    Utilizes the identical validated inference engine as predict_user()
    to guarantee strict cross-interface consistency.
    """
    return predict_lung_cancer(
        input_data=input_data,
        raise_on_error=raise_on_error,
        models_dir=models_dir,
    )
