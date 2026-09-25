from pathlib import Path
import json

import joblib
import pandas as pd


def predict_one(artifacts_dir, features, disease=None):
    """Run one screening prediction using saved model artifacts."""
    d = Path(artifacts_dir)
    metadata = json.loads((d / "model_metadata.json").read_text(encoding="utf-8"))

    disease_name = disease or metadata["disease"]
    model_path = d / f"{disease_name}_model.joblib"
    preprocessor_path = d / f"{disease_name}_preprocessor.joblib"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Saved model artifacts not found in {d}: "
            f"{model_path.name}, {preprocessor_path.name}"
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    expected = metadata["feature_columns"]
    missing = [column for column in expected if column not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([{column: features[column] for column in expected}])
    Xt = preprocessor.transform(X)
    prediction = int(model.predict(Xt)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(model.predict_proba(Xt)[0, classes.index(1)])

    risk_category = "unknown"
    if probability is not None:
        risk_category = (
            "low" if probability < 0.30
            else "moderate" if probability < 0.60
            else "high"
        )

    return {
        "disease": metadata["disease"],
        "module": "member1",
        "prediction": prediction,
        "probability": probability,
        "risk_category": risk_category,
        "screening_only": True,
    }
