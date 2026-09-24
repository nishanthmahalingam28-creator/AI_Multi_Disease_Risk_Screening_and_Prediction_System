from pathlib import Path
import json
import joblib
import pandas as pd

def predict_one(artifacts_dir, features):
    d = Path(artifacts_dir)
    model = joblib.load(d / "model.joblib")
    preprocessor = joblib.load(d / "preprocessor.joblib")
    metadata = json.loads((d / "model_metadata.json").read_text(encoding="utf-8"))

    expected = metadata["feature_columns"]
    missing = [c for c in expected if c not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([{c: features[c] for c in expected}])
    Xt = preprocessor.transform(X)
    pred = int(model.predict(Xt)[0])

    prob = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            prob = float(model.predict_proba(Xt)[0, classes.index(1)])

    risk = "unknown" if prob is None else (
        "low" if prob < 0.30 else "moderate" if prob < 0.60 else "high"
    )

    return {
        "disease": metadata["disease"],
        "module": "member1",
        "prediction": pred,
        "probability": prob,
        "risk_category": risk,
        "screening_only": True
    }
