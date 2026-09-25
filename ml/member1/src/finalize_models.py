from pathlib import Path
import importlib.util
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]


CONFIG = {
    "heart": (
        "Heart Disease",
        ROOT / "heart/data/cleaned.csv",
        ROOT / "heart/src/preprocessing.py",
        LogisticRegression(max_iter=3000, random_state=42, C=0.1, solver="lbfgs"),
    ),
    "diabetes": (
        "Outcome",
        ROOT / "diabetes/data/cleaned.csv",
        ROOT / "diabetes/src/preprocessing.py",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            max_depth=5,
            min_samples_leaf=4,
            n_estimators=200,
        ),
    ),
    "stroke": (
        "stroke",
        ROOT / "stroke/data/cleaned.csv",
        ROOT / "stroke/src/preprocessing.py",
        SVC(
            probability=True,
            random_state=42,
            class_weight="balanced",
            C=1,
            gamma="scale",
            kernel="linear",
        ),
    ),
}


def load_preprocessor(preprocessing_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"{preprocessing_path.parent.parent.name}_preprocessing",
        preprocessing_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preprocessing module: {preprocessing_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_preprocessor()


for disease, (target, data_path, preprocessing_path, model) in CONFIG.items():
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not preprocessing_path.exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")

    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target]), df[target]

    preprocessor = load_preprocessor(preprocessing_path)
    Xt = preprocessor.fit_transform(X)
    model.fit(Xt, y)

    out = ROOT / disease / "artifacts"
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / f"{disease}_model.joblib")
    joblib.dump(preprocessor, out / f"{disease}_preprocessor.joblib")

    metadata = {
        "disease": disease,
        "target_column": target,
        "positive_class": 1,
        "training_rows": len(df),
        "feature_columns": list(X.columns),
        "model_class": model.__class__.__name__,
        "model_parameters": model.get_params(),
        "screening_only": True,
        "medical_diagnosis": False,
    }

    (out / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "class_mapping.json").write_text(
        json.dumps({"0": "negative", "1": "positive"}, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] {disease}")
