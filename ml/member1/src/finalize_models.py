from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "heart": ("Heart Disease", ROOT / "heart/data/cleaned.csv", "heart.src.preprocessing",
              LogisticRegression(max_iter=3000, random_state=42, C=0.1, solver="lbfgs")),
    "diabetes": ("Outcome", ROOT / "diabetes/data/cleaned.csv", "diabetes.src.preprocessing",
                 RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, min_samples_leaf=4, n_estimators=200)),
    "stroke": ("stroke", ROOT / "stroke/data/cleaned.csv", "stroke.src.preprocessing",
               SVC(probability=True, random_state=42, class_weight="balanced", C=1, gamma="scale", kernel="linear")),
}

for disease, (target, data_path, module_name, model) in CONFIG.items():
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target]), df[target]
    module = __import__(module_name, fromlist=["build_preprocessor"])
    preprocessor = module.build_preprocessor()
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
        "medical_diagnosis": False
    }
    (out / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    (out / "class_mapping.json").write_text(
        json.dumps({"0": "negative", "1": "positive"}, indent=2), encoding="utf-8"
    )
    print(f"[OK] {disease}")
