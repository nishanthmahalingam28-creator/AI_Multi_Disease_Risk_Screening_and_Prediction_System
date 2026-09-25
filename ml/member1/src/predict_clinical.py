from pathlib import Path

from ml.member1.src.predict_saved import predict_one


MEMBER1_DIR = Path(__file__).resolve().parents[1]

ARTIFACT_DIRS = {
    "heart": MEMBER1_DIR / "heart" / "artifacts",
    "diabetes": MEMBER1_DIR / "diabetes" / "artifacts",
    "stroke": MEMBER1_DIR / "stroke" / "artifacts",
}


def predict_heart(features):
    return predict_one(
        ARTIFACT_DIRS["heart"],
        features,
        disease="heart",
    )


def predict_diabetes(features):
    return predict_one(
        ARTIFACT_DIRS["diabetes"],
        features,
        disease="diabetes",
    )


def predict_stroke(features):
    return predict_one(
        ARTIFACT_DIRS["stroke"],
        features,
        disease="stroke",
    )