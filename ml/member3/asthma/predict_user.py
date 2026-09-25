from pathlib import Path
import joblib,pandas as pd

MODEL_PATH = Path(__file__).resolve().parent / "asthma_model.joblib"

def _load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Asthma model is unavailable. Run `python -m ml.member3.asthma.train` first.")
    return joblib.load(MODEL_PATH)

def predict_user(features:dict)->dict:
    model = _load_model()
    row=pd.DataFrame([features]); p=float(model.predict_proba(row)[0,1]); y=int(model.predict(row)[0])
    return {"disease":"asthma","module":"user","prediction":y,"probability":p,"risk_category":"elevated_screening_signal" if p>=.5 else "lower_screening_signal","disclaimer":"Screening/risk prediction only; not a medical diagnosis."}
