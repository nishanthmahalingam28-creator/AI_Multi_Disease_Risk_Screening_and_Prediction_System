from pathlib import Path
import joblib,pandas as pd
MODEL=joblib.load(Path(__file__).resolve().parent/"parkinsons_model.joblib")
def predict_user(features:dict)->dict:
    row=pd.DataFrame([features]); p=float(MODEL.predict_proba(row)[0,1]); y=int(MODEL.predict(row)[0])
    return {"disease":"parkinsons","module":"user","prediction":y,"probability":p,"risk_category":"elevated_screening_signal" if p>=.5 else "lower_screening_signal","disclaimer":"Screening/risk prediction only; not a medical diagnosis."}
