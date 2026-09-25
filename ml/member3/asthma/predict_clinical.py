from .predict_user import predict_user
def predict_clinical(features:dict)->dict:
    r=predict_user(features); r["module"]="clinical"; return r
