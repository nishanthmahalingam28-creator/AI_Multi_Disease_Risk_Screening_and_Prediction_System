"""Member 1 baseline hyperparameter tuning utilities.

Tuning is performed only on the Phase 7 training split. Every candidate is
wrapped with the supplied preprocessing pipeline so preprocessing is fitted
inside each CV fold.
"""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

RANDOM_STATE=42
CV=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)

def tune_models(X,y,preprocessor,search_spaces,scoring="roc_auc"):
    rows=[]; searches={}
    for name,(model,params) in search_spaces.items():
        pipe=Pipeline([("preprocessor",preprocessor),("model",model)])
        grid=GridSearchCV(pipe,params,cv=CV,scoring=scoring,n_jobs=-1,refit=True,return_train_score=False)
        grid.fit(X,y)
        rows.append({"model":name,"best_cv_score":grid.best_score_,"best_params":str(grid.best_params_)})
        searches[name]=grid
    return pd.DataFrame(rows),searches
