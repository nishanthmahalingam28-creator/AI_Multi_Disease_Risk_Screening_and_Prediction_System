from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"dataset"/"Parkinsons_Disease_Dataset.xlsx"; OUT=ROOT/"parkinsons"; REPORTS=OUT/"reports"; RANDOM_STATE=42; TARGET="status"; ID_COL="name"

def load_data():
    df=pd.read_excel(DATA); df.columns=[str(c).strip() for c in df.columns]
    if TARGET not in df or ID_COL not in df: raise ValueError("Parkinson dataset must contain name and status columns.")
    for c in df.select_dtypes(include="object"): df[c]=df[c].map(lambda x:x.strip() if isinstance(x,str) else x)
    dups=int(df.duplicated().sum())
    if dups: df=df.drop_duplicates().reset_index(drop=True)
    df[TARGET]=pd.to_numeric(df[TARGET],errors="coerce"); df=df[df[TARGET].isin([0,1])].copy()
    extracted=df[ID_COL].astype(str).str.extract(r"(S\d+)",expand=False); df["_subject_id"]=extracted
    if df["_subject_id"].isna().any(): df["_subject_id"]=df[ID_COL].astype(str).str.rsplit("_",n=1).str[0]
    return df,dups

def eda(df):
    REPORTS.mkdir(parents=True,exist_ok=True); sns.set_theme(style="whitegrid"); numeric=[c for c in df.select_dtypes(include="number").columns if c!=TARGET]
    if numeric:
        plt.figure(figsize=(12,9)); sns.heatmap(df[numeric+[TARGET]].corr(numeric_only=True),cmap="vlag",center=0); plt.tight_layout(); plt.savefig(REPORTS/"correlation_heatmap.png",dpi=140); plt.close()
        plt.figure(figsize=(12,8)); sns.boxplot(data=df[numeric],orient="h"); plt.tight_layout(); plt.savefig(REPORTS/"numeric_boxplots.png",dpi=140); plt.close()
    plt.figure(figsize=(5,4)); sns.countplot(data=df,x=TARGET); plt.tight_layout(); plt.savefig(REPORTS/"class_distribution.png",dpi=140); plt.close()

def prep(X):
    nums=X.select_dtypes(include="number").columns.tolist()
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),nums)])

def compare(X,y,g):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=RANDOM_STATE)
    models={"logistic_regression":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=RANDOM_STATE),"svm":CalibratedClassifierCV(SVC(class_weight="balanced",random_state=RANDOM_STATE),method="sigmoid",cv=5,ensemble=False),"random_forest":RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1),"gradient_boosting":GradientBoostingClassifier(random_state=RANDOM_STATE)}
    rows=[]
    for name,m in models.items():
        pipe=Pipeline([("preprocessor",prep(X)),("model",m)])
        s=cross_validate(pipe,X,y,groups=g,cv=cv,scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc","balanced_accuracy":"balanced_accuracy"},n_jobs=-1)
        rows.append({"model":name,**{k:float(s["test_"+k].mean()) for k in ["accuracy","precision","recall","f1","roc_auc","balanced_accuracy"]},"roc_auc_std":float(s["test_roc_auc"].std())})
    r=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); r.to_csv(REPORTS/"model_comparison.csv",index=False); return r

def holdout(X,y,g):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=RANDOM_STATE); tr,te=next(cv.split(X,y,g)); tg=set(g.iloc[tr]); eg=set(g.iloc[te])
    if tg & eg: raise RuntimeError("Subject leakage detected between training and test groups.")
    return X.iloc[tr],X.iloc[te],y.iloc[tr],y.iloc[te],g.iloc[tr]

def tune(X,y,g,names):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=RANDOM_STATE); out=[]
    for name in names:
        if name=="random_forest": est=RandomForestClassifier(class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1); grid={"model__n_estimators":[200,400],"model__max_depth":[None,8,16],"model__min_samples_split":[2,5]}
        elif name=="svm": est=CalibratedClassifierCV(SVC(class_weight="balanced",random_state=RANDOM_STATE),method="sigmoid",cv=5,ensemble=False); grid={"model__estimator__C":[.1,1,10],"model__estimator__kernel":["rbf","linear"],"model__estimator__gamma":["scale","auto"]}
        elif name=="logistic_regression": est=LogisticRegression(max_iter=3000,class_weight="balanced",random_state=RANDOM_STATE); grid={"model__C":[.1,1,10],"model__solver":["lbfgs","liblinear"]}
        else: est=GradientBoostingClassifier(random_state=RANDOM_STATE); grid={"model__n_estimators":[100,200],"model__learning_rate":[.03,.1],"model__max_depth":[2,3]}
        s=GridSearchCV(Pipeline([("preprocessor",prep(X)),("model",est)]),grid,scoring="roc_auc",cv=cv,n_jobs=-1,refit=True); s.fit(X,y,groups=g); out.append((name,s.best_score_,s.best_estimator_,s.best_params_))
    out.sort(key=lambda x:x[1],reverse=True); pd.DataFrame([{"model":n,"cv_roc_auc":s,"best_params":json.dumps(p)} for n,s,_,p in out]).to_csv(REPORTS/"hyperparameter_tuning.csv",index=False); return out[0]

def main():
    REPORTS.mkdir(parents=True,exist_ok=True)
    df,dups=load_data(); eda(df)
    pd.DataFrame({"dtype":df.dtypes.astype(str),"missing":df.isna().sum(),"missing_pct":(df.isna().mean()*100).round(3),"n_unique":df.nunique(dropna=True)}).to_csv(REPORTS/"data_profile.csv")
    json.dump(df[TARGET].value_counts().sort_index().to_dict(),open(REPORTS/"class_distribution.json","w"),indent=2)
    X=df.drop(columns=[TARGET,ID_COL,"_subject_id"]); y=df[TARGET].astype(int); groups=df["_subject_id"].astype(str)
    Xtr,Xte,ytr,yte,gtr=holdout(X,y,groups); comparison=compare(Xtr,ytr,gtr); best_name,cv_auc,model,params=tune(Xtr,ytr,gtr,comparison.head(2)["model"].tolist()); model.fit(Xtr,ytr)
    pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
    metrics={"accuracy":accuracy_score(yte,pred),"balanced_accuracy":balanced_accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,prob),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"classification_report":classification_report(yte,pred,output_dict=True,zero_division=0)}
    json.dump(metrics,open(REPORTS/"evaluation_report.json","w"),indent=2,default=float)
    joblib.dump(model,OUT/"parkinsons_model.joblib"); joblib.dump(model.named_steps["preprocessor"],OUT/"parkinsons_preprocessor.joblib")
    features=X.columns.tolist(); mapping={"0":"Healthy","1":"Parkinson's disease screening positive"}; schema={"disease":"parkinsons","target":TARGET,"target_mapping":mapping,"features":[{"name":c,"dtype":str(X[c].dtype)} for c in features],"input_note":"Specialized voice measurements may require an appropriate measurement system.","disclaimer":"Screening/risk prediction only; not a medical diagnosis."}
    for f in ["user_schema.json","clinical_schema.json"]: json.dump(schema,open(OUT/f,"w"),indent=2)
    json.dump(mapping,open(OUT/"class_mapping.json","w"),indent=2)
    metadata={"disease":"parkinsons","model_name":best_name,"model_version":"member3a-1.0","dataset_name":DATA.name,"training_date":date.today().isoformat(),"training_samples":len(Xtr),"features":features,"target":TARGET,"class_labels":mapping,"preprocessing_version":"median-impute + standardization","validation_method":"StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)","group_column":"subject_id extracted from name","cv_roc_auc":cv_auc,"evaluation_metrics":metrics,"number_of_subjects":int(groups.nunique()),"known_limitations":["Screening/research decision support only; not diagnosis.","Performance depends on supplied voice measurements and dataset population."],"duplicate_recordings_removed":dups}
    json.dump(metadata,open(OUT/"model_metadata.json","w"),indent=2,default=str)

if __name__=="__main__": main()



