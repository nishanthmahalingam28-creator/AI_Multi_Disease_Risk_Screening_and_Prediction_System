from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import joblib, matplotlib.pyplot as plt, pandas as pd, seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"dataset"/"synthetic_asthma_dataset.csv"; OUT=ROOT/"asthma"; REPORTS=OUT/"reports"; RANDOM_STATE=42; TARGET="Has_Asthma"; DROP=["Patient_ID","Asthma_Control_Level"]

def load_and_clean():
    df=pd.read_csv(DATA); original_shape=df.shape; df.columns=[c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object"): df[c]=df[c].map(lambda x:x.strip() if isinstance(x,str) else x)
    duplicates=int(df.duplicated().sum())
    if duplicates: df=df.drop_duplicates().reset_index(drop=True)
    if TARGET not in df: raise ValueError(f"Missing required target column: {TARGET}")
    df[TARGET]=pd.to_numeric(df[TARGET],errors="coerce"); df=df[df[TARGET].isin([0,1])].copy()
    return df,original_shape,duplicates

def profile(df):
    p=pd.DataFrame({"dtype":df.dtypes.astype(str),"missing":df.isna().sum(),"missing_pct":(df.isna().mean()*100).round(3),"n_unique":df.nunique(dropna=True)})
    p.to_csv(REPORTS/"data_profile.csv")
    with open(REPORTS/"class_distribution.json","w") as f: json.dump(df[TARGET].value_counts().sort_index().to_dict(),f,indent=2)
    return df.isna().sum().sort_values(ascending=False)

def make_eda(df):
    REPORTS.mkdir(parents=True,exist_ok=True); sns.set_theme(style="whitegrid")
    num=[c for c in df.select_dtypes(include="number").columns if c!=TARGET]
    for col in num:
        plt.figure(figsize=(7,4)); sns.histplot(data=df,x=col,hue=TARGET,element="step",stat="density",common_norm=False); plt.tight_layout(); plt.savefig(REPORTS/f"hist_{col}.png",dpi=140); plt.close()
    if num:
        plt.figure(figsize=(10,8)); sns.heatmap(df[num+[TARGET]].corr(numeric_only=True),cmap="vlag",center=0); plt.tight_layout(); plt.savefig(REPORTS/"correlation_heatmap.png",dpi=140); plt.close()
        plt.figure(figsize=(10,7)); sns.boxplot(data=df[num],orient="h"); plt.tight_layout(); plt.savefig(REPORTS/"numeric_boxplots.png",dpi=140); plt.close()
    plt.figure(figsize=(5,4)); sns.countplot(data=df,x=TARGET); plt.tight_layout(); plt.savefig(REPORTS/"class_distribution.png",dpi=140); plt.close()

def build_pipeline(X):
    numeric=X.select_dtypes(include="number").columns.tolist(); categorical=X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical)])

def compare(X,y):
    cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE)
    models={"logistic_regression":LogisticRegression(max_iter=3000,class_weight="balanced",random_state=RANDOM_STATE),"svm":CalibratedClassifierCV(SVC(class_weight="balanced",random_state=RANDOM_STATE),method="sigmoid",cv=5,ensemble=False),"random_forest":RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1),"gradient_boosting":GradientBoostingClassifier(random_state=RANDOM_STATE)}
    rows=[]
    for name,m in models.items():
        pipe=Pipeline([("preprocessor",build_pipeline(X)),("model",m)])
        s=cross_validate(pipe,X,y,cv=cv,scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc","balanced_accuracy":"balanced_accuracy"},n_jobs=-1)
        rows.append({"model":name,**{k:float(s["test_"+k].mean()) for k in ["accuracy","precision","recall","f1","roc_auc","balanced_accuracy"]},"roc_auc_std":float(s["test_roc_auc"].std())})
    r=pd.DataFrame(rows).sort_values("roc_auc",ascending=False); r.to_csv(REPORTS/"model_comparison.csv",index=False); return r

def tune(X,y,names):
    cv=StratifiedKFold(5,shuffle=True,random_state=RANDOM_STATE); out=[]
    for name in names:
        if name=="random_forest": est=RandomForestClassifier(class_weight="balanced",random_state=RANDOM_STATE,n_jobs=-1); grid={"model__n_estimators":[200,400],"model__max_depth":[None,8,16],"model__min_samples_split":[2,5]}
        elif name=="svm": est=CalibratedClassifierCV(SVC(class_weight="balanced",random_state=RANDOM_STATE),method="sigmoid",cv=5,ensemble=False); grid={"model__estimator__C":[0.1,1,10],"model__estimator__kernel":["rbf","linear"],"model__estimator__gamma":["scale","auto"]}
        elif name=="logistic_regression": est=LogisticRegression(max_iter=3000,class_weight="balanced",random_state=RANDOM_STATE); grid={"model__C":[0.1,1,10],"model__solver":["lbfgs","liblinear"]}
        else: est=GradientBoostingClassifier(random_state=RANDOM_STATE); grid={"model__n_estimators":[100,200],"model__learning_rate":[0.03,0.1],"model__max_depth":[2,3]}
        search=GridSearchCV(Pipeline([("preprocessor",build_pipeline(X)),("model",est)]),grid,scoring="roc_auc",cv=cv,n_jobs=-1,refit=True); search.fit(X,y); out.append((name,search.best_score_,search.best_estimator_,search.best_params_))
    out.sort(key=lambda x:x[1],reverse=True); pd.DataFrame([{"model":n,"cv_roc_auc":s,"best_params":json.dumps(p)} for n,s,_,p in out]).to_csv(REPORTS/"hyperparameter_tuning.csv",index=False); return out[0]

def main():
    OUT.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    df,shape,dups=load_and_clean(); missing=profile(df); make_eda(df)
    X=df.drop(columns=[TARGET]+DROP,errors="ignore"); y=df[TARGET].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=RANDOM_STATE)
    comparison=compare(Xtr,ytr); best_name,cv_auc,model,params=tune(Xtr,ytr,comparison.head(2)["model"].tolist()); model.fit(Xtr,ytr)
    pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
    metrics={"accuracy":accuracy_score(yte,pred),"balanced_accuracy":balanced_accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),"roc_auc":roc_auc_score(yte,prob),"pr_auc":average_precision_score(yte,prob),"confusion_matrix":confusion_matrix(yte,pred).tolist(),"classification_report":classification_report(yte,pred,output_dict=True,zero_division=0)}
    json.dump(metrics,open(REPORTS/"evaluation_report.json","w"),indent=2,default=float)
    joblib.dump(model,OUT/"asthma_model.joblib"); joblib.dump(model.named_steps["preprocessor"],OUT/"asthma_preprocessor.joblib")
    features=X.columns.tolist(); mapping={"0":"No asthma","1":"Has asthma"}
    schema={"disease":"asthma","target":TARGET,"target_mapping":mapping,"features":[{"name":c,"dtype":str(X[c].dtype)} for c in features],"note":"Screening/risk prediction only; not a medical diagnosis."}
    for f in ["user_schema.json","clinical_schema.json"]: json.dump(schema,open(OUT/f,"w"),indent=2)
    json.dump(mapping,open(OUT/"class_mapping.json","w"),indent=2)
    metadata={"disease":"asthma","model_name":best_name,"model_version":"member3a-1.0","dataset_name":DATA.name,"training_date":date.today().isoformat(),"training_samples":len(Xtr),"features":features,"target":TARGET,"class_labels":mapping,"preprocessing_version":"median-impute/standardize numeric + most-frequent/one-hot categorical","validation_method":"StratifiedKFold(n_splits=5, shuffle=True, random_state=42)","cv_roc_auc":cv_auc,"evaluation_metrics":metrics,"known_limitations":["The asthma dataset is synthetic.","Performance does not establish clinical validity or real-world medical effectiveness.","Outputs are screening/risk estimates, not medical diagnosis."],"removed_columns":DROP,"original_shape":list(shape),"duplicate_rows_removed":dups,"missing_values":missing[missing>0].to_dict()}
    json.dump(metadata,open(OUT/"model_metadata.json","w"),indent=2,default=str)

if __name__=="__main__": main()

