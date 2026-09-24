from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
HERE=Path(__file__).resolve(); MEMBER1_DIR=HERE.parents[2]; sys.path.insert(0,str(MEMBER1_DIR))
from stroke.src.split import load_and_split
from stroke.src.preprocessing import build_preprocessor
from src.hyperparameter_tuning import tune_models
def main():
    X_train,_,y_train,_=load_and_split()
    spaces={
      "Logistic Regression":(LogisticRegression(max_iter=3000,random_state=42,class_weight="balanced"),{"model__C":[0.01,0.1,1,10],"model__solver":["liblinear","lbfgs"]}),
      "SVM":(SVC(probability=True,random_state=42,class_weight="balanced"),{"model__C":[0.1,1,10],"model__gamma":["scale","auto"],"model__kernel":["rbf","linear"]}),
      "Random Forest":(RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced"),{"model__n_estimators":[200,400],"model__max_depth":[None,5,10],"model__min_samples_leaf":[1,2,4],"model__class_weight":["balanced","balanced_subsample"]}),
      "Gradient Boosting":(GradientBoostingClassifier(random_state=42),{"model__n_estimators":[100,200],"model__learning_rate":[0.03,0.1],"model__max_depth":[2,3]})
    }
    results,_=tune_models(X_train,y_train,build_preprocessor(),spaces,scoring="average_precision")
    out=MEMBER1_DIR/"stroke"/"reports"/"tuning_results.csv"; results.to_csv(out,index=False); print(results.to_string(index=False))
if __name__=="__main__": main()
