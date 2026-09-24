from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT))
from diabetes.src.split import load_and_split
from diabetes.src.preprocessing import build_preprocessor
from src.tuned_model_evaluation import evaluate_models,save_results
X_train,X_test,y_train,y_test=load_and_split()
specs={"Logistic Regression":(LogisticRegression(max_iter=3000,random_state=42),{"C":0.1,"solver":"lbfgs"}),"SVM":(SVC(probability=True,random_state=42),{"C":10,"gamma":"scale","kernel":"linear"}),"Random Forest":(RandomForestClassifier(random_state=42,n_jobs=-1),{"max_depth":5,"min_samples_leaf":4,"n_estimators":200}),"Gradient Boosting":(GradientBoostingClassifier(random_state=42),{"learning_rate":0.03,"max_depth":2,"n_estimators":100})}
results,details=evaluate_models(X_train,X_test,y_train,y_test,build_preprocessor(),specs,positive_class=1)
out=ROOT/"diabetes"/"reports"; save_results(results,details,out/"tuned_test_results.csv",out/"tuned_test_details.json"); print(results.to_string(index=False))
