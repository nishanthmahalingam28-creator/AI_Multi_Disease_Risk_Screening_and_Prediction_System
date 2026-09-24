from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path.insert(0,str(ROOT))
from heart.src.split import load_and_split
from heart.src.preprocessing import build_preprocessor
from src.tuned_model_evaluation import evaluate_models,save_results
X_train,X_test,y_train,y_test=load_and_split()
specs={"Logistic Regression":(LogisticRegression(max_iter=3000,random_state=42),{"C":0.1,"solver":"lbfgs"}),"SVM":(SVC(probability=True,random_state=42),{"C":1,"gamma":"auto","kernel":"rbf"}),"Random Forest":(RandomForestClassifier(random_state=42,n_jobs=-1),{"max_depth":5,"min_samples_leaf":2,"n_estimators":400}),"Gradient Boosting":(GradientBoostingClassifier(random_state=42),{"learning_rate":0.1,"max_depth":3,"n_estimators":200})}
results,details=evaluate_models(X_train,X_test,y_train,y_test,build_preprocessor(),specs,positive_class="Presence")
out=ROOT/"heart"/"reports"; save_results(results,details,out/"tuned_test_results.csv",out/"tuned_test_details.json"); print(results.to_string(index=False))
