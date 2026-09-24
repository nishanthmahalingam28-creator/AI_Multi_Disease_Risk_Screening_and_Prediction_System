import sys
from pathlib import Path
HERE=Path(__file__).resolve(); MEMBER1_DIR=HERE.parents[2]; sys.path.insert(0,str(MEMBER1_DIR))
from heart.src.split import load_and_split
from heart.src.preprocessing import build_preprocessor
from src.model_experiment import run_model_experiments
REPORT_DIR=MEMBER1_DIR/"heart"/"reports"; RESULT_PATH=REPORT_DIR/"cv_model_results.csv"
def main():
    X_train, X_test, y_train, y_test=load_and_split()
    results,_=run_model_experiments(X_train,y_train,build_preprocessor(),include_xgboost=True)
    results.to_csv(RESULT_PATH,index=False); print(results.to_string(index=False)); print(f"Saved: {RESULT_PATH}")
if __name__=="__main__": main()