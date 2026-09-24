# Member 1 — Phase 10 Multiple Model Training

## Scope
Candidate classification models are defined for Heart Disease, Diabetes and Stroke.

## Candidates
1. Logistic Regression
2. Support Vector Machine
3. Random Forest
4. Gradient Boosting
5. XGBoost, only when the dependency is available

## Leakage prevention
Every experiment wraps the disease-specific preprocessing transformer and model in a scikit-learn Pipeline. Cross-validation therefore fits preprocessing separately within each training fold.

The untouched test set from Phase 7 is not used for candidate comparison or tuning.

## Reproducibility
- random_state = 42
- StratifiedKFold = 5 folds
- shuffle = True

## Phase boundary
This phase defines and runs the candidate experiment framework. It does not select a final model. Final tuning and test-set evaluation remain later steps.

Actual performance values must come from execution; no estimated metrics are recorded.
