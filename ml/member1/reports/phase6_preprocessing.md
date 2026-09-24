# Member 1 — Phase 6 Data Preprocessing

## Status
COMPLETE — preprocessing pipeline design and implementation only.

## Heart Disease
- 13 predictors: 5 numerical and 8 categorical/discrete.
- Numerical: median imputation + StandardScaler.
- Categorical/discrete: most-frequent imputation + OneHotEncoder(handle_unknown="ignore").
- Expected transformed width: 28.

## Diabetes
- 8 numerical predictors.
- Median imputation + StandardScaler.
- Pregnancies = 0 remains valid.
- Expected transformed width: 8.

## Stroke
- 10 predictors: 5 numerical and 5 categorical.
- id excluded.
- Numerical: median imputation + StandardScaler.
- Categorical: most-frequent imputation + OneHotEncoder(handle_unknown="ignore").
- Unknown smoking status and Other gender retained.
- No imbalance treatment in Phase 6.
- Expected transformed width: 21.

## Leakage prevention
All preprocessors are returned unfitted. Later training must split first and fit preprocessing only on training data. For cross-validation, preprocessing must be inside the model pipeline.

## Validation
A preprocessing-only pytest smoke test is included at ml/member1/tests/test_preprocessing.py. It checks target exclusion, predictor coverage, Stroke id exclusion, stratified split preprocessing, transformed dimensions, and feature-name consistency.

## Deferred
Model training, cross-validation, imbalance strategy evaluation, hyperparameter tuning, evaluation, final model selection, saved production artifacts, prediction functions, and API integration.
