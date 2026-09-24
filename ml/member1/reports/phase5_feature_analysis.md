# Member 1 — Phase 5 Feature Analysis & Feature Engineering

## Status
COMPLETE

Phase 5 was completed for Heart Disease, Diabetes, and Stroke using the Phase 3 cleaned datasets and Phase 4 EDA findings.

## Feature decisions

### Heart Disease
All cleaned predictive features are retained. Heart Disease remains the target and is excluded from predictors. No identifier or target-derived feature was identified.

### Diabetes
All eight cleaned predictors are retained. Outcome remains the target. Biological-zero cleaning from Phase 3 is preserved, and Pregnancies = 0 remains valid.

### Stroke
All ten predictive features in the cleaned dataset are retained. stroke remains the target. id remains excluded because it is an identifier removed in Phase 3. smoking_status = Unknown and gender = Other are retained.

## Feature engineering

No new predictive features were added in Phase 5. This avoids arbitrary combinations, target leakage, and premature learned transformations. Disease-specific feature_engineering.py modules provide a reusable non-learning interface.

## Leakage assessment

No target-derived columns were identified in the Phase 3 cleaned feature sets. No identifier is used as a predictor. No post-outcome information was introduced.

## Future preprocessing plan

These are later-phase considerations, not completed work:
- Diabetes and Stroke missing values must be handled in training-only preprocessing.
- Categorical/discrete variables require appropriate encoding.
- Numerical scaling/transformation should be evaluated by model family.
- Feature selection should be validated with training/CV evidence.
- Stroke class imbalance should be addressed/evaluated only in the later training stage using leakage-safe methods.

## Integrity and scope

Phase 5 did not modify raw datasets, modify Phase 3 cleaned CSVs, remove rows, impute missing values, fit scalers or encoders, perform SMOTE/resampling, train models, perform cross-validation, tune hyperparameters, select a final model, or implement APIs.