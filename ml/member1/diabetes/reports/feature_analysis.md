# Diabetes — Phase 5 Feature Analysis

## Feature inventory

| Feature | Role/type | Decision | Reason |
|---|---|---|---|
| Pregnancies | Numerical/count | Retain | Zero is valid and must not be treated as missing. |
| Glucose | Numerical | Retain | Strongest descriptive target association in Phase 4; missing values remain unresolved. |
| BloodPressure | Numerical | Retain | Numeric predictor; missing values require later training-only handling. |
| SkinThickness | Numerical | Retain | Retain despite substantial missingness. |
| Insulin | Numerical | Retain | Retain despite high missingness. |
| BMI | Numerical | Retain | Numeric predictor; missingness remains unresolved. |
| DiabetesPedigreeFunction | Numerical | Retain | Predictor with no leakage identified. |
| Age | Numerical | Retain | Predictor with descriptive association; no leakage identified. |

Target Outcome is not a predictor.

## Leakage assessment

No target-derived predictor or identifier was identified in the cleaned feature set. The target is kept separate.

## Missing-value decision

Phase 5 does not impute values. The Phase 3 biological-zero conversion remains in place:

- Glucose: 5 missing
- BloodPressure: 35 missing
- SkinThickness: 227 missing
- Insulin: 374 missing
- BMI: 11 missing

Pregnancies = 0 remains valid.

Later imputation, if selected, must be fitted only on training data and included inside cross-validation pipelines.

## Feature engineering decision

No arbitrary risk score or target-derived feature is created. Skewed-feature transformations may be compared later inside the training pipeline if justified by model behavior and validation results.

## Status

Phase 5 feature analysis is complete without modifying the cleaned dataset.