# Stroke — Phase 5 Feature Analysis

## Feature inventory

| Feature | Role/type | Decision | Reason |
|---|---|---|---|
| gender | Categorical | Retain | Preserve observed categories, including Other. |
| age | Numerical | Retain | Largest absolute numerical target association in Phase 4; no leakage identified. |
| hypertension | Binary | Retain | Predictor available before prediction; no leakage identified. |
| heart_disease | Binary | Retain | Existing predictor; no target-derived transformation created. |
| ever_married | Categorical | Retain | Preserve observed categories. |
| work_type | Categorical | Retain | Preserve observed categories. |
| Residence_type | Categorical | Retain | Preserve observed categories. |
| avg_glucose_level | Numerical | Retain | Numeric predictor; potential outliers are not removed. |
| bmi | Numerical | Retain | Numeric predictor; 201 missing values remain unresolved. |
| smoking_status | Categorical | Retain | Preserve Unknown as an observed category. |
| id | Identifier | Exclude | Removed during Phase 3; not a predictive feature. |

Target stroke is not a predictor.

## Leakage assessment

No target-derived predictor was identified. id remains excluded. No post-outcome feature was created.

The severe class imbalance is not treated as a feature-engineering problem. No SMOTE, oversampling, undersampling, or class weighting is performed in Phase 5.

## Feature engineering decision

No new engineered predictor is created. The EDA supports preserving existing interpretable predictors rather than introducing arbitrary risk scores or target-derived interactions.

## Later handling plan
- Categorical variables: evaluate appropriate encoding during preprocessing.
- BMI: evaluate training-only imputation.
- Numerical variables: assess scaling or transformations by model family.
- Potential outliers: retain for now and evaluate model robustness later.
- Class imbalance: address or evaluate only in the later training stage using leakage-safe methods.

## Status

Phase 5 feature analysis is complete. The cleaned dataset structure is preserved and no predictive identifier is reintroduced.