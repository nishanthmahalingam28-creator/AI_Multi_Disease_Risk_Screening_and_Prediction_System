# Stroke — Phase 6 Preprocessing Report

## Target
stroke

## Features
Numerical: age, hypertension, heart_disease, avg_glucose_level, bmi.
Categorical: gender, ever_married, work_type, Residence_type, smoking_status.

## Excluded
id is excluded because it is an identifier and was removed from the Phase 3 cleaned dataset.

## Strategy
Numerical: median imputation + StandardScaler.
Categorical: most-frequent imputation + OneHotEncoder(handle_unknown="ignore").

BMI has 201 missing values. Unknown smoking status and Other gender are preserved.

Stroke class imbalance is not addressed in Phase 6: no SMOTE, oversampling, undersampling, class weighting, or threshold adjustment.

## Leakage prevention
The preprocessor is returned unfitted and must be fitted only on training data or inside a later CV pipeline.

## Expected transformed width
21 features from the observed cleaned-data category levels.

No model training, CV, tuning, resampling, or final model selection is included in Phase 6.
