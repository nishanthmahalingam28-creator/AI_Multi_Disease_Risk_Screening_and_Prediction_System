# Heart Disease — Phase 6 Preprocessing Report

## Target
Heart Disease

## Features
Numerical: Age, BP, Cholesterol, Max HR, ST depression.
Categorical/discrete: Sex, Chest pain type, FBS over 120, EKG results, Exercise angina, Slope of ST, Number of vessels fluro, Thallium.

## Strategy
Numerical: median imputation + StandardScaler.
Categorical/discrete: most-frequent imputation + OneHotEncoder(handle_unknown="ignore").

The Phase 3 cleaned dataset has no explicit missing values. The imputer remains in the pipeline for a consistent later training/inference contract.

## Leakage prevention
The preprocessor is returned unfitted and must be fitted only on training data or inside a later CV pipeline.

## Expected transformed width
28 features from the observed cleaned-data category levels.

No model training, CV, tuning, resampling, or final model selection is included in Phase 6.
