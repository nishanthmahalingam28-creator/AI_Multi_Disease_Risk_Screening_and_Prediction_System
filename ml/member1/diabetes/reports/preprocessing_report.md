# Diabetes — Phase 6 Preprocessing Report

## Target
Outcome

## Features
All 8 predictors are numerical: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

## Strategy
Median imputation + StandardScaler.

Phase 3 converted biological-zero values to missing for Glucose, BloodPressure, SkinThickness, Insulin, and BMI. Pregnancies = 0 remains valid and is not imputed as missing.

## Leakage prevention
The preprocessor is returned unfitted and must be fitted only on training data or inside a later CV pipeline.

## Expected transformed width
8 features.

No model training, CV, tuning, resampling, or final model selection is included in Phase 6.
