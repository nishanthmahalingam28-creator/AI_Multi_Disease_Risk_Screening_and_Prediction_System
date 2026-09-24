# Diabetes — Multiple Model Training

Phase 10 starts the candidate-model comparison.

Candidates:
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost when the dependency is available

Each candidate is evaluated through a pipeline containing the Diabetes preprocessing transformer. Cross-validation uses stratified 5-fold validation on the training set only.

The Phase 3 biological-zero conversions and Diabetes Pregnancies = 0 handling are preserved. No final model is selected in this phase.
