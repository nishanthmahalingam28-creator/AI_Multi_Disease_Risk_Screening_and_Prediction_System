# Stroke — Multiple Model Training

Phase 10 starts the candidate-model comparison.

Candidates:
- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost when the dependency is available

Stroke requires imbalance-aware interpretation. Accuracy must not be used alone; recall, F1, ROC-AUC and PR-AUC should be considered during later evaluation. Any resampling or class-weight strategy must be applied only inside the training/CV workflow.

No final model is selected in this phase.
