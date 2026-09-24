# Phase 11 — Hyperparameter Tuning

Tuning uses only the Phase 7 training split with 5-fold stratified cross-validation. Preprocessing remains inside each GridSearchCV pipeline.

Heart Disease and Diabetes use ROC-AUC as the primary tuning score. Stroke uses Average Precision (PR-AUC) because of its severe positive-class imbalance and additionally uses class-weighted Logistic Regression, SVM and Random Forest candidates.

The untouched test set is not used for tuning. Tuning results are saved to each disease's reports directory. Final model selection and test evaluation remain separate steps.
