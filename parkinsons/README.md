# Parkinson's Disease — Member 3A

Run `python -m parkinsons.train` from repository root.

The name field is treated as an identifier and is not a predictive feature. A subject ID is extracted only for grouping. StratifiedGroupKFold is used for comparison and tuning so recordings from the same subject do not cross validation groups. The pipeline compares Logistic Regression, SVM, Random Forest and Gradient Boosting, tunes the strongest candidates and evaluates on a group-held-out test fold.

The model requires the specialized voice measurements supported by the dataset; do not invent simplified medical fields.
