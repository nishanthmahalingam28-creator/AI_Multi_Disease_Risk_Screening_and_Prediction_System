# Heart Disease Risk Screening Module

## Disease Name
Heart Disease (Coronary Artery / Cardiovascular Risk Screening)

## Member 1 ML Responsibility
Member 1 is responsible for designing, training, evaluating, and packaging the machine learning risk screening pipeline for Heart Disease. This includes data cleaning, exploratory analysis, reproducible preprocessing pipelines, multi-model evaluation, serialization, schema generation, and inference endpoints for team integration.

## Dataset Location
- **Path**: `dataset/heart.csv` (relative to repository root)
- **Source**: Statlog (Heart) Dataset
- **Records**: 270 instances
- **Features**: 13 predictive attributes + 1 target (`Heart Disease`)

## Current Project Status
- **Phase 0 (Project Inspection)**: Completed.
- **Phase 1 (ML Setup & Structure)**: Completed.
- **Phase 2 (Dataset Discovery & Analysis)**: Pending.
- **Current State**: Initial directory layout and modular source scaffolding established. No models trained yet.

## Planned ML Workflow
1. Dataset Discovery & Statistical Profiling (`reports/dataset_analysis.md`)
2. Feature Documentation (`reports/feature_documentation.md`)
3. Data Cleaning Pipeline (`src/cleaning.py`, `reports/cleaning_report.md`)
4. Exploratory Data Analysis & Visualizations (`reports/eda/`)
5. Target Encoding & Class Mapping (`class_mapping.json`)
6. Feature Analysis & Selection
7. Leak-Free Scikit-Learn Preprocessing Pipeline
8. Stratified Train/Test Split (80/20, fixed random state)
9. Candidate Model Training (Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost)
10. Stratified Cross-Validation (`reports/cross_validation_results.csv`)
11. Hyperparameter Tuning (`GridSearchCV`)
12. Final Evaluation on Holdout Test Set (`reports/model_comparison.csv`)
13. Model Selection & Decision-Support Rationale
14. Model Serialization (`models/heart_model.joblib`, `models/heart_preprocessor.joblib`)
15. Metadata & JSON Schemas (`user_schema.json`, `clinical_schema.json`, `model_metadata.json`)
16. Standalone Prediction Engines (`src/predict.py`)
17. Unit & Integration Testing
18. Documentation & Handover

> [!NOTE]
> This module is designed for risk screening and decision support only, not for medical diagnosis.
