# Diabetes Risk Screening Module

## Disease Name
Diabetes Mellitus (Type 2 Diabetes Risk Screening)

## Member 1 ML Responsibility
Member 1 is responsible for developing the complete machine learning screening pipeline for Diabetes risk assessment. This entails domain-specific handling of biological zero measurements, exploratory data analysis, leak-free numerical transformation, model experimentation, hyperparameter tuning, model packaging, schema definition, and standalone prediction modules.

## Dataset Location
- **Path**: `dataset/diabetes.csv` (relative to repository root)
- **Source**: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases)
- **Records**: 768 instances
- **Features**: 8 diagnostic attributes + 1 target (`Outcome`)

## Current Project Status
- **Phase 0 (Project Inspection)**: Completed.
- **Phase 1 (ML Setup & Structure)**: Completed.
- **Phase 2 (Dataset Discovery & Analysis)**: Pending.
- **Current State**: Scaffolding established. Raw dataset inspected. No models trained yet.

## Planned ML Workflow
1. Dataset Discovery & Statistical Profiling (`reports/dataset_analysis.md`)
2. Feature Documentation (`reports/feature_documentation.md`)
3. Data Cleaning Pipeline with Biological Zero Investigation (`src/cleaning.py`, `reports/cleaning_report.md`)
4. Exploratory Data Analysis & Visualizations (`reports/eda/`)
5. Target Encoding & Class Mapping (`class_mapping.json`)
6. Feature Analysis & Correlation Study
7. Leak-Free Numerical Imputation & Preprocessing Pipeline
8. Stratified Train/Test Split (80/20, fixed random state)
9. Candidate Model Training (Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost)
10. Stratified Cross-Validation (`reports/cross_validation_results.csv`)
11. Hyperparameter Tuning (`GridSearchCV`)
12. Final Evaluation on Holdout Test Set (`reports/model_comparison.csv`)
13. Model Selection & Decision-Support Rationale
14. Model Serialization (`models/diabetes_model.joblib`, `models/diabetes_preprocessor.joblib`)
15. Metadata & JSON Schemas (`user_schema.json`, `clinical_schema.json`, `model_metadata.json`)
16. Standalone Prediction Engines (`src/predict.py`)
17. Unit & Integration Testing
18. Documentation & Handover

> [!NOTE]
> This module is designed for risk screening and decision support only, not for medical diagnosis.
