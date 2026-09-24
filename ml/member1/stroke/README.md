# Stroke Risk Screening Module

## Disease Name
Cerebrovascular Accident (Stroke Risk Screening)

## Member 1 ML Responsibility
Member 1 is responsible for developing the complete machine learning screening pipeline for Stroke risk evaluation. Key challenges include handling extreme class imbalance (~4.87% positive prevalence), imputation of missing BMI values, elimination of patient identifier leakage, mixed categorical/numerical feature encoding, precision/recall-oriented evaluation, model serialization, and standalone prediction modules.

## Dataset Location
- **Path**: `dataset/stroke.csv` (relative to repository root)
- **Source**: Stroke Prediction Dataset (Kaggle / Healthcare)
- **Records**: 5,110 instances
- **Features**: 11 clinical/demographic attributes (including patient ID) + 1 target (`stroke`)

## Current Project Status
- **Phase 0 (Project Inspection)**: Completed.
- **Phase 1 (ML Setup & Structure)**: Completed.
- **Phase 2 (Dataset Discovery & Analysis)**: Pending.
- **Current State**: Scaffolding established. Raw dataset inspected. No models trained yet.

## Planned ML Workflow
1. Dataset Discovery & Statistical Profiling (`reports/dataset_analysis.md`)
2. Feature Documentation (`reports/feature_documentation.md`)
3. Data Cleaning Pipeline (Identifier dropping, BMI missingness analysis, categorical cleaning)
4. Exploratory Data Analysis & Imbalance Study (`reports/eda/`)
5. Target Preparation & Class Mapping (`class_mapping.json`)
6. Feature Analysis & Leakage Prevention
7. Leak-Free Preprocessing Pipeline (Imputation, One-Hot Encoding, Scaling)
8. Stratified Train/Test Split (80/20, fixed random state)
9. Candidate Model Training with Imbalance-Aware Strategies (Class weighting, balanced algorithms)
10. Stratified Cross-Validation (`reports/cross_validation_results.csv`)
11. Hyperparameter Tuning (`GridSearchCV`)
12. Final Evaluation on Holdout Test Set with Imbalanced Metrics (PR-AUC, Recall, F1)
13. Model Selection & Decision-Support Rationale
14. Model Serialization (`models/stroke_model.joblib`, `models/stroke_preprocessor.joblib`)
15. Metadata & JSON Schemas (`user_schema.json`, `clinical_schema.json`, `model_metadata.json`)
16. Standalone Prediction Engines (`src/predict.py`)
17. Unit & Integration Testing
18. Documentation & Handover

> [!NOTE]
> This module is designed for risk screening and decision support only, not for medical diagnosis.
