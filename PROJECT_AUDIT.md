# Project Audit: AI Multi-Disease Risk Screening and Prediction System

**Audit Date**: September 24, 2026  
**Auditor**: Member 1 Machine Learning Engineering Agent  
**Branch**: `member1`  
**Remote**: `https://github.com/nishanthmahalingam28-creator/AI_Multi_Disease_Risk_Screening_and_Prediction_System.git`  
**System Scope**: AI-based Risk Screening, Prediction, and Decision-Support System (**NOT** a medical diagnostic system)  
**Member 1 Assigned Diseases**: Heart Disease, Diabetes, Stroke  

---

## 1. Existing Project Structure

```
AI_Multi_Disease_Risk_Screening_and_Prediction_System/
├── .git/                                # Git version control metadata
├── README.md                            # Minimal repository title placeholder (3 lines, 112 bytes)
└── dataset/                             # Centralized raw dataset repository
    ├── Parkinsons_Disease_Dataset.xlsx  # 35,384 bytes
    ├── breast.csv                       # 125,204 bytes (33 cols)
    ├── chronic kidney disease.csv       # 22,275,331 bytes (17 cols)
    ├── diabetes.csv                     # 23,873 bytes (9 cols, 768 rows) [MEMBER 1]
    ├── heart.csv                        # 11,928 bytes (14 cols, 270 rows) [MEMBER 1]
    ├── indian_liver_patient.csv         # 23,931 bytes (11 cols)
    ├── stroke.csv                       # 332,562 bytes (12 cols, 5,110 rows) [MEMBER 1]
    ├── survey_lung_cancer.csv           # 12,498 bytes (16 cols)
    ├── synthetic_asthma_dataset.csv     # 975,170 bytes (17 cols)
    └── thyroidDF.csv                    # 758,265 bytes (31 cols)
```

No additional source directories, virtual environments, or application packages existed initially in the repository.

---

## 2. Existing Technologies

| Component | Status | Details |
|---|---|---|
| **Operating System** | Active | Windows (PowerShell shell environment) |
| **Python Runtime** | Installed | Python 3.14.6 (64-bit) at `C:\Users\Naveenraj K\AppData\Local\Programs\Python\Python314\python.exe` |
| **Data Science Libraries** | Installed | `pandas 3.0.3`, `numpy 2.5.1`, `matplotlib 3.11.0`, `seaborn 0.13.2`, `jupyterlab 4.6.1`, `ipykernel 7.3.0` |
| **Machine Learning Libraries** | Installed (Phase 1) | `scikit-learn 1.9.1`, `joblib 1.6.0`, `scipy 1.18.1`, `xgboost 3.4.1`, `pytest 9.1.1` (all verified compatible with Python 3.14.6) |
| **Backend Framework** | Missing | No FastAPI, Flask, or Django runtime or dependencies installed. |
| **Frontend Framework** | Missing | No web frameworks, build tools, or HTML/JS/CSS assets present. |
| **Environment Specification** | Established | `requirements.txt` created for Member 1 ML dependencies. |
| **Git Configuration** | Configured | Branch: `member1`. User: `Naveenraj K` (`imnavrio@gmail.com`). Project `.gitignore` created. |

---

## 3. Existing Datasets

The repository contains 10 dataset files located in `dataset/`. The three datasets assigned to Member 1 are all present:

### 3.1 Heart Disease Dataset (`dataset/heart.csv`)
- **Origin**: Statlog (Heart) Dataset.
- **Dimensions**: 270 records, 14 columns.
- **File Size**: 11,928 bytes.
- **Columns**: `Age`, `Sex`, `Chest pain type`, `BP`, `Cholesterol`, `FBS over 120`, `EKG results`, `Max HR`, `Exercise angina`, `ST depression`, `Slope of ST`, `Number of vessels fluro`, `Thallium`, `Heart Disease`.
- **Target Column**: `Heart Disease`.
- **Target Distribution**:
  - `Absence`: 150 instances (55.56%)
  - `Presence`: 120 instances (44.44%)
- **Missing Values**: 0 explicit `NaN` values across all columns.
- **Data Types**: All 13 features are numerical (integers and floats); the target is string (`Absence`/`Presence`).

### 3.2 Diabetes Dataset (`dataset/diabetes.csv`)
- **Origin**: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases).
- **Dimensions**: 768 records, 9 columns.
- **File Size**: 23,873 bytes.
- **Columns**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, `Outcome`.
- **Target Column**: `Outcome`.
- **Target Distribution**:
  - `0` (Negative / No Diabetes): 500 instances (65.10%)
  - `1` (Positive / Diabetes): 268 instances (34.90%)
- **Missing Values**: 0 explicit `NaN` values.
  - *Critical Observation*: Zero values in biological attributes (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) represent physiologically impossible values and indicate missing data requiring disease-specific cleaning.
- **Data Types**: All numerical (integers and floating-point values).

### 3.3 Stroke Dataset (`dataset/stroke.csv`)
- **Origin**: Stroke Prediction Dataset (Kaggle / Healthcare dataset).
- **Dimensions**: 5,110 records, 12 columns.
- **File Size**: 332,562 bytes.
- **Columns**: `id`, `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke`.
- **Target Column**: `stroke`.
- **Target Distribution**:
  - `0` (No Stroke): 4,861 instances (95.13%)
  - `1` (Stroke): 249 instances (4.87%)
  - *Critical Observation*: Extreme class imbalance (~19.5 : 1 ratio). Requires stratified evaluation, balanced metric focus (Recall, PR-AUC, F1-score), and potential sampling or class weighting strategies.
- **Missing Values**:
  - `bmi`: 201 records are missing (`NaN`).
- **Identifier Column**: `id` is a unique patient identifier and must be removed prior to model development to avoid data leakage.
- **Data Types**: Mixed numerical (`age`, `avg_glucose_level`, `bmi`, binary flags) and categorical strings (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`).

### 3.4 Non-Member 1 Datasets (Stored in `dataset/` for Other Team Members)
- `Parkinsons_Disease_Dataset.xlsx`
- `breast.csv`
- `chronic kidney disease.csv`
- `indian_liver_patient.csv`
- `survey_lung_cancer.csv`
- `synthetic_asthma_dataset.csv`
- `thyroidDF.csv`

---

## 4. Existing Machine Learning Code

- **Status**: Scaffolding created in Phase 1 (`ml/member1/{heart,diabetes,stroke}/src/`).
- Training and data processing pipelines will be implemented in subsequent phases.

---

## 5. Existing Machine Learning Models

- **Status**: No trained model files exist yet. Model training will occur in Phases 10–14.

---

## 6. Existing APIs / Backend

- **Status**: No web application framework (FastAPI, Flask) exists yet. Backend handover contract will be prepared in Phase 20–21.

---

## 7. Existing Frontend

- **Status**: No frontend UI exists yet.

---

## 8. Missing Member 1 Components & Roadmap

1. **Phase 1**: ML Structure & Environment Setup (**Completed**).
2. **Phase 2**: Dataset Discovery & Formal Analysis (`dataset_analysis.md`).
3. **Phase 3**: Feature Documentation (`feature_documentation.md`).
4. **Phase 4**: Disease-Specific Data Cleaning (`cleaning.py`, `cleaning_report.md`).
5. **Phase 5**: Exploratory Data Analysis & Visualization (`reports/eda/`).
6. **Phase 6**: Target Preparation & Class Mapping (`class_mapping.json`).
7. **Phase 7**: Feature Analysis & Selection.
8. **Phase 8**: Scikit-Learn Preprocessing Pipelines (leak-free).
9. **Phase 9**: Stratified Train/Test Splits.
10. **Phase 10–14**: Multi-Model Training, Cross-Validation, Tuning, Evaluation, Final Model Selection.
11. **Phase 15–16**: Model & Preprocessor Serialization (`.joblib`), Model Metadata (`model_metadata.json`).
12. **Phase 17–18**: User & Clinical JSON Schemas.
13. **Phase 19–21**: Standalone Inference Engines (`predict.py`), Standardized Response Structure.
14. **Phase 22**: Automated Test Suite.
15. **Phase 23**: Disease README Documentation.
16. **Phase 24**: End-to-End System Validation.

---

**AUDIT CONCLUSION**: Phase 0 audit confirmed. All three datasets present in `dataset/`. Ready for Phase 2.
