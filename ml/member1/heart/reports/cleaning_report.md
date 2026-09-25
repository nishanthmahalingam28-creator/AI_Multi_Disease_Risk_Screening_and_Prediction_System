# Heart Disease Data Cleaning & Standardization Report

**Document ID**: `ml/member1/heart/reports/cleaning_report.md`  
**Phase**: Phase 3 — Feature Documentation & Disease-Specific Cleaning  
**Working Dataset**: `ml/member1/heart/data/cleaned.csv`  
**Source Dataset**: `dataset/heart.csv` (Unaltered, Read-Only)  
**Execution Script**: [`ml/member1/heart/src/cleaning.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/heart/src/cleaning.py)  
**Scope**: Risk screening and clinical decision support data hygiene (**NOT** a medical diagnostic assessment)  

---

## 1. Overview of Cleaning Operations

| Operation | Action Taken | Rationale & Clinical Impact |
|---|---|---|
| **Whitespace Normalization** | Stripped leading/trailing whitespace from column headers | Ensures clean programmatic indexing across operating systems. |
| **Duplicate Row Audit** | Verified duplicates = 0. No rows dropped. | Preserves all 270 unique clinical records. |
| **Missing Value Audit** | Verified nulls = 0. No imputation applied. | Raw dataset contains complete records across all 14 columns. |
| **Target Column Encoding** | Mapped `'Absence'` $\rightarrow 0$, `'Presence'` $\rightarrow 1$ | Converts categorical string outcomes into standard binary target format for scikit-learn classifiers. |
| **Feature Type Standardization** | Enforced explicit integer and float types | 12 features cast to `int64`, `ST depression` cast to `float64`. |
| **Identifier Removal** | No columns removed | No administrative or index columns exist in this dataset. |

---

## 2. Feature Classification & Granular Inventory

### 2.1 Target Column
- **Column Name**: `Heart Disease`
- **Original Representation**: String objects (`'Absence'`, `'Presence'`)
- **Cleaned Representation**: Binary integer (`0`, `1`)
- **Class Distribution in Cleaned Dataset**:
  - Class `0` (`Absence`): **150** ($55.56\%$)
  - Class `1` (`Presence`): **120** ($44.44\%$)
- **Total Rows**: **270**

### 2.2 Continuous Numerical Features (5)
These features represent continuous biological measurements suitable for standard scaling or robust scaling during pipeline preprocessing:
- `Age` (Years, range: 29 – 77)
- `BP` (Resting systolic/diastolic blood pressure in mm Hg, range: 94 – 200)
- `Cholesterol` (Serum cholesterol in mg/dl, range: 126 – 564)
- `Max HR` (Maximum exercise heart rate, range: 71 – 202)
- `ST depression` (Exercise-induced ST depression in mm, range: 0.0 – 6.2)

### 2.3 Discrete & Categorical Clinical Features (8)
These features represent integer-coded discrete categories, ordinal scales, or diagnostic test findings. They are preserved in their native integer format and will receive explicit `OneHotEncoder` or passthrough treatment in Phase 8:
- `Sex` (Binary: 1 = Male, 0 = Female)
- `Chest pain type` (Nominal categorical: 1, 2, 3, 4)
- `FBS over 120` (Binary: 0 = False, 1 = True)
- `EKG results` (Nominal categorical: 0 = normal, 1 = ST-T wave abnormality, 2 = probable LVH)
- `Exercise angina` (Binary: 0 = No, 1 = Yes)
- `Slope of ST` (Ordinal categorical: 1 = upsloping, 2 = flat, 3 = downsloping)
- `Number of vessels fluro` (Discrete clinical count: 0, 1, 2, 3)
- `Thallium` (Nominal categorical test code: 3 = normal, 6 = fixed defect, 7 = reversible defect)

---

## 3. Duplicate and Missingness Verification

- **Duplicate Rows**: `0` found in raw data; `0` removed.
- **Missing Values**: `0` found in raw data; `0` introduced.
- **Dimensionality**:
  - Raw shape: `(270, 14)`
  - Cleaned shape: `(270, 14)`
  - Preservation rate: **$100.00\%$**

---

## 4. Output Working Dataset Specification

The cleaned working dataset has been generated deterministically and saved to:
`ml/member1/heart/data/cleaned.csv`

```
Shape: (270 rows, 14 columns)
Columns: Age, Sex, Chest pain type, BP, Cholesterol, FBS over 120, EKG results, 
         Max HR, Exercise angina, ST depression, Slope of ST, Number of vessels fluro, 
         Thallium, Heart Disease
```

*The raw dataset `dataset/heart.csv` remains strictly read-only and unaltered.*
