# Diabetes Data Cleaning & Biological Zero-Value Report

**Document ID**: `ml/member1/diabetes/reports/cleaning_report.md`  
**Phase**: Phase 3 — Feature Documentation & Disease-Specific Cleaning  
**Working Dataset**: `ml/member1/diabetes/data/cleaned.csv`  
**Source Dataset**: `dataset/diabetes.csv` (Unaltered, Read-Only)  
**Execution Script**: [`ml/member1/diabetes/src/cleaning.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/diabetes/src/cleaning.py)  
**Scope**: Risk screening and clinical decision support data hygiene (**NOT** a medical diagnostic assessment)  

---

## 1. Overview of Cleaning Operations

| Operation | Action Taken | Rationale & Clinical Impact |
|---|---|---|
| **Whitespace Normalization** | Stripped leading/trailing whitespace from column headers | Ensures clean column access. |
| **Duplicate Row Audit** | Verified duplicates = 0. No rows dropped. | Preserves all 768 unique patient records. |
| **Biological Zero Conversion** | Replaced physiologically implausible zeros with `np.nan` across 5 features | 0 values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI represent unrecorded lab tests rather than true zero values. |
| **Valid Zero Preservation** | Explicitly preserved `Pregnancies = 0` (111 records) | Zero pregnancies is completely plausible for nulliparous individuals. Converting to NaN would introduce catastrophic bias. |
| **Imputation Postponement** | Left `NaN` values untouched in working dataset | Imputation will be executed inside scikit-learn training pipelines during Phase 8 to strictly prevent data leakage. |
| **Target Validation** | Validated `Outcome` contains only binary integers (0, 1) | Target classes verified: 500 zeros, 268 ones. |

---

## 2. Detailed Biological Zero-Value Conversion Audit

A total of **652 physiologically impossible zero values** across 5 clinical measurements were converted to explicit `NaN`:

| Feature Name | Initial Zero Count | Total Rows | Zero Percentage (%) | Physiological Plausibility as Zero | Cleaning Action in Working Dataset |
|---|---:|---:|---:|:---:|---|
| **`Glucose`** | 5 | 768 | 0.65% | **Impossible** | Converted 5 zeros to `NaN`. Zero plasma glucose in OGTT indicates omitted test. |
| **`BloodPressure`** | 35 | 768 | 4.56% | **Impossible** | Converted 35 zeros to `NaN`. Zero resting diastolic blood pressure is non-viable. |
| **`SkinThickness`** | 227 | 768 | 29.56% | **Impossible** | Converted 227 zeros to `NaN`. Triceps skinfold caliper cannot be 0 mm. |
| **`Insulin`** | 374 | 768 | 48.70% | **Impossible** | Converted 374 zeros to `NaN`. 2-hour serum insulin cannot be absolute 0 $\mu$U/ml. |
| **`BMI`** | 11 | 768 | 1.43% | **Impossible** | Converted 11 zeros to `NaN`. Body Mass Index cannot be 0. |
| **`Pregnancies`** | 111 | 768 | 14.45% | **VALID** | **PRESERVED AS 0**. Indicates female who has never been pregnant. |
| **`DiabetesPedigreeFunction`** | 0 | 768 | 0.00% | Valid | Min is 0.078. No zero entries exist. |
| **`Age`** | 0 | 768 | 0.00% | Valid | Min is 21. No zero entries exist. |
| **`Outcome`** | 500 | 768 | 65.10% | Valid | Negative class indicator (Non-diabetic). |

---

## 3. Preservation of Dataset Integrity

- **Row Count**: Raw: `768` $\rightarrow$ Cleaned: `768` (0 rows deleted).
- **Target Distribution**:
  - Class `0` (Non-diabetic): **500** ($65.10\%$)
  - Class `1` (Diabetic): **268** ($34.90\%$)
- **Missing Value Count Before Cleaning**: `0` explicit `NaN`s.
- **Missing Value Count After Cleaning**:
  - `Glucose`: 5 `NaN`s
  - `BloodPressure`: 35 `NaN`s
  - `SkinThickness`: 227 `NaN`s
  - `Insulin`: 374 `NaN`s
  - `BMI`: 11 `NaN`s
  - Total `NaN`s introduced: **652**

---

## 4. Output Working Dataset Specification

The cleaned working dataset has been saved to:
`ml/member1/diabetes/data/cleaned.csv`

```
Shape: (768 rows, 9 columns)
Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
         BMI, DiabetesPedigreeFunction, Age, Outcome
```

*The raw dataset `dataset/diabetes.csv` remains strictly read-only and unaltered.*
