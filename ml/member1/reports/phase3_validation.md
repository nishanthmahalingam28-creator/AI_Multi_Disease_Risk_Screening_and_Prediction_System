# Phase 3 Cross-Dataset Validation Report

**Document ID**: `ml/member1/reports/phase3_validation.md`  
**Phase Completed**: Phase 3 — Feature Documentation & Disease-Specific Cleaning  
**Validation Date**: September 24, 2026  
**Status**: **ALL INTEGRITY & CLEANING CHECKS PASSED (100%)**  

---

## 1. Summary of Created and Modified Files

### 1.1 Source Cleaning Modules Created / Implemented
- [`ml/member1/heart/src/cleaning.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/heart/src/cleaning.py): Implements deterministic `clean_heart_data(df)`.
- [`ml/member1/diabetes/src/cleaning.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/diabetes/src/cleaning.py): Implements deterministic `clean_diabetes_data(df)` with biological zero handling.
- [`ml/member1/stroke/src/cleaning.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/stroke/src/cleaning.py): Implements deterministic `clean_stroke_data(df, drop_id=True)` with identifier exclusion.

### 1.2 Working Cleaned Datasets Generated
- [`ml/member1/heart/data/cleaned.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/heart/data/cleaned.csv): 270 rows, 14 columns.
- [`ml/member1/diabetes/data/cleaned.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/diabetes/data/cleaned.csv): 768 rows, 9 columns (652 converted NaNs).
- [`ml/member1/stroke/data/cleaned.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/stroke/data/cleaned.csv): 5,110 rows, 11 columns (`id` excluded, 201 BMI NaNs preserved).

### 1.3 Documentation & Reports Created
- [`ml/member1/heart/reports/feature_dictionary.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/heart/reports/feature_dictionary.csv)
- [`ml/member1/heart/reports/cleaning_report.md`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/heart/reports/cleaning_report.md)
- [`ml/member1/diabetes/reports/feature_dictionary.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/diabetes/reports/feature_dictionary.csv)
- [`ml/member1/diabetes/reports/cleaning_report.md`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/diabetes/reports/cleaning_report.md)
- [`ml/member1/stroke/reports/feature_dictionary.csv`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/stroke/reports/feature_dictionary.csv)
- [`ml/member1/stroke/reports/cleaning_report.md`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/stroke/reports/cleaning_report.md)
- [`ml/member1/reports/phase3_validation.md`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/reports/phase3_validation.md) (This document)

### 1.4 Files Modified
- **None**. No existing files were deleted, moved, or corrupted.

---

## 2. Dimensionality & Target Audit (Before vs After)

| Metric | Heart Disease | Diabetes | Stroke |
|---|:---:|:---:|:---:|
| **Raw Rows** | 270 | 768 | 5,110 |
| **Cleaned Rows** | 270 | 768 | 5,110 |
| **Row Preservation Rate** | **100.00%** | **100.00%** | **100.00%** |
| **Raw Columns** | 14 | 9 | 12 |
| **Cleaned Columns** | 14 | 9 | 11 (`id` dropped) |
| **Target Column** | `Heart Disease` | `Outcome` | `stroke` |
| **Raw Target Classes** | `Absence`: 150<br>`Presence`: 120 | `0`: 500<br>`1`: 268 | `0`: 4,861<br>`1`: 249 |
| **Cleaned Target Classes** | `0`: 150 ($55.56\%$)<br>`1`: 120 ($44.44\%$) | `0`: 500 ($65.10\%$)<br>`1`: 268 ($34.90\%$) | `0`: 4,861 ($95.13\%$)<br>`1`: 249 ($4.87\%$) |
| **Target Distribution Change** | **0.00%** (Exact match) | **0.00%** (Exact match) | **0.00%** (Exact match) |

---

## 3. Specific Cleaning Validations

### 3.1 Diabetes Biological Zero-to-NaN Conversions
- Total zero values converted to `np.nan`: **652**
  - `Glucose`: **5** zeros $\rightarrow$ NaN
  - `BloodPressure`: **35** zeros $\rightarrow$ NaN
  - `SkinThickness`: **227** zeros $\rightarrow$ NaN
  - `Insulin`: **374** zeros $\rightarrow$ NaN
  - `BMI`: **11** zeros $\rightarrow$ NaN
- Valid zeros strictly preserved:
  - `Pregnancies = 0`: **111** entries strictly retained (nulliparous women).
  - `Outcome = 0`: **500** entries strictly retained.
- Imputation status: **0 imputations performed** (deferred to Phase 8 leak-free training pipeline).

### 3.2 Stroke Identifier & Missingness Handling
- **`id` Removal**:
  - `id` present in `dataset/stroke.csv`: **Yes** (5,110 unique IDs).
  - `id` present in `ml/member1/stroke/data/cleaned.csv`: **No** (dropped cleanly).
- **BMI Missingness**:
  - Missing BMI count in raw: **201** ($3.93\%$).
  - Missing BMI count in cleaned: **201** ($3.93\%$).
  - Preserved in stroke patients (`stroke=1`): **40** records ($16.06\%$ of positive cases).
  - Rows deleted due to missing BMI: **0**.
- **Categorical Preservation**:
  - `smoking_status = 'Unknown'`: **1,544** records preserved.
  - `gender = 'Other'`: **1** record preserved.

---

## 4. Raw Dataset Cryptographic Integrity Check

| Raw Dataset Path | Verified File Size | SHA-256 Checksum | Integrity Status |
|---|---:|---|:---:|
| `dataset/heart.csv` | 11,928 bytes | `cba80a969a83288e84f4f981e7fc33fe4ef128a9708959292368e42227c9c2cb` | **100% UNCHANGED** |
| `dataset/diabetes.csv` | 23,873 bytes | `698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72` | **100% UNCHANGED** |
| `dataset/stroke.csv` | 332,562 bytes | `aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815` | **100% UNCHANGED** |

---

## 5. Warnings & Next Steps for Future Phases
1. **Heart Disease**:
   - `EKG results = 1` occurs in only 2 records. During train/test splitting, stratified K-fold cross-validation must be monitored to ensure rare clinical classes are accommodated.
2. **Diabetes**:
   - `Insulin` has 48.70% missingness and `SkinThickness` has 29.56% missingness. In Phase 8, median or multivariate iterative imputation within scikit-learn pipelines will be compared.
3. **Stroke**:
   - Imbalance is extreme (4.87% positive). Phase 10 candidate models must implement class weighting (`class_weight='balanced'`, `scale_pos_weight`) and be evaluated on PR-AUC and Recall.
