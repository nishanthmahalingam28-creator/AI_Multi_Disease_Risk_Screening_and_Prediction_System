# Stroke Data Cleaning & Leakage Prevention Report

**Document ID**: `ml/member1/stroke/reports/cleaning_report.md`  
**Phase**: Phase 3 — Feature Documentation & Disease-Specific Cleaning  
**Working Dataset**: `ml/member1/stroke/data/cleaned.csv`  
**Source Dataset**: `dataset/stroke.csv` (Unaltered, Read-Only)  
**Execution Script**: [`ml/member1/stroke/src/cleaning.py`](file:///c:/Users/Naveenraj%20K/OneDrive/Documents/AI_Multi_Disease_Risk_Screening_and_Prediction_System/ml/member1/stroke/src/cleaning.py)  
**Scope**: Risk screening and clinical decision support data hygiene (**NOT** a medical diagnostic assessment)  

---

## 1. Overview of Cleaning Operations

| Operation | Action Taken | Rationale & Clinical Impact |
|---|---|---|
| **Identifier Exclusion** | Dropped administrative column `id` from the cleaned ML working dataset | `id` is a 100% unique primary key. Retaining it would allow decision trees and ensemble algorithms to split on arbitrary patient IDs, causing extreme data leakage and false generalization. |
| **Preservation of Missing BMI** | Retained all 201 missing BMI values as explicit `NaN` | Deleting rows with missing BMI would permanently discard 40 out of 249 confirmed stroke cases (16.06% of the positive class), severely distorting clinical screening utility. |
| **Informative Category Retention** | Preserved `'Unknown'` in `smoking_status` (1,544 rows, 30.22%) | Represents uncollected/unavailable smoking history rather than non-smoking. Kept as an explicit categorical level. |
| **Rare Category Retention** | Preserved `'Other'` in `gender` (1 row) | Preserved to allow the one-hot encoding pipeline to handle rare/unseen gender inputs safely without artificial deletion. |
| **Categorical Whitespace Stripping** | Trimmed whitespace across string attributes (`gender`, `work_type`, `smoking_status`, etc.) | Eliminates accidental trailing space inconsistencies. |
| **Duplicate Row Audit** | Verified duplicates = 0. No rows dropped. | All 5,110 patient rows represent distinct clinical records. |

---

## 2. Feature-by-Feature Cleaning & Exclusion Inventory

| Feature Name | Initial Type | Cleaned Type | Role in Model | Cleaned Action Taken | Rationale |
|---|---|---|---|---|---|
| **`id`** | `int64` | *Excluded* | **EXCLUDED** | **Removed from ML dataset** | Administrative key ($100\%$ unique). No biological predictive value; high data leakage hazard. |
| **`gender`** | `object` | `object` | Predictor | Whitespace trimmed; `'Other'` retained | Biological sex covariate. |
| **`age`** | `float64` | `float64` | Predictor | Validated numerical range ($0.08 - 82.0$) | Primary non-modifiable stroke risk factor. |
| **`hypertension`** | `int64` | `int64` | Predictor | Validated binary flag ($0, 1$) | Chronic vascular comorbidity. |
| **`heart_disease`** | `int64` | `int64` | Predictor | Validated binary flag ($0, 1$) | Cardioembolic stroke risk factor. |
| **`ever_married`** | `object` | `object` | Predictor | Whitespace trimmed ($'Yes', 'No'$) | Sociodemographic indicator. |
| **`work_type`** | `object` | `object` | Predictor | Whitespace trimmed (5 categories) | Lifestyle and occupational stress proxy. |
| **`Residence_type`** | `object` | `object` | Predictor | Whitespace trimmed ($'Urban', 'Rural'$) | Geographic environment indicator. |
| **`avg_glucose_level`** | `float64` | `float64` | Predictor | Validated numerical range ($55.12 - 271.74$) | Metabolic and diabetes comorbidity biomarker. |
| **`bmi`** | `float64` | `float64` | Predictor | **Preserved 201 NaNs** | Imputation deferred to Phase 8 leak-free pipeline. |
| **`smoking_status`** | `object` | `object` | Predictor | Whitespace trimmed; `'Unknown'` preserved | Tobacco exposure risk factor. |
| **`stroke`** | `int64` | `int64` | **Target** | Validated binary class ($0, 1$) | Target screening outcome ($4.87\%$ positive prevalence). |

---

## 3. Dimensionality & Imbalance Validation

- **Row Count**: Raw: `5,110` $\rightarrow$ Cleaned: `5,110` (0 rows deleted).
- **Column Count**: Raw: `12` $\rightarrow$ Cleaned: `11` (only `id` excluded).
- **Target Distribution**:
  - Class `0` (No Stroke): **4,861** ($95.13\%$)
  - Class `1` (Stroke): **249** ($4.87\%$)
  - Imbalance Ratio: **$19.52 : 1$** (Strictly preserved)
- **Missing Values in Cleaned Dataset**:
  - `bmi`: Exactly **201** missing entries ($3.93\%$).
  - All other 10 columns: **0** missing values.

---

## 4. Output Working Dataset Specification

The cleaned working dataset has been saved to:
`ml/member1/stroke/data/cleaned.csv`

```
Shape: (5110 rows, 11 columns)
Columns: gender, age, hypertension, heart_disease, ever_married, work_type, 
         Residence_type, avg_glucose_level, bmi, smoking_status, stroke
```

*The raw dataset `dataset/stroke.csv` remains strictly read-only and unaltered.*
