# Stroke Dataset: Statistical Profiling and Discovery Report

**Document ID**: `ml/member1/stroke/reports/dataset_analysis.md`  
**Phase**: Phase 2 — Dataset Discovery & In-Depth Statistical Analysis  
**Dataset Source**: `dataset/stroke.csv` (Stroke Prediction Dataset)  
**Analysis Date**: September 24, 2026  
**Scope**: Risk screening and clinical decision support profiling (**NOT** a medical diagnostic assessment)  

---

## A. Basic Information

| Attribute | Value |
|---|---|
| **File Path** | `dataset/stroke.csv` |
| **Number of Records (Rows)** | 5,110 |
| **Number of Features (Columns)** | 12 (11 candidate features + 1 target) |
| **Total Memory Usage** | 1,697,453 bytes (~1.62 MB) |
| **Data Types** | 4 integers (`int64`), 3 floating-points (`float64`), 5 strings/objects (`object`) |

### Column List & Native Data Types
1. `id`: `int64` (Unique patient identifier)
2. `gender`: `object` / `string` (`'Female'`, `'Male'`, `'Other'`)
3. `age`: `float64` (Patient age in years; includes fractional ages for infants)
4. `hypertension`: `int64` (`0` = No history of chronic hypertension, `1` = Diagnosed hypertension)
5. `heart_disease`: `int64` (`0` = No history of heart disease, `1` = Diagnosed heart disease)
6. `ever_married`: `object` / `string` (`'No'`, `'Yes'`)
7. `work_type`: `object` / `string` (`'children'`, `'Govt_job'`, `'Never_worked'`, `'Private'`, `'Self-employed'`)
8. `Residence_type`: `object` / `string` (`'Rural'`, `'Urban'`)
9. `avg_glucose_level`: `float64` (Average blood glucose level in mg/dL)
10. `bmi`: `float64` (Body mass index in $\text{kg}/\text{m}^2$; contains missing values)
11. `smoking_status`: `object` / `string` (`'formerly smoked'`, `'never smoked'`, `'smokes'`, `'Unknown'`)
12. `stroke`: `int64` (Target outcome: `1` if patient experienced a stroke, `0` otherwise)

---

## B. Target Column Analysis & Severe Class Imbalance

The target column is **`stroke`**, indicating whether a patient had a cerebrovascular incident.

| Target Class | Record Count | Percentage (%) | Screening Meaning |
|---|---:|---:|---|
| **`0` (No Stroke)** | 4,861 | 95.13% | Majority class (Negative screening) |
| **`1` (Stroke)** | 249 | 4.87% | Minority class (Positive event) |
| **Total** | **5,110** | **100.00%** | Binary classification task |

### Critical Target Imbalance Metrics
- **Prevalence of Stroke**: **$4.87\%$**
- **Majority-to-Minority Ratio**: **$19.52 : 1$**
- **Imbalance Severity**: **Extreme**. A trivial baseline model predicting 0 for all patients would achieve **$95.13\%$ accuracy** while achieving **$0\%$ recall** (failing to identify all 249 stroke patients).
- **Modeling Implications**: Accuracy is an uninformative metric for this problem. Future evaluation must rely primarily on **Recall (Sensitivity)**, **PR-AUC (Precision-Recall Area Under Curve)**, **ROC-AUC**, and **F1-Score**. Model training will require class weighting (`scale_pos_weight`, `class_weight='balanced'`).
- **Phase 2 Status**: In accordance with instructions, **no resampling (SMOTE, undersampling, oversampling) has been applied yet**.

---

## C. Missing Values Analysis & BMI Missingness Investigation

| Column Name | Missing Count (`NaN`) | Missing Percentage (%) | Phase 2 Action |
|---|---:|---:|---|
| `id` | 0 | 0.00% | None |
| `gender` | 0 | 0.00% | None |
| `age` | 0 | 0.00% | None |
| `hypertension` | 0 | 0.00% | None |
| `heart_disease` | 0 | 0.00% | None |
| `ever_married` | 0 | 0.00% | None |
| `work_type` | 0 | 0.00% | None |
| `Residence_type` | 0 | 0.00% | None |
| `avg_glucose_level` | 0 | 0.00% | None |
| **`bmi`** | **201** | **3.93%** | **Flagged for pipeline imputation (DO NOT drop rows)** |
| `smoking_status` | 0 | 0.00% | Note: Contains 1,544 `'Unknown'` entries (audited below) |
| `stroke` | 0 | 0.00% | None |

### Special Investigation: Non-Random Missingness of BMI
An in-depth cross-tabulation of BMI missingness against the target `stroke` reveals a critical clinical finding:
- Patients with `stroke = 1`: **40 out of 249 records are missing BMI ($16.06\%$)**
- Patients with `stroke = 0`: **161 out of 4,861 records are missing BMI ($3.31\%$)**

> [!WARNING]
> **Data Loss Danger**: Missingness in `bmi` is strongly correlated with the stroke positive class ($16.06\%$ vs $3.31\%$). If rows with missing BMI were simply deleted:
> - $40$ out of $249$ confirmed stroke cases would be permanently lost ($16.06\%$ of all positive cases).
> - This would worsen the class imbalance and introduce acute selection bias.
> - **Mandate for Phase 8**: BMI **must be imputed** using a median or regression imputer fitted strictly inside the training fold, never row-deleted.

---

## D. Duplicate Records Analysis

- **Duplicates including `id`**: `0` rows ($0.00\%$)
- **Duplicates excluding `id`**: `0` rows ($0.00\%$)
- All 5,110 records represent unique clinical profiles. No rows were removed.

---

## E. Unique Values for Categorical Features

| Feature | Distinct Count | Values & Frequency Distribution | Observations |
|---|---:|---|---|
| **`gender`** | 3 | `'Female'`: 2,994 (58.59%)<br>`'Male'`: 2,115 (41.39%)<br>`'Other'`: 1 (0.02%) | The single `'Other'` entry represents a rare category that requires careful handling during cross-validation. |
| **`hypertension`** | 2 | `0`: 4,612 (90.25%)<br>`1`: 498 (9.75%) | Clinically relevant risk factor present in 9.75% of patients. |
| **`heart_disease`** | 2 | `0`: 4,834 (94.60%)<br>`1`: 276 (5.40%) | Comorbid cardiovascular condition present in 5.40% of patients. |
| **`ever_married`** | 2 | `'Yes'`: 3,353 (65.62%)<br>`'No'`: 1,757 (34.38%) | Binary demographic indicator. |
| **`work_type`** | 5 | `'Private'`: 2,925 (57.24%)<br>`'Self-employed'`: 819 (16.03%)<br>`'children'`: 687 (13.44%)<br>`'Govt_job'`: 657 (12.86%)<br>`'Never_worked'`: 22 (0.43%) | Reflects socioeconomic and lifestyle status. Pediatric group (`'children'`) corresponds with low ages. |
| **`Residence_type`** | 2 | `'Urban'`: 2,596 (50.80%)<br>`'Rural'`: 2,514 (49.20%) | Balanced demographic split. |
| **`smoking_status`** | 4 | `'never smoked'`: 1,892 (37.03%)<br>`'Unknown'`: 1,544 (30.22%)<br>`'formerly smoked'`: 885 (17.32%)<br>`'smokes'`: 789 (15.44%) | `'Unknown'` indicates uncollected smoking history (especially common in children and acute admissions). Must be treated as a distinct informative category. |

---

## F. Numerical Statistics

Detailed distribution statistics computed directly on `dataset/stroke.csv`:

| Feature | Count | Mean | Std Dev | Min | 25% (Q1) | Median (Q2) | 75% (Q3) | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`id`** | 5110 | 36517.83 | 21161.72 | 67.00 | 17741.25 | 36932.00 | 54682.00 | 72940.00 |
| **`age`** | 5110 | 43.23 | 22.61 | 0.08 | 25.00 | 45.00 | 61.00 | 82.00 |
| **`avg_glucose_level`** | 5110 | 106.15 | 45.28 | 55.12 | 77.25 | 91.88 | 114.09 | 271.74 |
| **`bmi`** | 4909 | 28.89 | 7.85 | 10.30 | 23.50 | 28.10 | 33.10 | 97.60 |

---

## G. Special Investigation: The `id` Column

In accordance with Phase 2 instructions, a dedicated audit of `id` was conducted:

| Metric | Result | Interpretation |
|---|---|---|
| **Total Rows** | 5,110 | Dataset length |
| **Unique Values in `id`** | 5,110 | Exact 1-to-1 cardinality with rows |
| **Is Unique?** | **`True`** | Every patient has a distinct integer ID |
| **Minimum ID** | 67 | Arbitrary hospital/database index |
| **Maximum ID** | 72,940 | Arbitrary hospital/database index |
| **Behaves as Identifier?** | **`YES`** | Pure non-clinical primary key |

### Why `id` Must Be Excluded in Subsequent Phases
1. **Zero Predictive Value**: The patient's database ID has no physiological or biological relationship with stroke pathology.
2. **Extreme Data Leakage Risk**: Complex non-linear models (e.g., Random Forests, XGBoost) could split on `id` thresholds, memorizing training instances and yielding artificially inflated training metrics that completely collapse on new clinical data.
3. **Phase 2 Status**: Documented for exclusion. In accordance with Phase 2 guidelines, `id` **has not been dropped yet**.

---

## H. Dataset Limitations & Modeling Implications

1. **Extreme Class Imbalance ($4.87\%$ Positive)**:
   - Stroke is a high-stakes, low-frequency event.
   - Models must be optimized using balanced scoring: **PR-AUC**, **Recall**, and **F1-score**.
   - Class weighting (`balanced` weights) will be required.
2. **Missing BMI ($3.93\%$, $16.06\%$ in Stroke Cases)**:
   - Imputation must occur strictly within training folds to avoid data leakage.
   - Deletion of incomplete rows must be avoided to preserve positive cases.
3. **Information Gap in `smoking_status`**:
   - $30.22\%$ of records have `'Unknown'`. Rather than treating this as missing data, keeping `'Unknown'` as an explicit categorical level provides valuable signal.
4. **Pediatric Inclusion**:
   - Ages range from $0.08$ (infant) to $82.0$. The screening model must appropriately handle age-dependent risks (stroke risk in infants is driven by congenital/perinatal factors rather than adult vascular disease).
5. **Singleton Category in `gender` (`'Other'` = 1)**:
   - The single instance of `'Other'` will be handled during one-hot encoding with `handle_unknown='ignore'`.

---

**Report Certification**: All statistics reported above were calculated programmatically from the authentic raw dataset `dataset/stroke.csv`. The `id` column was audited but **not** removed; missing BMI values were analyzed but **not** imputed.
