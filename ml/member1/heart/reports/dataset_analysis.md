# Heart Disease Dataset: Statistical Profiling and Discovery Report

**Document ID**: `ml/member1/heart/reports/dataset_analysis.md`  
**Phase**: Phase 2 — Dataset Discovery & In-Depth Statistical Analysis  
**Dataset Source**: `dataset/heart.csv` (Statlog Heart Dataset)  
**Analysis Date**: September 24, 2026  
**Scope**: Risk screening and clinical decision support profiling (**NOT** a medical diagnostic assessment)  

---

## A. Basic Information

| Attribute | Value |
|---|---|
| **File Path** | `dataset/heart.csv` |
| **Number of Records (Rows)** | 270 |
| **Number of Features (Columns)** | 14 (13 input features + 1 target) |
| **Total Memory Usage** | 43,452 bytes (~42.43 KB) |
| **Data Types** | 12 integer features (`int64`), 1 float feature (`float64`), 1 string/object target (`object`) |

### Column List & Native Data Types
1. `Age`: `int64` (Patient age in years)
2. `Sex`: `int64` (Biological sex; 1 = Male, 0 = Female)
3. `Chest pain type`: `int64` (Categorical rating from 1 to 4)
4. `BP`: `int64` (Resting blood pressure in mm Hg)
5. `Cholesterol`: `int64` (Serum cholesterol in mg/dl)
6. `FBS over 120`: `int64` (Fasting blood sugar > 120 mg/dl; 1 = True, 0 = False)
7. `EKG results`: `int64` (Resting electrocardiographic results; 0, 1, 2)
8. `Max HR`: `int64` (Maximum heart rate achieved during exercise)
9. `Exercise angina`: `int64` (Exercise-induced angina; 1 = Yes, 0 = No)
10. `ST depression`: `float64` (ST depression induced by exercise relative to rest)
11. `Slope of ST`: `int64` (Slope of the peak exercise ST segment; 1, 2, 3)
12. `Number of vessels fluro`: `int64` (Number of major vessels colored by fluoroscopy; 0 to 3)
13. `Thallium`: `int64` (Thallium heart scan; 3 = normal, 6 = fixed defect, 7 = reversible defect)
14. `Heart Disease`: `object` / `string` (Target outcome: `'Absence'`, `'Presence'`)

---

## B. Target Column Analysis

The target column is **`Heart Disease`**, representing the presence or absence of cardiovascular/heart disease.

| Target Class | Record Count | Percentage | Medical Context in Dataset |
|---|---:|---:|---|
| **`Absence`** | 150 | 55.56% | Negative screening / No diagnosed heart disease |
| **`Presence`** | 120 | 44.44% | Positive screening / Diagnosed heart disease |
| **Total** | **270** | **100.00%** | Binary classification task |

### Distinction Between Target and Features
- **Target**: `Heart Disease` (categorical string outcome to be predicted).
- **Predictors (13)**: All remaining 13 columns constitute physiological, demographic, and clinical diagnostic measurements.
- **Class Balance**: The target distribution is balanced (55.56% to 44.44%), with a majority-to-minority ratio of 1.25 : 1. No extreme class imbalance mitigation (e.g., SMOTE) is mandatory, though stratified splitting is critical due to the compact sample size.

---

## C. Missing Values Analysis

| Column Name | Missing Count (`NaN`) | Missing Percentage (%) | Phase 2 Action |
|---|---:|---:|---|
| `Age` | 0 | 0.00% | None (Complete) |
| `Sex` | 0 | 0.00% | None (Complete) |
| `Chest pain type` | 0 | 0.00% | None (Complete) |
| `BP` | 0 | 0.00% | None (Complete) |
| `Cholesterol` | 0 | 0.00% | None (Complete) |
| `FBS over 120` | 0 | 0.00% | None (Complete) |
| `EKG results` | 0 | 0.00% | None (Complete) |
| `Max HR` | 0 | 0.00% | None (Complete) |
| `Exercise angina` | 0 | 0.00% | None (Complete) |
| `ST depression` | 0 | 0.00% | None (Complete) |
| `Slope of ST` | 0 | 0.00% | None (Complete) |
| `Number of vessels fluro` | 0 | 0.00% | None (Complete) |
| `Thallium` | 0 | 0.00% | None (Complete) |
| `Heart Disease` | 0 | 0.00% | None (Complete) |

**Conclusion on Missingness**: There are **0 explicit null/NaN values** across all 270 rows and 14 columns. No values were modified or imputed during this discovery phase.

---

## D. Duplicate Records Analysis

- **Total duplicate rows detected**: `0`
- **Duplicate percentage**: `0.00%`
- All 270 rows contain distinct feature combinations. No rows were removed.

---

## E. Unique Values for Discrete and Categorical Columns

Several columns in this dataset are numerically encoded representations of discrete or clinical categories:

| Column | Distinct Values Count | Observed Values & Distribution | Clinical Interpretation |
|---|---:|---|---|
| **`Sex`** | 2 | `1`: 183 (67.78%)<br>`0`: 87 (32.22%) | 1 = Male; 0 = Female |
| **`Chest pain type`** | 4 | `4`: 129 (47.78%)<br>`3`: 79 (29.26%)<br>`2`: 42 (15.56%)<br>`1`: 20 (7.41%) | 1 = Typical angina<br>2 = Atypical angina<br>3 = Non-anginal pain<br>4 = Asymptomatic |
| **`FBS over 120`** | 2 | `0`: 230 (85.19%)<br>`1`: 40 (14.81%) | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| **`EKG results`** | 3 | `2`: 137 (50.74%)<br>`0`: 131 (48.52%)<br>`1`: 2 (0.74%) | 0 = Normal<br>1 = Having ST-T wave abnormality<br>2 = Showing probable/definite left ventricular hypertrophy |
| **`Exercise angina`** | 2 | `0`: 181 (67.04%)<br>`1`: 89 (32.96%) | Exercise-induced angina (1 = Yes, 0 = No) |
| **`Slope of ST`** | 3 | `1`: 130 (48.15%)<br>`2`: 122 (45.19%)<br>`3`: 18 (6.67%) | 1 = Upsloping<br>2 = Flat<br>3 = Downsloping |
| **`Number of vessels fluro`** | 4 | `0`: 160 (59.26%)<br>`1`: 58 (21.48%)<br>`2`: 33 (12.22%)<br>`3`: 19 (7.04%) | Number of major coronary vessels (0–3) colored by fluoroscopy |
| **`Thallium`** | 3 | `3`: 152 (56.30%)<br>`7`: 104 (38.52%)<br>`6`: 14 (5.19%) | Thallium scintigraphy stress test:<br>3 = Normal<br>6 = Fixed defect<br>7 = Reversible defect |

---

## F. Numerical Statistics

Detailed distribution statistics for continuous and ordinal features calculated directly on `dataset/heart.csv`:

| Feature | Count | Mean | Std Dev | Min | 25% (Q1) | Median (Q2) | 75% (Q3) | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Age`** | 270 | 54.43 | 9.11 | 29.0 | 48.0 | 55.0 | 61.0 | 77.0 |
| **`BP`** | 270 | 131.34 | 17.86 | 94.0 | 120.0 | 130.0 | 140.0 | 200.0 |
| **`Cholesterol`** | 270 | 249.66 | 51.69 | 126.0 | 213.0 | 245.0 | 280.0 | 564.0 |
| **`Max HR`** | 270 | 149.68 | 23.17 | 71.0 | 133.0 | 153.5 | 166.0 | 202.0 |
| **`ST depression`** | 270 | 1.05 | 1.15 | 0.00 | 0.00 | 0.80 | 1.60 | 6.20 |
| **`Number of vessels fluro`** | 270 | 0.67 | 0.94 | 0.0 | 0.0 | 0.0 | 1.0 | 3.0 |
| **`Slope of ST`** | 270 | 1.59 | 0.61 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 |

---

## G. Potential Identifiers & Administrative Fields

- **Screening Result**: **No identifier columns present**.
- There are no row IDs, patient reference numbers, medical record numbers (MRNs), or hospital administrative tracking codes.
- Every attribute represents a clinical, demographic, or physiological feature.
- **Risk Assessment**: No identifier columns need to be excluded for data leakage reasons in this dataset.

---

## H. Dataset Limitations & Key Findings for Later Phases

1. **Compact Sample Size ($N = 270$)**:
   - The dataset is relatively small. Overfitting is a primary risk, especially for high-capacity models (e.g., deep gradient boosting, non-linear SVMs with high $C$).
   - Strict stratified cross-validation (5-fold) and regularization are essential.
2. **Rare Categorical Subclasses**:
   - `EKG results = 1`: Present in only **2 records** ($0.74\%$). A standard random split could easily isolate these records entirely into either the train or test set.
   - `Thallium = 6`: Present in only **14 records** ($5.19\%$).
   - `Slope of ST = 3`: Present in only **18 records** ($6.67\%$).
3. **Clinical vs User Screening Input Gap**:
   - Several features require specialized hospital diagnostic equipment (`Number of vessels fluro` requires cardiac catheterization/fluoroscopy; `Thallium` requires nuclear scintigraphy; `ST depression` and `Slope of ST` require exercise ECG stress testing).
   - In subsequent phases, schema design must clearly separate the **Clinical Decision Support pathway** (which uses all 13 features) from a potential lightweight **Public User Risk Screening pathway** (which relies on demographic and basic physical measurements like Age, Sex, BP, Cholesterol, and Max HR).
4. **Target Encoding Requirement**:
   - The target values are string tokens (`'Absence'`, `'Presence'`). In Phase 6, these will be mapped cleanly (`'Absence' -> 0`, `'Presence' -> 1`) in `class_mapping.json`.

---

**Report Certification**: All statistics reported above were calculated programmatically from the authentic raw dataset `dataset/heart.csv`. No data was modified, cleaned, or transformed.
