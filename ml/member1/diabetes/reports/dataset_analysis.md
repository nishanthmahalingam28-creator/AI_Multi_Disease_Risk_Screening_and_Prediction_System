# Diabetes Dataset: Statistical Profiling and Discovery Report

**Document ID**: `ml/member1/diabetes/reports/dataset_analysis.md`  
**Phase**: Phase 2 — Dataset Discovery & In-Depth Statistical Analysis  
**Dataset Source**: `dataset/diabetes.csv` (Pima Indians Diabetes Database)  
**Analysis Date**: September 24, 2026  
**Scope**: Risk screening and clinical decision support profiling (**NOT** a medical diagnostic assessment)  

---

## A. Basic Information

| Attribute | Value |
|---|---|
| **File Path** | `dataset/diabetes.csv` |
| **Number of Records (Rows)** | 768 |
| **Number of Features (Columns)** | 9 (8 input features + 1 target) |
| **Total Memory Usage** | 55,428 bytes (~54.13 KB) |
| **Data Types** | 7 integer features (`int64`), 2 floating-point features (`float64`) |

### Column List & Native Data Types
1. `Pregnancies`: `int64` (Number of times pregnant)
2. `Glucose`: `int64` (Plasma glucose concentration at 2 hours in an oral glucose tolerance test)
3. `BloodPressure`: `int64` (Diastolic blood pressure in mm Hg)
4. `SkinThickness`: `int64` (Triceps skin fold thickness in mm)
5. `Insulin`: `int64` (2-Hour serum insulin in $\mu$U/ml)
6. `BMI`: `float64` (Body mass index: $\text{weight in kg} / (\text{height in m})^2$)
7. `DiabetesPedigreeFunction`: `float64` (Diabetes pedigree function: genetic risk score based on family history)
8. `Age`: `int64` (Age in years, all subjects $\ge 21$)
9. `Outcome`: `int64` (Target class: `0` = No Diabetes, `1` = Diabetes)

---

## B. Target Column Analysis

The target column is **`Outcome`**, indicating whether the patient tested positive for diabetes within 5 years of the clinical examination.

| Target Class | Record Count | Percentage | Medical / Screening Interpretation |
|---|---:|---:|---|
| **`0`** | 500 | 65.10% | Negative screening / Non-diabetic |
| **`1`** | 268 | 34.90% | Positive screening / Diabetic |
| **Total** | **768** | **100.00%** | Binary classification task |

### Distinction Between Target and Features
- **Target**: `Outcome` (binary integer: `0` or `1`).
- **Features (8)**: Continuous clinical, biometric, and demographic risk indicators.
- **Class Balance**: Moderately balanced (65.10% vs 34.90%, majority-to-minority ratio of 1.87 : 1). Standard stratified sampling will preserve this ratio during train/test partitioning.

---

## C. Missing Values Analysis

| Column Name | Explicit Missing Count (`NaN`) | Missing Percentage (%) |
|---|---:|---:|
| `Pregnancies` | 0 | 0.00% |
| `Glucose` | 0 | 0.00% |
| `BloodPressure` | 0 | 0.00% |
| `SkinThickness` | 0 | 0.00% |
| `Insulin` | 0 | 0.00% |
| `BMI` | 0 | 0.00% |
| `DiabetesPedigreeFunction` | 0 | 0.00% |
| `Age` | 0 | 0.00% |
| `Outcome` | 0 | 0.00% |

**Discovery Insight**: While there are **0 explicit NaN/null entries**, the Pima Indians dataset historically recorded unmeasured biological variables as `0`. These implicit missing values are rigorously investigated in Section H below.

---

## D. Duplicate Records Analysis

- **Total duplicate rows detected**: `0`
- **Duplicate percentage**: `0.00%`
- All 768 rows represent distinct patient profiles. No rows were removed.

---

## E. Unique Values Analysis

| Feature | Distinct Count | Min Value | Max Value | Nature |
|---|---:|---:|---:|---|
| `Pregnancies` | 17 | 0 | 17 | Discrete count |
| `Glucose` | 136 | 0 | 199 | Continuous measurement |
| `BloodPressure` | 47 | 0 | 122 | Continuous measurement |
| `SkinThickness` | 51 | 0 | 99 | Continuous measurement |
| `Insulin` | 186 | 0 | 846 | Continuous measurement |
| `BMI` | 248 | 0.0 | 67.1 | Continuous measurement |
| `DiabetesPedigreeFunction` | 517 | 0.078 | 2.420 | Continuous ratio |
| `Age` | 52 | 21 | 81 | Discrete integer |
| `Outcome` | 2 | 0 | 1 | Binary target |

---

## F. Numerical Statistics

Detailed distribution statistics computed directly on `dataset/diabetes.csv`:

| Feature | Count | Mean | Std Dev | Min | 25% (Q1) | Median (Q2) | 75% (Q3) | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`Pregnancies`** | 768 | 3.85 | 3.37 | 0.00 | 1.00 | 3.00 | 6.00 | 17.00 |
| **`Glucose`** | 768 | 120.89 | 31.97 | 0.00 | 99.00 | 117.00 | 140.25 | 199.00 |
| **`BloodPressure`** | 768 | 69.11 | 19.36 | 0.00 | 62.00 | 72.00 | 80.00 | 122.00 |
| **`SkinThickness`** | 768 | 20.54 | 15.95 | 0.00 | 0.00 | 23.00 | 32.00 | 99.00 |
| **`Insulin`** | 768 | 79.80 | 115.24 | 0.00 | 0.00 | 30.50 | 127.25 | 846.00 |
| **`BMI`** | 768 | 31.99 | 7.88 | 0.00 | 27.30 | 32.00 | 36.60 | 67.10 |
| **`DiabetesPedigreeFunction`** | 768 | 0.47 | 0.33 | 0.078 | 0.244 | 0.373 | 0.626 | 2.420 |
| **`Age`** | 768 | 33.24 | 11.76 | 21.00 | 24.00 | 29.00 | 41.00 | 81.00 |

---

## G. Potential Identifiers & Administrative Fields

- **Screening Result**: **No identifier columns present**.
- There are no patient ID fields, serial numbers, or administrative timestamps.
- All 8 input attributes represent real biometric and demographic indicators.
- **Risk Assessment**: No columns require exclusion due to identifier leakage.

---

## H. Special Zero-Value Investigation & Dataset Limitations

### In-Depth Zero-Value Breakdown

In accordance with Phase 2 instructions, each feature was audited for zero values to evaluate physiological plausibility without performing automatic data modifications:

| Feature | Zero Count | Zero Percentage (%) | Physiologically / Clinically Plausible as Zero? | Domain Justification & Later Phase Action |
|---|---:|---:|:---:|---|
| **`Glucose`** | 5 | 0.65% | **NO** | Plasma glucose of 0 mg/dl is incompatible with life. Represents unrecorded lab test. **Requires imputation decision in Phase 4.** |
| **`BloodPressure`** | 35 | 4.56% | **NO** | Diastolic blood pressure of 0 mm Hg indicates absence of circulation (cardiac arrest). Represents unmeasured clinical vitals. **Requires imputation decision in Phase 4.** |
| **`SkinThickness`** | 227 | 29.56% | **NO** | Triceps skinfold caliper measurement cannot physically be 0 mm. Indicates omitted physical exam measurement in nearly 30% of cohort. **Requires imputation decision in Phase 4.** |
| **`Insulin`** | 374 | 48.70% | **NO** | Fasting/2-hour postprandial serum insulin cannot be absolute 0 $\mu$U/ml for viable subjects. Omitted in 48.7% of cohort due to assay expense. **Requires imputation decision in Phase 4.** |
| **`BMI`** | 11 | 1.43% | **NO** | Body Mass Index cannot be 0 ($BMI = weight/height^2$). Indicates missing height or weight. **Requires imputation decision in Phase 4.** |
| **`Pregnancies`** | 111 | 14.45% | **YES** | 0 indicates a nulliparous female (never pregnant). This is a completely valid, meaningful clinical count. **Must NOT be converted to NaN.** |
| **`DiabetesPedigreeFunction`** | 0 | 0.00% | **YES** | Minimum observed value is 0.078. No zero entries exist. |
| **`Age`** | 0 | 0.00% | **YES** | Minimum age is 21. No zero entries exist. |

### Dataset Limitations & Modeling Implications
1. **Severe Missingness Encoded as Zeros**:
   - `Insulin` (48.70%) and `SkinThickness` (29.56%) have substantial latent missingness. Blind deletion of rows with any zero would eliminate over 50% of the dataset ($N=768 \rightarrow N=392$), drastically biasing the training population.
   - Robust median or iterative imputation (within a pipeline fitted strictly on training data) will be required in Phase 8.
2. **Homogeneous Study Population**:
   - The data is derived exclusively from female patients of Pima Indian heritage aged $\ge 21$. Models trained on this dataset should be documented as reflecting this cohort's risk distribution.
3. **Distribution Skewness**:
   - `Insulin` (max 846, 75th percentile 127.25) and `DiabetesPedigreeFunction` (max 2.42, 75th percentile 0.626) exhibit significant positive right-skewness and extreme upper outliers. Robust scaling or non-linear tree-based models will be well-suited.

---

**Report Certification**: All statistics reported above were calculated programmatically from the authentic raw dataset `dataset/diabetes.csv`. Zero values were documented but **not** modified.
