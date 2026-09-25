# Member 1 Cross-Dataset Discovery & Comparative Summary

**Document ID**: `ml/member1/reports/dataset_summary.md`  
**Phase**: Phase 2 — Dataset Discovery & In-Depth Statistical Analysis  
**Analysis Date**: September 24, 2026  
**Datasets Covered**: Heart Disease, Diabetes, Stroke  
**Scope**: Risk screening, prediction, and decision support (**NOT** a medical diagnostic system)  

---

## 1. Cross-Dataset Comparative Profiling Table

All values in this table are calculated programmatically from the authentic raw files stored in `dataset/`:

| Dataset | Rows | Columns | Target Column | Positive/Class 1 Count | Positive/Class 1 % | Missing Values Count (%) | Duplicate Rows (%) | Identifier Candidate |
|:---|---:|---:|:---|---:|---:|---:|---:|:---|
| **Heart Disease** (`dataset/heart.csv`) | 270 | 14 | `Heart Disease` | 120 (`'Presence'`) | 44.44% | 0 (0.00%) | 0 (0.00%) | None |
| **Diabetes** (`dataset/diabetes.csv`) | 768 | 9 | `Outcome` | 268 (`1`) | 34.90% | 0 explicit (652 implicit biological zeros) | 0 (0.00%) | None |
| **Stroke** (`dataset/stroke.csv`) | 5,110 | 12 | `stroke` | 249 (`1`) | 4.87% | 201 in `bmi` (3.93%) | 0 (0.00%) | `id` (5,110 unique IDs) |

---

## 2. Important Findings for Later Phases

### Heart Disease Module
1. **Target Encoding Required**: The target column `Heart Disease` consists of string labels (`'Absence'` and `'Presence'`). In Phase 6, this will be mapped to binary integers (`Absence` $\rightarrow 0$, `Presence` $\rightarrow 1$).
2. **Compact Cohort ($N = 270$)**: The sample size is modest. Overfitting is a primary hazard. Stratified 5-fold cross-validation and regularized models are mandatory.
3. **Discrete Feature Encoding**: Several numeric columns represent categorical clinical scales (e.g., `Chest pain type`, `Slope of ST`, `Thallium`, `EKG results`). Pipeline preprocessors must apply appropriate encoding.
4. **Clinical vs User Input Pathway**: Features like fluoroscopy vessel count and thallium stress testing require specialized medical testing. In Phases 17–19, a dual-pathway architecture (User Screening vs Clinical Decision Support) will be structured.

### Diabetes Module
1. **Biological Zero-Value Investigation**:
   - `Glucose` ($5$ zeros, $0.65\%$), `BloodPressure` ($35$ zeros, $4.56\%$), `SkinThickness` ($227$ zeros, $29.56\%$), `Insulin` ($374$ zeros, $48.70\%$), and `BMI` ($11$ zeros, $1.43\%$) represent physiologically impossible values that indicate unrecorded data.
   - `Pregnancies` ($111$ zeros, $14.45\%$) is biologically valid for nulliparous individuals and must **not** be modified.
   - **Crucial Rule**: Rows with zeros must **not** be blindly dropped (which would discard $>50\%$ of the dataset). Instead, pipeline imputation within training folds must be applied.

### Stroke Module
1. **Severe Target Imbalance**:
   - Positive rate is only **$4.87\%$** ($249$ vs $4,861$, ratio of $19.52 : 1$).
   - Models must be evaluated using **PR-AUC**, **Recall**, and **F1-Score** rather than raw accuracy. Class weighting (`balanced`) will be critical during training.
2. **Patient Identifier Removal**:
   - The `id` column is an arbitrary integer primary key ($100\%$ unique). It must be excluded before modeling to prevent acute data leakage and overfitting.
3. **Non-Random BMI Missingness**:
   - $16.06\%$ of stroke patients ($40$ out of $249$) have missing BMI, compared to only $3.31\%$ in non-stroke patients.
   - Deleting rows with missing BMI would discard $16\%$ of our rare positive class. BMI must be imputed within the preprocessing pipeline.
4. **Categorical Information**:
   - Features like `gender`, `ever_married`, `work_type`, `Residence_type`, and `smoking_status` require robust `OneHotEncoder(handle_unknown='ignore')` preprocessing. `'Unknown'` in `smoking_status` ($30.22\%$) serves as an informative category and should be retained.

---

## 3. Dataset Integrity Verification

- **Heart dataset hash**: `cba80a969a83288e84f4f981e7fc33fe4ef128a9708959292368e42227c9c2cb` (11,928 bytes)
- **Diabetes dataset hash**: `698c203a14aa31941d2251175330c9199f3ccdb31597abbba2a3e35416257a72` (23,873 bytes)
- **Stroke dataset hash**: `aab4117b8c3c18e7cf7711033abc8adf97595d1a23fc29ea2f07904f68d09815` (332,562 bytes)

*Integrity Confirmation*: All raw datasets in `dataset/` remain unaltered. No rows, columns, or values were modified or imputed during Phase 2.
