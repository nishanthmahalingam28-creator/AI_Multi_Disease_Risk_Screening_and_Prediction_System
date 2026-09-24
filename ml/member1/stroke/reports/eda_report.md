# Stroke — Exploratory Data Analysis

## Dataset overview
- Cleaned dataset: `ml/member1/stroke/data/cleaned.csv`
- Shape: **5110 rows × 11 columns**
- Target: `stroke`
- No Stroke (0) = **4861 (95.13%)**
- Stroke (1) = **249 (4.87%)**
- Positive-to-negative imbalance ratio: **19.52:1**
- Duplicate rows: **0**
- `id` is not present in the cleaned dataset because Phase 3 removed it as an identifier.

## Missingness
The cleaned dataset retains **201 BMI missing values (3.93%)**.

BMI missingness by stroke class:
- No Stroke: **3.31%**
- Stroke: **16.06%**

No BMI values were imputed during EDA.

## Numerical target relationships
Pearson correlations are descriptive associations only.

| Feature | Correlation with stroke |
|---|---:|
| age | 0.245 |
| heart_disease | 0.135 |
| avg_glucose_level | 0.132 |
| hypertension | 0.128 |
| bmi | 0.042 |

Among the numerical variables, age has the largest absolute correlation with stroke (**0.245**), followed by heart_disease (**0.135**) and avg_glucose_level (**0.132**).

## Categorical features
The EDA includes gender, hypertension, heart disease, marital status, work type, residence type, and smoking status. The `Unknown` smoking-status category and the single `Other` gender record were retained.

## Outlier screening
IQR screening was descriptive only and did not remove observations.

| Feature | Outlier count | Percentage of non-missing values |
|---|---:|---:|
| age | 0 | 0.00% |
| avg_glucose_level | 627 | 12.27% |
| bmi | 110 | 2.24% |

## Class imbalance
The stroke-positive class is only **4.87%** of the cleaned dataset. Therefore, later model evaluation should not rely on accuracy alone. No resampling, SMOTE, or class weighting was performed in Phase 4.

## Generated plots
- `reports/plots/target_distribution.png`
- `reports/plots/numerical_distributions.png`
- `reports/plots/numerical_boxplots.png`
- `reports/plots/categorical_vs_stroke.png`
- `reports/plots/bmi_missingness_by_stroke.png`
- `reports/plots/correlation_heatmap.png`

## Phase 4 interpretation
The EDA documents severe target imbalance, BMI missingness, categorical distributions, numerical relationships, and potential outliers. These are descriptive findings for later preprocessing and modeling.