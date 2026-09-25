# Diabetes — Exploratory Data Analysis

## Dataset overview
- Cleaned dataset: `ml/member1/diabetes/data/cleaned.csv`
- Shape: **768 rows × 9 columns**
- Target: `Outcome`
- Outcome 0 = **500 (65.10%)**
- Outcome 1 = **268 (34.90%)**
- Duplicate rows: **0**
- Total missing values after Phase 3 cleaning: **652**

## Missingness
Phase 3 converted biologically implausible zeros to missing values in five measurements. No imputation was performed during EDA.

| Feature | Missing count | Missing % |
|---|---:|---:|
| Glucose | 5 | 0.65% |
| BloodPressure | 35 | 4.56% |
| SkinThickness | 227 | 29.56% |
| Insulin | 374 | 48.70% |
| BMI | 11 | 1.43% |

`Pregnancies = 0` was retained as a valid value and was not treated as missing.

## Target relationships
Pearson correlations with the binary target are descriptive associations only.

| Feature | Correlation with Outcome |
|---|---:|
| Glucose | 0.495 |
| BMI | 0.314 |
| Insulin | 0.303 |
| SkinThickness | 0.259 |
| Age | 0.238 |
| Pregnancies | 0.222 |
| DiabetesPedigreeFunction | 0.174 |
| BloodPressure | 0.171 |

The largest absolute numerical association in this EDA is for **Glucose (0.495)**, followed by BMI (**0.314**) and Insulin (**0.303**).

## Numerical feature distributions
The numerical distribution and feature-by-outcome plots were generated from the cleaned dataset. Missing observations were not filled for visualization.

## Outlier screening
IQR screening was descriptive only.

| Feature | Outlier count | Percentage of non-missing values |
|---|---:|---:|
| Pregnancies | 4 | 0.52% |
| Glucose | 0 | 0.00% |
| BloodPressure | 14 | 1.91% |
| SkinThickness | 3 | 0.55% |
| Insulin | 24 | 6.09% |
| BMI | 8 | 1.06% |
| DiabetesPedigreeFunction | 29 | 3.78% |
| Age | 9 | 1.17% |

## Generated plots
- `reports/plots/target_distribution.png`
- `reports/plots/missingness_analysis.png`
- `reports/plots/numerical_distributions.png`
- `reports/plots/features_by_outcome.png`
- `reports/plots/correlation_heatmap.png`

## Phase 4 interpretation
The dataset contains substantial missingness in Insulin and SkinThickness after the Phase 3 biological-zero cleaning. Missingness remains unresolved intentionally; preprocessing and imputation decisions belong to later phases.