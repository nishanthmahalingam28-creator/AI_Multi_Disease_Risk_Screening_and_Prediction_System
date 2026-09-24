# Heart Disease — Exploratory Data Analysis

## Dataset overview
- Cleaned dataset: `ml/member1/heart/data/cleaned.csv`
- Shape: **270 rows × 14 columns**
- Target: `Heart Disease`
- Target classes: Absence (0) = **150 (55.56%)**; Presence (1) = **120 (44.44%)**
- Explicit missing values: **0**
- Duplicate rows: **0**

## Numerical feature distributions
| Feature | Mean | Median | Std. Dev. | Min | Max |
|---|---:|---:|---:|---:|---:|
| Age | 54.43 | 55.00 | 9.11 | 29.00 | 77.00 |
| BP | 131.34 | 130.00 | 17.86 | 94.00 | 200.00 |
| Cholesterol | 249.66 | 245.00 | 51.69 | 126.00 | 564.00 |
| Max HR | 149.68 | 153.50 | 23.17 | 71.00 | 202.00 |
| ST depression | 1.05 | 0.80 | 1.15 | 0.00 | 6.20 |

## Target relationships
Pearson correlations with the binary target are descriptive associations only and do not establish causation.

| Feature | Correlation with target |
|---|---:|
| Thallium | 0.525 |
| Number of vessels fluro | 0.455 |
| Exercise angina | 0.419 |
| Max HR | -0.419 |
| ST depression | 0.418 |
| Chest pain type | 0.417 |
| Slope of ST | 0.338 |
| Sex | 0.298 |
| Age | 0.212 |
| EKG results | 0.182 |
| BP | 0.155 |
| Cholesterol | 0.118 |
| FBS over 120 | -0.016 |

The strongest absolute numerical associations with the target are observed for Thallium (**0.525**), Number of vessels fluro (**0.455**), Exercise angina (**0.419**), ST depression (**0.418**), and Max HR (**-0.419**).

## Discrete feature distributions
The categorical/discrete EDA plot covers the clinical-coded variables. Low-frequency values were retained; no category was removed.

## Outlier screening
IQR screening was used only to identify observations outside 1.5×IQR bounds. No observations were removed or transformed.

| Feature | Outlier count | Percentage |
|---|---:|---:|
| Age | 0 | 0.00% |
| BP | 9 | 3.33% |
| Cholesterol | 5 | 1.85% |
| Max HR | 1 | 0.37% |
| ST depression | 4 | 1.48% |

## Generated plots
- `reports/plots/target_distribution.png`
- `reports/plots/numerical_distributions.png`
- `reports/plots/numerical_boxplots.png`
- `reports/plots/categorical_vs_target.png`
- `reports/plots/correlation_heatmap.png`

## Phase 4 interpretation
The EDA identifies distributions, class proportions, numerical associations, and potential outliers for later ML work. These findings are descriptive and are not medical diagnoses or causal conclusions.