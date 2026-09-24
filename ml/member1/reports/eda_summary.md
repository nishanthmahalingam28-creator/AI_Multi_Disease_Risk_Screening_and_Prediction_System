# Member 1 — Phase 4 EDA Summary

## Dataset comparison

| Disease | Rows | Columns | Positive/Presence class | Missing values |
|---|---:|---:|---:|---:|
| Heart Disease | 270 | 14 | 120 (44.44%) | 0 |
| Diabetes | 768 | 9 | 268 (34.90%) | 652 |
| Stroke | 5110 | 11 | 249 (4.87%) | 201 |

## Main observations

### Heart Disease
- The target is moderately balanced: 150 Absence and 120 Presence records.
- The strongest absolute numerical target associations are Thallium, Number of vessels fluro, Exercise angina, ST depression, and Max HR.
- IQR screening identified potential outliers in BP, Cholesterol, Max HR, and ST depression.
- No outliers were removed.

### Diabetes
- Outcome 1 represents 34.90% of the dataset.
- Phase 3 biological-zero cleaning leaves 652 missing values across Glucose, BloodPressure, SkinThickness, Insulin, and BMI.
- Insulin has the largest missing count (374; 48.70%), followed by SkinThickness (227; 29.56%).
- Glucose has the strongest numerical correlation with Outcome in this EDA.
- Potential outliers were identified using IQR only; none were removed.
- `Pregnancies = 0` remains a valid value.

### Stroke
- The positive class is 249/5,110 (4.87%), giving an approximately 19.52:1 negative-to-positive ratio.
- BMI has 201 missing values (3.93%).
- BMI missingness is higher among stroke-positive records (16.06%) than stroke-negative records (3.31%) in this dataset.
- Age has the largest absolute numerical correlation with stroke among the analyzed numerical variables.
- `smoking_status = Unknown` and `gender = Other` were retained.
- IQR screening identifies potential high values, especially for average glucose level, but no values were removed.

## Future considerations
These are observations for later phases, not completed modeling decisions:
- Diabetes will require a documented strategy for missing-value handling inside training-only preprocessing.
- Stroke should be evaluated with metrics that reflect the severe class imbalance rather than accuracy alone.
- Feature transformations, selection, preprocessing, and model training must be performed in later phases without data leakage.

## Integrity
Phase 4 used the cleaned datasets only. No rows were removed, no missing values were imputed, no outliers were removed, and no ML models were trained.