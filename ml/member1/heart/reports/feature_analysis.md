# Heart Disease — Phase 5 Feature Analysis

## Feature inventory

| Feature | Role/type | Decision | Reason |
|---|---|---|---|
| Age | Numerical | Retain | Predictive candidate; no leakage identified. |
| Sex | Discrete categorical/code | Retain | Existing predictor; encode later as appropriate. |
| Chest pain type | Discrete categorical/code | Retain | Preserve observed clinical-coded levels. |
| BP | Numerical | Retain | Numeric predictor; outliers are descriptive only. |
| Cholesterol | Numerical | Retain | Numeric predictor; no leakage identified. |
| FBS over 120 | Binary/discrete | Retain | Existing predictor; preserve observed values. |
| EKG results | Discrete categorical/code | Retain | Preserve low-frequency levels. |
| Max HR | Numerical | Retain | Numeric predictor with observed target association. |
| Exercise angina | Binary/discrete | Retain | Existing predictor with target association. |
| ST depression | Numerical | Retain | Numeric predictor; transformation can be considered later. |
| Slope of ST | Discrete categorical/code | Retain | Preserve observed categories. |
| Number of vessels fluro | Discrete count/code | Retain | Observed association; no leakage identified. |
| Thallium | Discrete categorical/code | Retain | Strong descriptive target association; preserve categories. |

Target Heart Disease is not a predictor.

## Leakage assessment

No identifier or target-derived predictor was identified in the cleaned feature set. No feature is removed solely because of correlation strength.

## Feature engineering decision

No new engineered predictor is created in Phase 5. The available EDA does not justify arbitrary interactions or a manual risk score. Later preprocessing may encode categorical/discrete variables and scale numerical variables inside training-only pipelines.

## Later handling plan
- Numerical features: assess scaling or transformations inside training pipelines.
- Discrete/categorical codes: determine appropriate encoding inside training pipelines.
- Low-frequency categories: preserve unless a later documented strategy is justified.
- Outliers: do not remove automatically; evaluate model sensitivity later.

## Status

Phase 5 feature analysis is complete. No source CSV was modified, no rows were removed, and no learned preprocessing was fitted.