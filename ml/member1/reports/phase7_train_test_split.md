# Member 1 — Phase 7 Train/Test Split

## Configuration

- Test size: 0.20
- Random state: 42
- Stratification: Yes
- Split utility: ml/member1/src/split_data.py

| Disease | Total | Train | Test | Test Size | Random State | Stratified |
|---|---:|---:|---:|---:|---:|---|
| Heart Disease | 270 | 216 | 54 | 0.20 | 42 | Yes |
| Diabetes | 768 | 614 | 154 | 0.20 | 42 | Yes |
| Stroke | 5110 | 4088 | 1022 | 0.20 | 42 | Yes |

## Targets

- Heart Disease → Heart Disease
- Diabetes → Outcome
- Stroke → stroke

## Leakage prevention

The split is performed before any preprocessing is fitted. Phase 6 preprocessors remain reusable and unfitted. The test set is reserved for later final evaluation and is not used for training, cross-validation, tuning, feature selection, or threshold selection.

## Disease-specific notes

Stroke uses stratification because its positive class is 4.87% of the cleaned dataset. No resampling or class-weight changes are made during the split.

For Stroke, id is excluded from model features. For Diabetes, Pregnancies = 0 remains unchanged.

## Phase boundary

Phase 7 only establishes the reproducible train/test boundary. Model training begins in the next phase.
