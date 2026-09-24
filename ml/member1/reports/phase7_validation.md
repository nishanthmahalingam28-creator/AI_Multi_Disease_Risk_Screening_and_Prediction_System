# Member 1 — Phase 7 Validation

## Status
COMPLETE

## Validation checklist

- Reusable stratified split utility created.
- Heart split: 216 train / 54 test.
- Diabetes split: 614 train / 154 test.
- Stroke split: 4088 train / 1022 test.
- Test size is 20% for all diseases.
- Random state is 42.
- Target is separated from predictors.
- Stroke id is excluded from predictors.
- Train/test index sets are disjoint.
- Repeated calls with random_state=42 are reproducible.
- A different random state changes the split.
- Stratification is enabled for all three classification datasets.
- No preprocessing is fitted on the complete dataset.
- No model is trained.
- No CV or hyperparameter tuning is performed.
- No resampling or class-weight optimization is performed.
- No source or Phase 3 cleaned CSV is modified.

## Tests

ml/member1/tests/test_split.py validates row counts, train/test disjointness, reproducibility, target separation, Stroke identifier exclusion, and seed sensitivity.

The split modules do not fit preprocessing or train models.
