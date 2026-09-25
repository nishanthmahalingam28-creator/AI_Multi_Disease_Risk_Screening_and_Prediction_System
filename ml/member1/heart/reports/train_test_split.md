# Heart Disease — Phase 7 Train/Test Split

- Total rows: 270
- Training rows: 216
- Test rows: 54
- Test size: 0.20
- Random state: 42
- Stratified: Yes
- Target: Heart Disease

## Full distribution

| Class | Count | Percentage |
|---|---:|---:|
| 0 | 150 | 55.56% |
| 1 | 120 | 44.44% |

## Stratified split distribution

| Set | Class 0 | Class 1 | Total |
|---|---:|---:|---:|
| Train | 120 | 96 | 216 |
| Test | 30 | 24 | 54 |

The split is generated with sklearn train_test_split and stratify=y. The test set is not used for preprocessing fitting, training, tuning, or model selection in Phase 7.
