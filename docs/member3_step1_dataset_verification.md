# Member 3A — Step 1: Dataset Verification

Branch: `member3`

## Objective

Verify the real Asthma and Parkinson's datasets before cleaning, feature selection, preprocessing, model training, or defining final user/clinical input schemas.

## Asthma dataset

File: `dataset/synthetic_asthma_dataset.csv`

Verified from the repository:

- Rows: 10,000
- Columns: 17
- Missing values: 0 in every column
- Duplicate rows: 0
- Target: `Has_Asthma`
- Target distribution:
  - 0: 7,567 (75.67%)
  - 1: 2,433 (24.33%)
- `Patient_ID`: unique identifier; every row has a unique ID.
- No constant columns were found.
- `Asthma_Control_Level` contains: `N/A`, `Poorly Controlled`, `Not Controlled`, `Well Controlled`.

Columns:

```
Patient_ID
Age
Gender
BMI
Smoking_Status
Family_History
Allergies
Air_Pollution_Level
Physical_Activity_Level
Occupation_Type
Comorbidities
Medication_Adherence
Number_of_ER_Visits
Peak_Expiratory_Flow
FeNO_Level
Has_Asthma
Asthma_Control_Level
```

### Step-1 decisions

1. `Patient_ID` is an identifier and must not be used as a predictive feature.
2. `Asthma_Control_Level` requires leakage analysis before it can be used. It may represent a post-outcome/control-status field rather than an independent screening input.
3. The target mapping must be explicitly recorded before training: `0 -> No Asthma`, `1 -> Asthma`, subject to the dataset's actual semantics.
4. Because this is a synthetic dataset, model performance must not be presented as clinical validation.

## Parkinson's dataset

File: `dataset/Parkinsons_Disease_Dataset.xlsx`

The repository file is present on `member3`.

The working procedure identifies the Parkinson dataset as a voice-measurement classification dataset with target `status`, identifier/subject field `name`, and voice features including fundamental-frequency, jitter, shimmer, noise/harmonic, RPDE, DFA, spread, D2 and PPE measurements.

### Mandatory verification before Step 2

The actual workbook must be opened and measured for:

- row and column count
- exact column names
- data types
- missing values
- duplicate rows
- target values and class counts
- number of unique subjects
- recordings per subject
- whether subject IDs are encoded in `name`
- identifier columns
- constant columns
- numeric feature ranges

This is especially important because multiple recordings from the same subject must never be split across train/test or validation folds.

### Current status

Asthma verification is complete.

Parkinson workbook presence is verified, but the binary XLSX contents could not be parsed in the current repository connector environment. Therefore **no Parkinson row count, missing-value count, duplicate count, or target distribution is being claimed yet**.

## Step-1 completion gate

Step 2 (cleaning) should begin only after the Parkinson workbook verification above has been executed successfully.

## Local verification command

After cloning/checking out `member3` and installing the project requirements:

```bash
python scripts/verify_member3_datasets.py
```

The script prints the actual workbook/CSV schema and quality statistics and is intended to be the authoritative repeatable Step-1 check.
