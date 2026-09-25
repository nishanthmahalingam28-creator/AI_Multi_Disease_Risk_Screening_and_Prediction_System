# Asthma — Member 3A

Run `python -m asthma.train` from repository root. The pipeline verifies/cleans the synthetic CSV, removes Patient_ID and Asthma_Control_Level from model features, performs EDA, preprocessing, stratified CV, model comparison, tuning and holdout evaluation, then saves model/preprocessor/schema/metadata artifacts.

The dataset is synthetic; performance does not establish clinical validity or real-world medical effectiveness. Outputs are screening/risk estimates, not medical diagnosis.
