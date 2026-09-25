# AI Multi-Disease Risk Screening and Prediction System

## Project layout

```text
backend/             FastAPI endpoints for heart, diabetes, and stroke
dataset/             Version-controlled raw datasets
docs/                Project and data-verification documentation
ml/member1/          Heart, diabetes, and stroke modules
ml/member3/          Asthma and Parkinson's modules
member3b/            Flask cancer-prediction service (breast and lung)
scripts/             Repository-level utility scripts
```

## Training the Member 3 modules

```powershell
python -m ml.member3.asthma.train
python -m ml.member3.parkinsons.train
```

Training creates the model artifacts and evaluation reports in each disease module.

Important limitations:
- The Asthma dataset is synthetic, so model performance does not establish clinical validity or real-world medical effectiveness.
- Both modules are screening/research decision-support tools and must not be presented as medical diagnosis.
- Parkinson's validation is group-aware so recordings from the same subject do not cross training and validation groups.
