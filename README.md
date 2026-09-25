# AI Multi-Disease Risk Screening and Prediction System

## Member 3A — Asthma and Parkinson's Disease

The member3 branch contains the Member 3A ML implementation.

Train Asthma:
`python -m asthma.train`

Train Parkinson's:
`python -m parkinsons.train`

Training generates model files, preprocessors, schemas, metadata and evaluation reports.

Important limitations:
- The Asthma dataset is synthetic, so model performance does not establish clinical validity or real-world medical effectiveness.
- Both modules are screening/research decision-support tools and must not be presented as medical diagnosis.
- Parkinson's validation is group-aware so recordings from the same subject do not cross training and validation groups.
