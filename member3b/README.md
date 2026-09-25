# Member 3B — Unified Cancer Prediction Backend

> **Project Purpose:**  
> Member 3B provides an AI-based risk screening prediction and risk estimation backend for:
> - **Breast Cancer Risk Screening** (`member3b/breast_cancer/`)
> - **Lung Cancer Risk Screening** (`member3b/lung_cancer/`)
> - **Unified Multi-Disease Orchestration API** (`member3b/common/`)

> **Medical Safety Disclaimer:**  
> This software is an AI-based machine-learning risk screening and decision-support tool. It provides **screening predictions** and **risk estimates** only. It is **NOT a medical diagnostic system**, does not provide confirmed medical diagnoses, and must never replace clinical judgment or professional medical evaluation, diagnosis, or treatment by qualified healthcare providers.

---

## 1. System Architecture

The Member 3B backend is designed with a decoupled, modular service-oriented architecture:

```text
Client Application / Reverse Proxy
               │
               ▼
┌───────────────────────────────────────────────┐
│     Member 3B Unified REST API (Flask)        │
│   (member3b/app.py & member3b/common/api.py)  │
│  - Envelope validation & route dispatch       │
│  - Controlled error boundary (JSON only)      │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│        CancerPredictionService                │
│       (member3b/common/service.py)            │
│  - Orchestration & health aggregation         │
│  - Dispatches to disease-specific services    │
└──────────────┬──────────────────┬─────────────┘
               │                  │
               ▼                  ▼
┌──────────────────────────┐ ┌──────────────────────────┐
│ Breast Cancer Service    │ │ Lung Cancer Service      │
│ (member3b/breast_cancer) │ │ (member3b/lung_cancer)   │
│ - Strict 30-feature check│ │ - Strict 15-feature check│
│ - StandardScaler pipeline│ │ - StandardScaler pipeline│
│ - Logistic Regression    │ │ - Logistic Regression    │
└──────────────────────────┘ └──────────────────────────┘
```

* **Delegation Principle:** The unified service validates only routing envelope parameters (`disease`, `data`, `mode`) and delegates feature schema validation, domain checking, scaling, and inference strictly to the underlying disease modules.
* **Direct Pass-Through Support:** All original disease-specific endpoints (`/api/breast-cancer/*` and `/api/lung-cancer/*`) remain fully accessible and backwards-compatible.

---

## 2. Directory Structure

```text
member3b/
|-- .env.example                                  <- Environment configuration template
|-- Procfile                                      <- Production WSGI process declaration
|-- README.md                                     <- Complete system documentation & handover
|-- __init__.py                                   <- Package initialization
|-- app.py                                        <- Production WSGI & CLI entry point
|-- requirements.txt                              <- Locked runtime & testing dependencies
|-- breast_cancer/                                <- Breast Cancer screening module
|   |-- README.md                                 <- Module documentation
|   |-- __init__.py
|   |-- models/
|   |   |-- breast_cancer_model.joblib            <- Trained Logistic Regression model
|   |   |-- breast_cancer_preprocessor.joblib     <- Fitted StandardScaler pipeline
|   |   |-- class_mapping.json
|   |   |-- clinical_schema.json
|   |   |-- model_metadata.json
|   |   |-- preprocessing_metadata.json
|   |   \-- user_schema.json
|   |-- reports/                                  <- Evaluation visual assets & CSVs
|   |-- src/
|   |   |-- api.py                                <- Flask blueprint / endpoints
|   |   |-- data_loader.py                        <- Raw dataset loader
|   |   |-- evaluate.py                           <- Model evaluation metrics
|   |   |-- predict.py                            <- Inference & validation engine
|   |   |-- preprocessing.py                      <- Leak-free data transformation
|   |   |-- service.py                            <- Decoupled service layer
|   |   \-- train.py                              <- Model training script
|   \-- tests/                                    <- Automated unit & integration tests
|-- common/                                       <- Unified orchestration layer
|   |-- __init__.py
|   |-- api.py                                    <- Unified Flask REST API factory
|   |-- config.py                                 <- Environment-aware configuration
|   |-- service.py                                <- Unified CancerPredictionService
|   \-- tests/                                    <- Unified, packaging, security tests
\-- lung_cancer/                                  <- Lung Cancer screening module
    |-- README.md                                 <- Module documentation
    |-- __init__.py
    |-- models/
    |   |-- lung_cancer_model.joblib              <- Trained Logistic Regression model
    |   |-- lung_cancer_preprocessor.joblib       <- Fitted StandardScaler pipeline
    |   |-- model_metadata.json
    |   |-- preprocessing_metadata.json
    |   \-- split_metadata.json
    |-- reports/                                  <- Evaluation visual assets & CSVs
    |-- src/
    |   |-- api.py                                <- Flask blueprint / endpoints
    |   |-- data_loader.py                        <- Raw dataset loader
    |   |-- eda_analysis.py                       <- Exploratory data analysis
    |   |-- evaluate.py                           <- Model evaluation metrics
    |   |-- predict.py                            <- Inference & validation engine
    |   |-- preprocessing.py                      <- Leak-free data transformation
    |   |-- service.py                            <- Decoupled service layer
    |   \-- train.py                              <- Model training script
    \-- tests/                                    <- Automated unit & integration tests
```

---

## 3. Disease Modules & Machine Learning Models

### Breast Cancer Risk Screening
* **Clinical Task:** Machine-learning screening prediction based on digitized fine needle aspirate (FNA) cytological features.
* **Selected Model:** `Logistic Regression` with L2 regularization (`C=1.0`, solver=`lbfgs`).
* **Input Features (30 continuous numerical cytological measurements):**
  * *Mean:* `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`
  * *Standard Error:* `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`
  * *Worst / Extreme:* `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`
* **Performance Metrics (from evaluation reports):**
  * Test Accuracy: `0.9825` (98.25%)
  * ROC-AUC: `0.9960`
  * Precision: `0.9767`
  * Recall: `0.9767`
  * F1-Score: `0.9767`
* **Known Limitations:**
  * Dataset size: 569 instances (Wisconsin Diagnostic dataset).
  * Cohort specificity: Retrospective FNA cytology from a single institutional cohort.
  * Absence of prospective clinical trial validation.

### Lung Cancer Risk Screening
* **Clinical Task:** Survey-based risk screening prediction assessing demographic, behavioral, and clinical symptom indicators.
* **Selected Model:** `Logistic Regression` with L2 regularization (`C=0.1`, solver=`lbfgs`).
* **Input Features (15 survey-derived features):**
  1. `gender`: Demographic identifier (`"MALE"` / `"FEMALE"`, case-insensitive, or `0`/`1`).
  2. `age`: Patient age in years (numerical range `[1, 120]`).
  3. `smoking`: History of smoking (`0`=No, `1`=Yes, or `"NO"`/`"YES"`).
  4. `yellow_fingers`: Presence of yellow staining on fingers (`0`/`1`).
  5. `anxiety`: Self-reported chronic anxiety (`0`/`1`).
  6. `peer_pressure`: Exposure to peer smoking pressure (`0`/`1`).
  7. `chronic_disease`: Diagnosed chronic systemic conditions (`0`/`1`).
  8. `fatigue`: Persistent chronic fatigue (`0`/`1`).
  9. `allergy`: History of respiratory allergies (`0`/`1`).
  10. `wheezing`: Respiratory wheezing episodes (`0`/`1`).
  11. `alcohol_consuming`: Alcohol consumption habits (`0`/`1`).
  12. `coughing`: Persistent chronic cough (`0`/`1`).
  13. `shortness_of_breath`: Dyspnea or shortness of breath (`0`/`1`).
  14. `swallowing_difficulty`: Dysphagia or swallowing difficulty (`0`/`1`).
  15. `chest_pain`: Retrosternal or chest pain (`0`/`1`).
* **Performance Metrics (from evaluation reports):**
  * Test Accuracy: `0.9355` (93.55%)
  * ROC-AUC: `0.9630`
  * Sensitivity (Recall): `0.9630`
  * Specificity: `0.7500`
  * F1-Score: `0.9630`
* **Known Limitations:**
  * Dataset size: 309 survey responses.
  * Severe class imbalance: ~87% positive survey records.
  * Self-reported survey features; subjective patient perception.
  * Duplicate and conflicting symptom records identified in raw data.
  * Retrospective analysis; no prospective clinical validation.

---

## 4. API Endpoints & Request/Response Contracts

### A. Unified Member 3B Endpoints

#### 1. Unified Health Check
```http
GET /api/member3b/health
```
* **Status:** `200 OK` (when both disease models are ready) or `503 Service Unavailable`
* **Response (HTTP 200):**
  ```json
  {
    "status": "ready",
    "service": "member3b-cancer-prediction",
    "diseases": {
      "breast_cancer": {
        "status": "ready",
        "disease": "breast_cancer",
        "model_loaded": true,
        "preprocessor_loaded": true
      },
      "lung_cancer": {
        "status": "ready",
        "service": "lung-cancer-prediction",
        "disease": "lung_cancer",
        "model_loaded": true,
        "preprocessor_loaded": true
      }
    }
  }
  ```

#### 2. Unified Model Metadata
```http
GET /api/member3b/model-info
```
* **Status:** `200 OK`
* **Response:** Returns sanitized public model metadata (feature names, counts, thresholds) without exposing file paths, internal weights, or secrets.

#### 3. Unified Cancer Screening Prediction
```http
POST /api/member3b/predict
Content-Type: application/json
```

* **Breast Cancer Request Example:**
  ```json
  {
    "disease": "breast_cancer",
    "data": {
      "radius_mean": 17.99,
      "texture_mean": 10.38,
      "perimeter_mean": 122.8,
      "area_mean": 1001.0,
      "smoothness_mean": 0.1184,
      "compactness_mean": 0.2776,
      "concavity_mean": 0.3001,
      "concave points_mean": 0.1471,
      "symmetry_mean": 0.2419,
      "fractal_dimension_mean": 0.07871,
      "radius_se": 1.095,
      "texture_se": 0.9053,
      "perimeter_se": 8.589,
      "area_se": 153.4,
      "smoothness_se": 0.006399,
      "compactness_se": 0.04904,
      "concavity_se": 0.05373,
      "concave points_se": 0.01587,
      "symmetry_se": 0.03003,
      "fractal_dimension_se": 0.006193,
      "radius_worst": 25.38,
      "texture_worst": 17.33,
      "perimeter_worst": 184.6,
      "area_worst": 2019.0,
      "smoothness_worst": 0.1622,
      "compactness_worst": 0.6656,
      "concavity_worst": 0.7119,
      "concave points_worst": 0.2654,
      "symmetry_worst": 0.4601,
      "fractal_dimension_worst": 0.1189
    }
  }
  ```

* **Breast Cancer Response (HTTP 200):**
  ```json
  {
    "status": "success",
    "disease": "breast_cancer",
    "model": "Logistic Regression",
    "prediction": "Malignant",
    "prediction_class": 1,
    "probability": 1.0,
    "risk_percentage": 100.0,
    "threshold": 0.5
  }
  ```

* **Lung Cancer Request Example:**
  ```json
  {
    "disease": "lung_cancer",
    "data": {
      "gender": "MALE",
      "age": 69,
      "smoking": 0,
      "yellow_fingers": 1,
      "anxiety": 1,
      "peer_pressure": 0,
      "chronic_disease": 0,
      "fatigue": 1,
      "allergy": 0,
      "wheezing": 1,
      "alcohol_consuming": 1,
      "coughing": 1,
      "shortness_of_breath": 1,
      "swallowing_difficulty": 1,
      "chest_pain": 1
    }
  }
  ```

* **Lung Cancer Response (HTTP 200):**
  ```json
  {
    "status": "success",
    "disease": "lung_cancer",
    "service": "lung-cancer-prediction",
    "model": "LogisticRegression",
    "prediction": 1,
    "prediction_class": "YES",
    "probability": 0.8709,
    "risk_percentage": 87.09,
    "screening_interpretation": "Higher predicted risk",
    "threshold": 0.5
  }
  ```

---

### B. Existing Disease-Specific Endpoints

#### Breast Cancer Endpoints
* `GET  /api/breast-cancer/health` — Module health & artifact readiness.
* `GET  /api/breast-cancer/model-info` — 30-feature metadata & decision threshold.
* `POST /api/breast-cancer/predict/user` — User-facing screening prediction.
* `POST /api/breast-cancer/predict/clinical` — Clinical evaluation prediction.

#### Lung Cancer Endpoints
* `GET  /api/lung-cancer/health` — Module health & artifact readiness.
* `GET  /api/lung-cancer/model-info` — 15-feature metadata & decision threshold.
* `POST /api/lung-cancer/predict/user` — User-facing screening prediction.
* `POST /api/lung-cancer/predict/clinical` — Clinical evaluation prediction.

---

## 5. Local Setup & Execution

### Prerequisites
* **Python:** 3.10+ (Verified on **Python 3.14.6**)
* **OS:** Windows 10/11, Linux, macOS

### Installation
1. Clone the repository and navigate to repository root:
   ```bash
   cd p:\AI_Multi_Disease_Risk_Screening_and_Prediction_System
   ```
2. Create and activate a virtual environment:
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r member3b/requirements.txt
   ```

### Local Development Server
Launch the Flask development server:
```bash
python -m member3b.app
```
* **Listening Address:** `http://127.0.0.1:5000`
* **Default Settings:** `DEBUG=False`, `ENV=production`

---

## 6. Environment Configuration

Configuration is managed through environment variables with safe defaults:

| Variable | Description | Default | Allowed Values |
|---|---|---|---|
| `HOST` | Server bind IP address | `127.0.0.1` | Valid IP (use `0.0.0.0` for containers) |
| `PORT` | Listening TCP port | `5000` | Integer in range `1` to `65535` |
| `FLASK_DEBUG` | Enable Flask interactive debugger | `false` | `false`, `0`, `true`, `1` |
| `FLASK_ENV` | Environment identifier | `production` | `production`, `development` |

A clean template is provided at [member3b/.env.example](file:///p:/AI_Multi_Disease_Risk_Screening_and_Prediction_System/member3b/.env.example). Real secrets, tokens, or credentials must never be committed.

---

## 7. Production WSGI Deployment

* **WSGI Target:** `member3b.app:app`
* **WSGI Standard:** 100% compliant with PEP 3333 (verified via `wsgiref.validate.validator`).
* **Production Server:** Gunicorn (for Linux VMs, Docker, Render, Heroku).
* **Procfile:** Located at the repository root:
  ```text
  web: gunicorn member3b.app:app
  ```
* **Recommended Production Command:**
  ```bash
  gunicorn --workers 2 --bind 0.0.0.0:${PORT:-5000} --timeout 60 member3b.app:app
  ```
* **Windows Local Note:** Gunicorn requires POSIX `fcntl` APIs available only on Linux/Unix systems. For Windows local development, use `python -m member3b.app`.

> **Deployment Status:** Production WSGI configuration is verified and deployment-ready. **Cloud deployment has NOT been performed** as part of this project verification; no live cloud URL is claimed.

---

## 8. Model Artifact & Dataset Integrity

All datasets and model artifacts are versioned and strictly preserved. Their SHA-256 hashes must match:

| Resource | Path | SHA-256 Checksum |
|---|---|---|
| Breast Dataset | `dataset/breast.csv` | `1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33` |
| Lung Dataset | `dataset/survey_lung_cancer.csv` | `b5df44c7a33d095457bd67d59806ce96797d2aef591781af6ef4f1d93c5ac3d2` |
| Breast Model | `member3b/breast_cancer/models/breast_cancer_model.joblib` | `0379e06d00ad57a172c6c8c6b91eaba48a619e0783644ead1cf5b3e855b32024` |
| Breast Preprocessor | `member3b/breast_cancer/models/breast_cancer_preprocessor.joblib` | `7dc70392e4bda3f3b53f5cb6b0b6be6a085059b3d706284e392157beb41c3092` |
| Lung Model | `member3b/lung_cancer/models/lung_cancer_model.joblib` | `4c31a9af4b304a21fc9eed29f9c5a9f74caec0cadf02f0d36afde3747eeb6785` |
| Lung Preprocessor | `member3b/lung_cancer/models/lung_cancer_preprocessor.joblib` | `9d7e68f4b1a9bfa81ea8ce4053bfac4d9460d0d13605e8859238747beb47c459` |

Artifacts are resolved using `__file__`-relative paths and function independently of the process working directory.

---

## 9. Security & Reliability Verification

* **Automated Security Tests:** 100 dedicated tests in `member3b/common/tests/test_security_reliability.py`.
* **Vulnerability Audit:** 0 Critical, 0 High, 0 Medium, 0 Low findings.
* **Input Validation:**
  * Rejects missing envelope keys (`disease`, `data`).
  * Rejects non-dict JSON bodies (`[]`, strings, numbers, booleans, null).
  * Rejects duplicate JSON keys at all levels (`DuplicateKeyError` -> HTTP 400).
  * Rejects numerical anomalies (`NaN`, `Infinity`, `-Infinity`, strings for numbers).
  * Validates physiological boundaries (Lung age strictly bounded in `[1, 120]`).
* **Injection Defense:**
  * Path traversal vectors (`../../../../etc/passwd`, `C:\Windows\System32`) safely rejected.
  * Command injection strings (`; whoami`, `$(whoami)`, `` `whoami` ``) safely rejected.
* **Information Disclosure Prevention:**
  * All error responses (400, 404, 405, 500, 503) return sanitized JSON.
  * Zero tracebacks, zero internal source code paths, and zero environment variables leaked.
* **Idempotence & Reliability:**
  * 20 repeated sequential predictions produce 100% deterministic identical outputs.
  * Repeated invalid requests do not degrade server health or corrupt internal state.

---

## 10. Automated Testing

Run the full automated test suite from the repository root:

```bash
python -m pytest -q
```

* **Current Verified Status:** **435 passed**, 0 failed, 0 errors.
* **Test Breakdown:**
  * Breast Cancer tests: 71 passed
  * Lung Cancer tests: 175 passed
  * Common unified tests: 47 passed
  * Packaging & readiness tests (Step 17): 18 passed
  * Production deployment tests (Step 18): 24 passed
  * Security & reliability tests (Step 19): 100 passed
  * **Total: 435 passing tests**
