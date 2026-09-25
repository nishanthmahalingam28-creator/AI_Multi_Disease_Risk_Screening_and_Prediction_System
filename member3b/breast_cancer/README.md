# Breast Cancer Prediction Module

## 1. Overview
* **Disease Name:** Breast Cancer Prediction
* **Module Ownership:** Member 3B
* **Dataset Path:** `dataset/breast.csv` (Read-only source dataset; verified SHA-256: `1425d9affa78ba8e53afc81d0ef8a19069ee10c4b21fe89b3cf514071b12ee33`)
* **Target Feature:** `diagnosis`
* **Target Mapping:** Deterministic binary mapping:
  * `B` (Benign) $\to$ `0` (357 samples, 62.74%)
  * `M` (Malignant) $\to$ `1` (212 samples, 37.26%)
* **Current Status:** Step 8 Complete (Backend Integration & Production Hardening Finalized).

---

## 2. Experimental Methodology & Leakage Prevention

* **Train / Test Partitioning:**
  * 80% Training ($N = 455$: 285 Benign, 170 Malignant)
  * 20% Held-Out Testing ($N = 114$: 72 Benign, 42 Malignant)
  * Stratification enforced with `stratify=y`, fixed seed `RANDOM_STATE = 42`.
  * The test set was held strictly isolated and never accessed during cross-validation, hyperparameter tuning, or model selection.
* **Preprocessing:**
  * Dropped `id` (non-predictive accession number) and `Unnamed: 32` (100% missing values CSV parsing artifact).
  * 30 continuous morphometric predictor features preserved.
  * `StandardScaler` fitted **strictly on `X_train`**; applied to `X_test` via frozen transformer parameters.
* **Cross-Validation Strategy:**
  * 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) conducted on training data only.
* **Hyperparameter Optimization:**
  * Performed using `GridSearchCV(scoring='roc_auc', cv=5, n_jobs=-1)` on `X_train_scaled`.
* **Primary Selection Metric:** Mean Cross-Validation ROC-AUC (`cv_roc_auc_mean`).

---

## 3. Evaluated Candidate Models & Cross-Validation Results

Five candidate classification algorithms were tuned and benchmarked on the training set:

| Model | CV ROC-AUC (Mean ± Std) | CV Recall (Mean ± Std) | CV Precision (Mean ± Std) | CV F1-Score (Mean ± Std) | CV Accuracy (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.9958 ± 0.0047** | 0.9529 ± 0.0399 | 0.9771 ± 0.0280 | 0.9640 ± 0.0207 | 0.9736 ± 0.0149 |
| **Support Vector Machine** | 0.9954 ± 0.0053 | 0.9529 ± 0.0399 | 0.9889 ± 0.0222 | 0.9698 ± 0.0174 | 0.9780 ± 0.0120 |
| **Extra Trees** | 0.9929 ± 0.0051 | 0.9235 ± 0.0440 | 0.9755 ± 0.0233 | 0.9482 ± 0.0267 | 0.9626 ± 0.0192 |
| **Gradient Boosting** | 0.9921 ± 0.0050 | 0.9412 ± 0.0416 | 0.9755 ± 0.0223 | 0.9577 ± 0.0295 | 0.9692 ± 0.0213 |
| **Random Forest** | 0.9891 ± 0.0062 | 0.9529 ± 0.0300 | 0.9599 ± 0.0279 | 0.9557 ± 0.0132 | 0.9670 ± 0.0098 |

---

## 4. Selected Final Model & Test Set Performance

* **Selected Champion Model:** **Logistic Regression** (L2 Regularized)
* **Optimal Hyperparameters:** `{'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}`
* **Decision Threshold:** Default standard probability threshold (`0.50`).

### Held-Out Test Evaluation ($N = 114$ Unseen Samples):
* **Accuracy:** **96.49%** (110 / 114 correct)
* **ROC-AUC:** **0.9960**
* **Sensitivity / Recall:** **92.86%** (39 / 42 malignant cases detected)
* **Specificity:** **98.61%** (71 / 72 benign cases correctly identified)
* **Precision:** **97.50%** (39 / 40 malignant predictions confirmed)
* **F1-Score:** **0.9512**
* **Confusion Matrix:**
  * True Negatives (TN): **71**
  * False Positives (FP): **1**
  * False Negatives (FN): **3**
  * True Positives (TP): **39**

---

## 5. Prediction Interface Engine & Service Layer

* **Inference Engine (`predict.py`):** Loads `breast_cancer_model.joblib` and `breast_cancer_preprocessor.joblib`, validates inputs, scales features, and calculates calibrated probabilities.
* **Service Layer (`service.py`):** Encapsulated via `BreastCancerPredictionService`. Manages business logic, standardized responses, and lifecycle checks without HTTP dependencies.

---

## 6. Flask REST API

The HTTP API layer is implemented in [member3b/breast_cancer/src/api.py](file:///p:/AI_Multi_Disease_Risk_Screening_and_Prediction_System/member3b/breast_cancer/src/api.py) using an application factory pattern (`create_app()`).

### Architectural Pipeline
```text
Client / Future Frontend
           ↓ HTTP JSON
     Flask REST API (api.py)
           ↓
BreastCancerPredictionService (service.py)
           ↓
     Inference Engine (predict.py)
           ↓
  Saved StandardScaler + Model Artifacts
           ↓
  JSON Response (200 OK / 400 Bad Request)
```

### A. API Endpoints

#### 1. Capability & Health Check
* **Endpoint:** `GET /api/breast-cancer/health`
* **Response Codes:** `200 OK` (operational), `503 Service Unavailable` (artifact loading failure)
* **Response (200):**
  ```json
  {
    "status": "ready",
    "disease": "breast_cancer",
    "model_loaded": true,
    "preprocessor_loaded": true
  }
  ```

#### 2. Model Metadata Information
* **Endpoint:** `GET /api/breast-cancer/model-info`
* **Response Codes:** `200 OK`
* **Response (200):**
  ```json
  {
    "disease": "breast_cancer",
    "model": "Logistic Regression",
    "threshold": 0.50,
    "feature_count": 30,
    "class_mapping": {
      "0": "Benign",
      "1": "Malignant"
    },
    "features": ["radius_mean", "..."]
  }
  ```

#### 3. User Prediction
* **Endpoint:** `POST /api/breast-cancer/predict/user`
* **Headers:** `Content-Type: application/json`
* **Payload Structure:** Accepts grouped (`mean_features`, `se_features`, `worst_features`) or flat 30-feature JSON object.
* **Response Codes:** `200 OK` (success), `400 Bad Request` (missing/invalid JSON or validation error).
* **Success Response (200):**
  ```json
  {
    "status": "success",
    "disease": "breast_cancer",
    "prediction": "Benign",
    "prediction_class": 0,
    "probability": 0.0638,
    "risk_percentage": 6.38,
    "model": "Logistic Regression",
    "threshold": 0.50
  }
  ```

#### 4. Clinical Diagnostic Prediction
* **Endpoint:** `POST /api/breast-cancer/predict/clinical`
* **Headers:** `Content-Type: application/json`
* **Payload Structure:** Full 30-feature fine needle aspirate (FNA) measurement dictionary.
* **Response Codes:** `200 OK` (success), `400 Bad Request` (missing/invalid JSON or validation error).

### B. Error Handling & HTTP Status Codes
* **`400 Bad Request`:** Returned when JSON payload is missing, invalid, or violates schema constraints (e.g. missing required features, NaN, infinite values, extra features).
  ```json
  {
    "status": "error",
    "disease": "breast_cancer",
    "error_type": "validation_error",
    "message": "Missing required feature: radius_mean"
  }
  ```
* **`404 Not Found`:** Returned for unrecognized API paths.
* **`405 Method Not Allowed`:** Returned for invalid HTTP methods (e.g., GET on prediction endpoints).
* **`503 Service Unavailable`:** Returned if model or preprocessor artifacts cannot be loaded.
* **Security & Privacy:** Internal tracebacks and file paths are never exposed in error responses.

---

## 7. Backend Integration & Production Hardening

Step 8 finalizes the backend integration, input validation hardening, and end-to-end service integration for the Breast Cancer Prediction Module.

### A. Architectural Pipeline
```text
Wisconsin Diagnostic Breast Cancer Dataset (dataset/breast.csv)
                               ↓
                 Data Loader (data_loader.py)
                               ↓
             Leakage-Safe Preprocessing (preprocessing.py)
                               ↓
          Trained Logistic Regression Model (train.py)
                               ↓
               Prediction Engine (predict.py)
                               ↓
             Prediction Service Layer (service.py)
                               ↓
              Flask REST API Layer (api.py)
                               ↓
            Client Application / Consuming Systems
```

### B. Prediction Contract (Exactly 30 Model Features)
The model strictly requires the 30 Wisconsin Diagnostic Breast Cancer morphometric cell nucleus features in deterministic ordering:
* **Mean Features (10):** `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`
* **Standard Error Features (10):** `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`
* **Worst / Largest Features (10):** `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`

Excluded non-predictive attributes: `id`, `diagnosis`, and `Unnamed: 32`.

### C. Standardized REST Endpoints

#### 1. Capability & Health Check
* **Endpoint:** `GET /api/breast-cancer/health`
* **Description:** Verifies that both the model classifier and preprocessor pipeline are loaded, cached, and operational.
* **Success Response (`200 OK`):**
  ```json
  {
    "status": "ready",
    "disease": "breast_cancer",
    "model_loaded": true,
    "preprocessor_loaded": true
  }
  ```

#### 2. Model Metadata Information
* **Endpoint:** `GET /api/breast-cancer/model-info`
* **Description:** Retrieves operational model metadata, threshold, feature list, and class mapping without exposing internal filepaths or model weights.
* **Success Response (`200 OK`):**
  ```json
  {
    "disease": "breast_cancer",
    "model": "Logistic Regression",
    "threshold": 0.50,
    "feature_count": 30,
    "class_mapping": { "0": "Benign", "1": "Malignant" },
    "features": ["radius_mean", "..."]
  }
  ```

#### 3. User Risk Screening
* **Endpoint:** `POST /api/breast-cancer/predict/user`
* **Payload Structure:** Accepts grouped (`mean_features`, `se_features`, `worst_features`) or flat 30-feature JSON object.
* **Success Response (`200 OK`):**
  ```json
  {
    "status": "success",
    "disease": "breast_cancer",
    "prediction": "Benign",
    "prediction_class": 0,
    "probability": 0.0638,
    "risk_percentage": 6.38,
    "model": "Logistic Regression",
    "threshold": 0.50
  }
  ```

#### 4. Clinical Diagnostic Prediction
* **Endpoint:** `POST /api/breast-cancer/predict/clinical`
* **Payload Structure:** Full 30-parameter fine needle aspirate (FNA) cytological measurement dictionary.
* **Success Response (`200 OK`):**
  ```json
  {
    "status": "success",
    "disease": "breast_cancer",
    "prediction": "Benign",
    "prediction_class": 0,
    "probability": 0.0638,
    "risk_percentage": 6.38,
    "model": "Logistic Regression",
    "threshold": 0.50
  }
  ```

### D. Production Input Validation & Hardening
The API and prediction validation layers enforce strict validation against all input anomalies:
1. **Missing Features:** Any missing feature among the 30 required attributes returns `HTTP 400` with descriptive error naming the missing feature.
2. **Unexpected Features:** Extra or unmodeled feature keys return `HTTP 400` immediately.
3. **Non-Numeric Values:** Strings and non-numeric entries return `HTTP 400`.
4. **Null / None Values:** Explicit `null` values return `HTTP 400` (`"Feature '...' cannot be None."`).
5. **NaN & Infinity Rejection:** Mathematical `NaN`, `Infinity`, and `-Infinity` are detected and rejected with `HTTP 400`.
6. **Duplicate JSON Keys:** A custom JSON parser hook rejects payloads containing duplicate keys with `HTTP 400` (`"Duplicate feature key encountered: '...'"`).
7. **Safe Error Masking:** Stack traces, internal file paths, model paths, and environment variables are never returned in client error responses.

### E. Local Prediction Latency Benchmark
Benchmarked over 10 sequential inference requests via the Flask application test client:
* **Minimum Latency:** ~2.38 ms
* **Maximum Latency:** ~3.20 ms
* **Average Latency:** ~2.85 ms

---

## 8. How to Run Automated Unit & Integration Tests
From the repository root, execute pytest across all 71 automated tests:
```bash
python -m pytest member3b/breast_cancer/tests/ -v
```
Test suite coverage:
* `test_integration.py` (12 tests): End-to-end integration covering health check, model-info, user/clinical prediction workflows, cross-mode consistency, missing feature rejection, unexpected feature rejection, non-numeric rejection, null rejection, NaN/infinity rejection, duplicate key rejection, and artifact availability.
* `test_api.py` (15 tests): HTTP status codes, content-type checks, validation error handling, grouped/flat payload parsing, cross-mode consistency, and live test-client inference.
* `test_service.py` (13 tests): Service lifecycle, health check, model info, user/clinical modes, delegation, error propagation.
* `test_prediction.py` (13 tests): Input validation, probability ranges, feature order invariance, class mapping.
* `test_preprocessing.py` (9 tests): Data loading, cleaning, deterministic encoding, leakage-free scaling.
* `test_training.py` (6 tests): Candidate model tuning, artifact serialization, metadata validation.
* `test_evaluation.py` (3 tests): Diagnostic metric computation (Sensitivity, Specificity, ROC-AUC).

---

## 9. Medical & Project Disclaimer
> [!IMPORTANT]
> **Academic & Software Research Disclaimer:**
> This Breast Cancer prediction module and REST API are machine-learning screening/risk prediction components developed for an academic software project. They are **NOT** a medical diagnosis and must not replace evaluation by a qualified healthcare professional. All reported metrics represent retrospective experimental results on the Wisconsin Diagnostic Breast Cancer dataset and do not establish clinical effectiveness, medical safety, or regulatory approval.
