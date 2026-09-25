# Lung Cancer Prediction Module

## STEP 14 — FLASK REST API LAYER COMPLETE

> [!IMPORTANT]
> **Status Check:**
> The Lung Cancer Flask REST API Layer is complete.
> **Backend / ML only.** No frontend, HTML, CSS, JavaScript, React, UI, or dashboard components have been created. Step 14 strictly establishes the HTTP REST interface layer.

---

## 1. Overview
* **Disease Name:** Lung Cancer Prediction
* **Module Ownership:** Member 3B
* **Dataset Path:** `dataset/survey_lung_cancer.csv` (Read-only source dataset; verified SHA-256: `b5df44c7a33d095457bd67d59806ce96797d2aef591781af6ef4f1d93c5ac3d2`)
* **Current Status:** Step 14 Complete (Flask REST API Layer Implemented, Tested, and Verified).
* **Selected Model:** **Logistic Regression** ($C=0.1$, `class_weight='balanced'`, `solver='lbfgs'`).

---

## 2. Dataset Dimensions & Physical Integrity
* **File Path:** `dataset/survey_lung_cancer.csv`
* **File Size:** `12,188` bytes
* **Encoding:** `UTF-8`
* **Delimiter:** Comma (`,`)
* **Line Count:** `311` lines (1 header line + 309 data rows)
* **Dataset Dimensions:** **309 rows $\times$ 16 columns**
* **Cryptographic Hash (SHA-256):** `b5df44c7a33d095457bd67d59806ce96797d2aef591781af6ef4f1d93c5ac3d2` (Verified 100% byte-for-byte unchanged)

---

## 3. End-to-End System Architecture

The Flask REST API layer provides a standardized HTTP interface that delegates prediction, feature validation, and capability checks down through the architectural stack:

```text
HTTP Client Request
         ↓
Flask REST API (api.py)
  [create_app() / Route Handlers / HTTP Validation]
         ↓
Lung Cancer Service (service.py)
  [LungCancerPredictionService]
         ↓
Lung Cancer Prediction Engine (predict.py)
  [predict_user() / predict_clinical() / validate_input()]
         ↓
Preprocessor (lung_cancer_preprocessor.joblib)
  [StandardScaler on 'age', passthrough on 14 binary features]
         ↓
Selected Logistic Regression Model (lung_cancer_model.joblib)
  [Log-odds probability calculation with threshold = 0.50]
         ↓
Structured JSON Prediction Response
```

### Single Source of Truth
The Flask API layer (`api.py`):
* Does **not** perform ML inference or feature transformation
* Does **not** fit scalers or compute probabilities
* Delegates all service queries directly to `LungCancerPredictionService`
* Acts strictly as an HTTP transport adapter and security exception boundary

---

## 4. Flask REST API Endpoints

Located in [`member3b/lung_cancer/src/api.py`](file:///p:/AI_Multi_Disease_Risk_Screening_and_Prediction_System/member3b/lung_cancer/src/api.py):

### 4.1 Health Check Endpoint
```http
GET /api/lung-cancer/health
```
* **Success Status:** `200 OK`
* **Unavailable Status:** `503 Service Unavailable`
* **Response Body:**
  ```json
  {
    "status": "ready",
    "service": "lung-cancer-prediction",
    "disease": "lung_cancer",
    "model_loaded": true,
    "preprocessor_loaded": true
  }
  ```

### 4.2 Model Information Endpoint
```http
GET /api/lung-cancer/model-info
```
* **Success Status:** `200 OK`
* **Response Body:**
  ```json
  {
    "status": "success",
    "service": "lung-cancer-prediction",
    "disease": "lung_cancer",
    "model": "LogisticRegression",
    "threshold": 0.5,
    "feature_count": 15,
    "target_mapping": {
      "NO": 0,
      "YES": 1
    },
    "class_mapping": {
      "0": "NO",
      "1": "YES"
    },
    "features": [
      "gender", "age", "smoking", "yellow_fingers", "anxiety",
      "peer_pressure", "chronic_disease", "fatigue", "allergy",
      "wheezing", "alcohol_consuming", "coughing", "shortness_of_breath",
      "swallowing_difficulty", "chest_pain"
    ]
  }
  ```

### 4.3 User Screening Prediction Endpoint
```http
POST /api/lung-cancer/predict/user
```
* **Content-Type:** `application/json`
* **Request Body:**
  ```json
  {
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
  ```
* **Success Status:** `200 OK`
* **Response Body:**
  ```json
  {
    "status": "success",
    "service": "lung-cancer-prediction",
    "disease": "lung_cancer",
    "prediction": 1,
    "prediction_class": "YES",
    "probability": 0.8709,
    "risk_percentage": 87.09,
    "model": "LogisticRegression",
    "threshold": 0.5,
    "screening_interpretation": "Higher predicted risk"
  }
  ```

### 4.4 Clinical Screening Prediction Endpoint
```http
POST /api/lung-cancer/predict/clinical
```
* **Content-Type:** `application/json`
* **Request Body:** Same 15-feature contract as user endpoint.
* **Success Status:** `200 OK`
* **Response Consistency:** Produces identical prediction, probability, and risk percentage as the user endpoint for identical inputs.

---

## 5. Request Validation & Security Boundaries

### 5.1 Strict Request Validation
* **Missing / Non-JSON Content-Type:** Rejected with `400 Bad Request`.
* **Empty Request Body:** Rejected with `400 Bad Request`.
* **Malformed JSON:** Rejected with `400 Bad Request`.
* **Non-Object JSON (arrays, strings, numbers, null):** Rejected with `400 Bad Request`.
* **Duplicate JSON Keys:** Detected during JSON parsing via custom `object_pairs_hook` and rejected with `400 Bad Request` (`Duplicate JSON field: <key>`).
* **Feature Schema Validation:** Missing features, extra features, invalid age ranges, non-binary symptom values, and non-numeric values are rejected with `400 Bad Request` (`error_type: "validation_error"`).
* **NaN & Infinity Rejection:** Rejection of `NaN`, `+Inf`, and `-Inf` with `400 Bad Request`.

### 5.2 Standardized HTTP Status Codes
* **`200 OK`:** Successful health check, model info, or prediction.
* **`400 Bad Request`:** Validation failures, malformed JSON, missing fields, or duplicate keys.
* **`404 Not Found`:** Unknown API endpoints return standardized JSON.
* **`405 Method Not Allowed`:** Disallowed HTTP methods return standardized JSON.
* **`500 Internal Server Error`:** Unexpected server exceptions return generic safe JSON.
* **`503 Service Unavailable`:** Missing model or preprocessor artifacts.

### 5.3 Information Leakage Prevention
* Stack traces, tracebacks, and internal Python error dumps are strictly suppressed.
* Absolute paths (`P:\...`, `/home/...`, `/Users/...`) and usernames are redacted.
* Debug mode is disabled by default.

---

## 6. Champion Model Performance Summary (Step 11 Review)

* **Selected Champion:** **Logistic Regression** ($C=0.1$, `class_weight='balanced'`, `solver='lbfgs'`)
* **Mean 5-Fold CV ROC-AUC:** **0.9441 ± 0.0174**
* **Held-Out Test Set ($N=62$, 54 YES, 8 NO):**
  * **Accuracy:** **88.71%** (55 / 62)
  * **Balanced Accuracy:** **93.52%**
  * **Precision:** **100.0%** (0 False Positives)
  * **Recall / Sensitivity:** **87.04%** (47 / 54)
  * **Specificity:** **100.0%** (8 / 8)
  * **ROC-AUC:** **0.9838**
  * **PR-AUC:** **0.9979**
  * **Confusion Matrix:** $\text{TN}=8, \text{FP}=0, \text{FN}=7, \text{TP}=47$

---

## 7. Verification & Test Commands

Run the complete test suite:

```bash
# 1. Run Lung Cancer Flask API tests specifically (Step 14)
python -m pytest member3b/lung_cancer/tests/test_api.py -v

# 2. Run complete Lung Cancer test suite (analysis + preprocessing + model + prediction + service + api)
python -m pytest member3b/lung_cancer/tests/ -v

# 3. Run Breast Cancer regression suite (ensures zero regression)
python -m pytest member3b/breast_cancer/tests/ -v

# 4. Run full repository test suite
python -m pytest -q
```

### Current Test Results:
* **Lung Cancer Module:** **149 passed, 0 failed**
  * `test_dataset.py`: 5 passed
  * `test_preprocessing.py`: 21 passed
  * `test_model.py`: 13 passed
  * `test_predict.py`: 28 passed
  * `test_service.py`: 35 passed
  * `test_api.py`: 47 passed
* **Breast Cancer Module:** **71 passed, 0 failed**
* **Repository Total:** **220 passed, 0 failed**

---

## 8. Medical & Academic Disclaimer
> [!IMPORTANT]
> **Academic & Research Disclaimer:**
> This system provides a machine-learning-based lung cancer risk/screening prediction from the supplied survey features. It is not a medical diagnosis and must not be used as a substitute for professional medical evaluation. The model was developed and evaluated on the supplied dataset and has not been established here as a clinically validated diagnostic tool.
