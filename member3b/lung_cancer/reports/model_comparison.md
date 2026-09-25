# Lung Cancer Model Comparison & Evaluation Report

## 1. Study & Preprocessing Reference
* **Disease Name:** Lung Cancer Prediction (Member 3B)
* **Dataset:** `dataset/survey_lung_cancer.csv` (309 samples, 16 columns)
* **Target Feature:** `lung_cancer` (YES: 270, NO: 39, Imbalance: 6.92:1)
* **Step 10 Preprocessing:**
  - Gender mapped (`FEMALE` $\to 0$, `MALE` $\to 1$)
  - 13 symptom indicators passed through unscaled (`{0, 1}`)
  - `age` scaled with `StandardScaler` fitted strictly on $X_{\text{train}}$
  - Group-aware stratified train/test split (247 train / 62 test)
  - Duplicate feature vectors strictly segregated (zero cross-split leakage)
  - Conflicting survey pair (Rows 257 & 272) assigned strictly to train partition

---

## 2. Cross-Validation & Model Selection Methodology
* **Cross-Validation Scheme:** 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) executed strictly inside the training partition (247 samples: 216 YES, 31 NO).
* **Minority Class Representation:** Each training fold contained ~6 minority (NO) samples and ~43 majority (YES) samples.
* **Class Imbalance Strategy:** `class_weight='balanced'` utilized across candidate classifiers to penalize minority misclassification.
* **Primary Selection Criterion:** Highest Mean 5-Fold Cross-Validation **ROC-AUC**.
* **Secondary Selection Criteria:** PR-AUC, Balanced Accuracy, Minority-Class Recall/Specificity.
* **Held-Out Test Policy:** The held-out test partition ($N=62$) was strictly sealed and never touched during hyperparameter tuning, model comparison, or threshold selection.

---

## 3. Candidate Models & Cross-Validation Results

| Model Family | Best Hyperparameters | Mean CV ROC-AUC | Mean CV PR-AUC | Mean CV Balanced Acc | Mean CV Sensitivity | Mean CV Specificity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `{"C": 0.1}` | 0.9441 ± 0.0174 | 0.9920 ± 0.0026 | 0.8640 ± 0.0333 | 0.8614 | 0.8667 |
| **Extra Trees** | `{"max_depth": null, "min_samples_leaf": 4, "n_estimators": 200}` | 0.9335 ± 0.0155 | 0.9906 ± 0.0025 | 0.8635 ± 0.0498 | 0.8889 | 0.8381 |
| **Support Vector Machine** | `{"C": 1.0, "gamma": 0.01}` | 0.9334 ± 0.0264 | 0.9902 ± 0.0042 | 0.8617 ± 0.0270 | 0.8567 | 0.8667 |
| **Random Forest** | `{"max_depth": 5, "min_samples_leaf": 1, "n_estimators": 400}` | 0.9283 ± 0.0195 | 0.9898 ± 0.0032 | 0.7628 ± 0.0548 | 0.9399 | 0.5857 |
| **HistGradientBoosting** | `{"learning_rate": 0.05, "max_depth": 3, "max_iter": 100, "min_samples_leaf": 10}` | 0.9032 ± 0.0270 | 0.9857 ± 0.0045 | 0.7753 ± 0.0653 | 0.9030 | 0.6476 |

---

## 4. Final Champion Model Selection Rationale
* **Selected Model:** **Logistic Regression**
* **Optimal Hyperparameters:** `{"C": 0.1}`
* **Selection Rationale:**
  1. **Primary Metric Leadership:** Logistic Regression achieved the highest mean cross-validation ROC-AUC (**0.9441 ± 0.0174**), outperforming Extra Trees (0.9335), Support Vector Machine (0.9334), Random Forest (0.9283), and HistGradientBoosting (0.9032).
  2. **Superior PR-AUC & Balanced Accuracy:** In addition to ROC-AUC, Logistic Regression ranked first in PR-AUC (**0.9920**) and Balanced Accuracy (**0.8640**).
  3. **Balanced Diagnostic Capability:** While decision-tree ensembles suffered significant specificity drops (Random Forest specificity dropped to 0.5857), Logistic Regression maintained high specificity (**0.8667**) alongside high sensitivity (**0.8614**).
  4. **Interpretability & Clinical Suitability:** L2 regularization ($C=0.1$) prevents overfitting on small survey datasets with discrete categorical predictors while maintaining clear feature attribution.

---

## 5. Held-Out Test Evaluation (Step 10 Test Set, N=62)
Evaluated exactly once on the untouched held-out test set (54 YES, 8 NO):

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **88.71%** | Overall classification correctness (55 / 62) |
| **Balanced Accuracy** | **93.52%** | Mean of Sensitivity (87.04%) and Specificity (100.00%) |
| **Precision** | **100.00%** | Positive predictive value (47 true positives out of 47 positive predictions) |
| **Recall / Sensitivity** | **87.04%** | True positive detection rate (47 / 54) |
| **Specificity** | **100.00%** | True negative detection rate (8 / 8) |
| **F1-Score** | **0.9307** | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9838** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | **0.9979** | Area under Precision-Recall curve |

### Confusion Matrix Breakdown
* **True Negatives (TN):** 8 (Correctly identified negative cases; 100% of actual NO cases)
* **False Positives (FP):** 0 (Zero false alarms on negative cases)
* **False Negatives (FN):** 7 (Underdiagnosed positive cases)
* **True Positives (TP):** 47 (Correctly identified positive cancer cases)

---

## 6. No Data Leakage Audit
- [x] **Test Partition Isolation:** Raw test rows were never used for model training, tuning, or cross-validation.
- [x] **Scaler Isolation:** Preprocessor scaler parameters (age mean and scale) were fitted solely on $X_{\text{train}}$.
- [x] **Group Separation:** Zero duplicate feature vectors cross the train/test split boundary.
- [x] **Conflicting Observations:** Conflicting survey observations (Rows 257 & 272) reside exclusively in training.
- [x] **Single Test Evaluation:** Held-out evaluation was computed once after model selection was locked.

---

## 7. Limitations & Academic Disclaimer
* **Cohort Scale Limitation:** The dataset contains only 309 total observations with 39 negative cases. The held-out test partition includes 8 negative samples. While 100% specificity was observed on these 8 samples, confidence intervals are necessarily wide due to small sample size.
* **Academic Disclaimer:** This model is an academic research screening prediction model developed on survey data. It is not clinically validated, not approved for medical diagnostic use, and does not replace professional clinical assessment, imaging, or biopsy.
