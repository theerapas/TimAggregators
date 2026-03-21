# Model Performance Comparison: Paper vs This Implementation

# 1. Paper Performance (Reference)

## 10-Fold Cross Validation

| Model      | MCC  | F1   | Precision | Accuracy | AUROC | AUPRC |
|------------|------|------|-----------|----------|-------|-------|
| FP & MD    | 0.32 | 0.28 | 0.62      | 0.94     | 0.86  | 0.37  |
| Only FP    | 0.30 | 0.27 | 0.60      | 0.94     | 0.85  | 0.36  |
| Only MD    | 0.07 | 0.14 | 0.13      | 0.88     | 0.66  | 0.11  |

## Leave-One-Drug-Out Validation

| Model      | MCC  | F1   | Precision | Accuracy | AUROC | AUPRC |
|------------|------|------|-----------|----------|-------|-------|
| FP & MD    | 0.28 | 0.26 | 0.55      | 0.94     | 0.71  | 0.23  |
| Only FP    | 0.26 | 0.26 | 0.55      | 0.89     | 0.70  | 0.22  |
| Only MD    | 0.09 | 0.15 | 0.14      | 0.88     | 0.63  | 0.10  |

---

# 2. This Implementation

## Model & Feature Notes

- Model: **Random Forest (same as paper)**
- Library: `scikit-learn` (newer version than original implementation)
- Fingerprint: **Morgan fingerprint (radius = 4, 2048 bits)**

### Feature Differences

| Component | Paper | This Implementation |
|----------|------|--------------------|
| Chemical Features | 4,496 | ~4,530 |
| MD Features | +19 | Not included |
| Total Features | 4,515 | ~4,530 |

**Reason for difference:**
- Slight variation in RDKit descriptor set across versions
- No Molecular Dynamics (MD) features included in this implementation

Therefore, this implementation corresponds to **"Only FP" (chemical-only model)** in the paper, not the full FP & MD model.

---

## Threshold = 0.5 (Paper-aligned Evaluation)

### 10-Fold Cross-Validation

| Metric     | Mean ± Std |
|------------|-----------|
| MCC        | 0.3168 ± 0.1549 |
| F1         | 0.2754 ± 0.1415 |
| Precision  | 0.6667 ± 0.3216 |
| Recall     | 0.1833 ± 0.1052 |
| Accuracy   | 0.9396 ± 0.0104 |
| AUROC      | 0.8655 ± 0.0648 |
| AUPRC      | 0.4409 ± 0.1058 |

### Leave-One-Drug-Out Validation

| Metric     | Mean ± Std |
|------------|-----------|
| MCC        | 0.3066 ± 0.2492 |
| F1         | 0.2660 ± 0.2332 |
| Precision  | 0.5938 ± 0.4171 |
| Recall     | 0.2160 ± 0.2679 |
| Accuracy   | 0.9375 ± 0.0515 |
| AUROC      | 0.8055 ± 0.1506 |
| AUPRC      | 0.4547 ± 0.2486 |

---

## Threshold = 0.2 (Screening-Oriented Setting)

### 10-Fold Cross-Validation

| Metric     | Mean ± Std |
|------------|-----------|
| MCC        | 0.3620 ± 0.0786 |
| F1         | 0.4018 ± 0.0700 |
| Precision  | 0.3439 ± 0.0702 |
| Recall     | 0.4989 ± 0.1189 |
| Accuracy   | 0.9035 ± 0.0162 |
| AUROC      | 0.8655 ± 0.0648 |
| AUPRC      | 0.4409 ± 0.1058 |

### Leave-One-Drug-Out

| Metric     | Mean ± Std |
|------------|-----------|
| MCC        | 0.2825 ± 0.2261 |
| F1         | 0.2863 ± 0.2127 |
| Precision  | 0.3326 ± 0.2990 |
| Recall     | 0.3838 ± 0.3322 |
| Accuracy   | 0.9097 ± 0.0452 |
| AUROC      | 0.8055 ± 0.1506 |
| AUPRC      | 0.4547 ± 0.2486 |

---

# 3. Direct Comparison

## Cross-Validation (Main Benchmark)

| Model                        | MCC  | F1   | Precision | Recall | Accuracy | AUROC | AUPRC |
|-----------------------------|------|------|-----------|--------|----------|-------|-------|
| Paper (FP & MD)             | 0.32 | 0.28 | 0.62      | -      | 0.94     | 0.86  | 0.37  |
| Paper (Only FP)             | 0.30 | 0.27 | 0.60      | -      | 0.94     | 0.85  | 0.36  |
| This (Threshold 0.5)        | 0.32 | 0.28 | **0.67**  | 0.18   | 0.94     | **0.87** | **0.44** |
| This (Threshold 0.2)        | **0.36** | **0.40** | 0.34 | **0.50** | 0.90 | **0.87** | **0.44** |

---

# 4. Threshold Analysis (Cross-Validation)

| Threshold | MCC   | F1    | Precision | Recall | Accuracy |
|----------|-------|-------|----------|--------|----------|
| 0.1      | **0.3814** | 0.3931 | 0.2698 | **0.7234** | 0.8542 |
| 0.2      | 0.3611 | **0.4034** | 0.3381 | 0.5000 | 0.9035 |
| 0.3      | 0.2918 | 0.3333 | 0.3625 | 0.3085 | 0.9194 |
| 0.4      | 0.3080 | 0.3165 | 0.4889 | 0.2340 | 0.9340 |
| 0.5      | 0.3158 | 0.2810 | **0.6296** | 0.1809 | **0.9396** |

---

# 5. Key Insights

- This implementation uses the same model architecture (Random Forest) as the paper

- Performance at threshold = 0.5 closely matches the paper’s chemical-only (FP) model

- Differences from paper:
  - Slightly different feature dimension (~4530 vs 4515) due to RDKit version
  - No Molecular Dynamics (MD) features included

- Threshold behavior:
  - Higher threshold → higher precision, lower recall
  - Lower threshold → higher recall, lower precision

- At threshold = 0.2:
  - MCC improves (0.32 → 0.36)
  - F1 improves (0.28 → 0.40)
  - Recall increases significantly (~0.18 → ~0.50)