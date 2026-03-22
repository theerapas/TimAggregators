# Model Performance Comparison: Paper vs This Implementation

# 1. Paper Performance (Reference)

## 10-Fold Cross Validation

| Model   | MCC  | F1   | Precision | Accuracy | AUROC | AUPRC |
| ------- | ---- | ---- | --------- | -------- | ----- | ----- |
| FP & MD | 0.32 | 0.28 | 0.62      | 0.94     | 0.86  | 0.37  |
| Only FP | 0.30 | 0.27 | 0.60      | 0.94     | 0.85  | 0.36  |
| Only MD | 0.07 | 0.14 | 0.13      | 0.88     | 0.66  | 0.11  |

## Leave-One-Drug-Out Validation

| Model   | MCC  | F1   | Precision | Accuracy | AUROC | AUPRC |
| ------- | ---- | ---- | --------- | -------- | ----- | ----- |
| FP & MD | 0.28 | 0.26 | 0.55      | 0.94     | 0.71  | 0.23  |
| Only FP | 0.26 | 0.26 | 0.55      | 0.89     | 0.70  | 0.22  |
| Only MD | 0.09 | 0.15 | 0.14      | 0.88     | 0.63  | 0.10  |

---

# 2. This Implementation

## Model & Feature Notes

- Models evaluated:
  - **Random Forest** (same as the paper)
  - **ExtraTrees**
  - **Logistic Regression**
- Library: `scikit-learn`
- Fingerprint: **Morgan fingerprint (radius = 4, 2048 bits)**

### Feature Differences

| Component         | Paper | This Implementation |
| ----------------- | ----- | ------------------- |
| Chemical Features | 4,496 | ~4,530              |
| MD Features       | +19   | Not included        |
| Total Features    | 4,515 | ~4,530              |

**Reason for difference:**

- Slight variation in RDKit descriptor set across versions
- No Molecular Dynamics (MD) features included in this implementation

Therefore, this implementation corresponds to **"Only FP" (chemical-only model)** in the paper, not the full FP & MD model.

---

# 3. Direct Comparison

## Cross-Validation (Threshold = 0.5)

| Model               | MCC    | F1         | Precision  | Recall     | Accuracy | AUROC      | AUPRC      |
| ------------------- | ------ | ---------- | ---------- | ---------- | -------- | ---------- | ---------- |
| Paper (FP & MD)     | 0.32   | 0.28       | 0.62       | -          | 0.94     | 0.86       | 0.37       |
| Paper (Only FP)     | 0.30   | 0.27       | 0.60       | -          | 0.94     | 0.85       | 0.36       |
| Random Forest       | 0.3168 | 0.2754     | **0.6667** | 0.1833     | 0.9396   | 0.8655     | **0.4409** |
| ExtraTrees          | 0.2947 | **0.3016** | 0.4750     | **0.2233** | 0.9340   | **0.8740** | 0.4126     |
| Logistic Regression | 0.2523 | 0.2612     | 0.4267     | 0.1933     | 0.9299   | 0.8359     | 0.3748     |

## Cross-Validation (Threshold = 0.2)

| Model               | MCC        | F1         | Precision  | Recall     | Accuracy   | AUROC      | AUPRC      |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Random Forest       | 0.3620     | 0.4018     | 0.3439     | **0.4989** | 0.9035     | 0.8655     | **0.4409** |
| ExtraTrees          | **0.3828** | **0.4189** | **0.3831** | 0.4911     | **0.9118** | **0.8740** | 0.4126     |
| Logistic Regression | 0.3072     | 0.3506     | 0.3151     | 0.4244     | 0.8965     | 0.8359     | 0.3748     |

## Leave-One-Drug-Out (Threshold = 0.5)

| Model               | MCC        | F1         | Precision  | Recall     | Accuracy   | AUROC      | AUPRC      |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Random Forest       | 0.3066     | 0.2660     | 0.5938     | 0.2160     | 0.9375     | 0.8055     | 0.4547     |
| ExtraTrees          | 0.2946     | 0.2599     | 0.5625     | 0.2077     | 0.9368     | 0.8185     | **0.4802** |
| Logistic Regression | **0.3691** | **0.3366** | **0.6250** | **0.2780** | **0.9403** | **0.8274** | 0.4742     |

## Leave-One-Drug-Out (Threshold = 0.2)

| Model               | MCC        | F1         | Precision  | Recall     | Accuracy   | AUROC      | AUPRC      |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Random Forest       | 0.2825     | 0.2863     | 0.3326     | **0.3838** | 0.9097     | 0.8055     | 0.4547     |
| ExtraTrees          | 0.2600     | 0.2637     | 0.3427     | 0.3159     | 0.9188     | 0.8185     | **0.4802** |
| Logistic Regression | **0.3362** | **0.3362** | **0.4312** | 0.3718     | **0.9229** | **0.8274** | 0.4742     |

---

# 4. Threshold Analysis

## Random Forest

| Threshold | MCC        | F1         | Precision  | Recall     | Accuracy   |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 0.1       | **0.3814** | 0.3931     | 0.2698     | **0.7234** | 0.8542     |
| 0.2       | 0.3611     | **0.4034** | 0.3381     | 0.5000     | 0.9035     |
| 0.3       | 0.2918     | 0.3333     | 0.3625     | 0.3085     | 0.9194     |
| 0.4       | 0.3080     | 0.3165     | 0.4889     | 0.2340     | 0.9340     |
| 0.5       | 0.3158     | 0.2810     | **0.6296** | 0.1809     | **0.9396** |

## ExtraTrees

| Threshold | MCC    | F1     | Precision | Recall | Accuracy |
| --------- | ------ | ------ | --------- | ------ | -------- |
| 0.2       | 0.3828 | 0.4189 | 0.3831    | 0.4911 | 0.9118   |
| 0.5       | 0.2947 | 0.3016 | 0.4750    | 0.2233 | 0.9340   |

## Logistic Regression

| Threshold | MCC    | F1     | Precision | Recall | Accuracy |
| --------- | ------ | ------ | --------- | ------ | -------- |
| 0.2       | 0.3072 | 0.3506 | 0.3151    | 0.4244 | 0.8965   |
| 0.5       | 0.2523 | 0.2612 | 0.4267    | 0.1933 | 0.9299   |

---

# 5. Key Insights

- Differences from paper:
  - Slightly different feature dimension (~4530 vs 4515) due to RDKit version
  - No Molecular Dynamics (MD) features included

- Random Forest reproduces strong paper-aligned performance and gives the highest **CV AUPRC** among the three compared models

- ExtraTrees performs best in **CV at threshold 0.2** for MCC and F1:
  - MCC = 0.3828
  - F1 = 0.4189

- ExtraTrees gives the best **LOGO AUPRC**:
  - AUPRC = 0.4802
  - This makes it attractive for **ranking candidate pairs** in screening

- Logistic Regression gives the best **LOGO MCC**, **LOGO F1**, and **LOGO AUROC**:
  - MCC = 0.3691 at threshold 0.5
  - F1 = 0.3366 at threshold 0.5
  - AUROC = 0.8274
  - This suggests the linear model generalizes better to unseen drugs

- Threshold behavior is consistent across models:
  - Lower threshold (0.2) increases recall and F1
  - Higher threshold (0.5) increases precision and accuracy
