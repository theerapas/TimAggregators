# Ablation Study

This document tracks the step-by-step ablation study for improving the
drug-excipient co-aggregation prediction pipeline.

## Stage A: Baseline

Stage A used the reproduced baseline setup:

- Features: Morgan fingerprint + RDKit descriptors
- Model: Random Forest
- Imbalance handling: none
- Evaluation:
  - 10-fold cross-validation (CV)
  - Leave-One-Drug-Out (LOGO)

The dataset is highly imbalanced:

- Positive pairs: 94
- Negative pairs: 1346

Because the goal is to identify promising nanoparticle candidates for unseen
drugs, LOGO performance is treated as the more important validation setting.

## Stage B: Imbalance Handling

Stage B tested whether imbalance handling improves performance over the
baseline. The tested methods were:

| Stage | Method |
| --- | --- |
| A0 | No imbalance handling |
| B1 | Class weight |
| B2 | Random oversampling |
| B3 | SMOTE |
| B4 | Threshold tuning only |
| B5 | Class weight + threshold tuning |

Oversampling and SMOTE were applied only inside each training fold after the
train/test split, so the validation fold remained untouched.

## Main Result

Imbalance handling did not clearly improve LOGO ranking performance. The
baseline still had the best LOGO AUPRC.

| Method | LOGO AUROC | LOGO AUPRC |
| --- | ---: | ---: |
| A0: no imbalance | 0.8055 | **0.4547** |
| B1: class weight | **0.8306** | 0.4472 |
| B2: random oversampling | 0.8271 | 0.4475 |
| B3: SMOTE | 0.8177 | 0.4439 |
| B5: class weight + threshold tuning | **0.8306** | 0.4472 |

Class weighting and oversampling slightly improved AUROC, but they did not
improve AUPRC. Since AUPRC is more informative for an imbalanced discovery
task, these methods are not clear ranking improvements over the baseline.

## Threshold-Dependent Performance

Although imbalance handling did not improve ranking, threshold tuning improved
threshold-dependent metrics such as MCC and F1.

Top LOGO MCC results:

| Method | Threshold | MCC | F1 | Precision | Recall | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B5: class weight + threshold tuning | 0.40 | **0.3357** | **0.3203** | 0.5312 | 0.2780 | **0.8306** | 0.4472 |
| B4: threshold tuning only | 0.40 | 0.3329 | 0.2994 | 0.5833 | 0.2598 | 0.8055 | **0.4547** |
| B4: threshold tuning only | 0.35 | 0.3281 | 0.3118 | 0.5208 | 0.2780 | 0.8055 | **0.4547** |
| A0: baseline | 0.50 | 0.3066 | 0.2660 | **0.5938** | 0.2160 | 0.8055 | **0.4547** |

This suggests that the baseline model is reasonable, but the default decision
threshold is not optimal. A threshold around 0.35-0.40 works better for LOGO
MCC than the original 0.20 or 0.50 thresholds.

## CV vs LOGO Behavior

SMOTE looked strong in regular cross-validation:

| Evaluation | Method | Threshold | MCC | F1 | Precision | Recall | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CV | B3: SMOTE | 0.20 | **0.4041** | 0.4243 | 0.3170 | 0.6711 | **0.8763** | 0.4415 |
| CV | B4: threshold tuning only | 0.15 | 0.4010 | **0.4286** | 0.3369 | 0.6178 | 0.8655 | 0.4409 |
| CV | B2: random oversampling | 0.20 | 0.3867 | 0.4100 | 0.3060 | 0.6500 | 0.8671 | 0.4354 |
| CV | A0: baseline | 0.20 | 0.3620 | 0.4018 | 0.3439 | 0.4989 | 0.8655 | 0.4409 |

However, this improvement did not transfer to LOGO. SMOTE had lower LOGO MCC
and AUPRC than threshold tuning and the baseline. This suggests that SMOTE may
help random CV folds but does not provide better generalization to unseen
drugs. This is also chemically plausible, because synthetic interpolation
between sparse fingerprint vectors may not correspond to meaningful molecular
structure.

## Interpretation

The strongest conclusion from Stage B is:

- Imbalance handling alone does not clearly improve candidate ranking.
- Threshold tuning improves threshold-based decisions.
- LOGO should remain the primary selection criterion.
- SMOTE should be treated cautiously because it improves CV but not LOGO.

For discovery, the choice depends on the desired operating point:

- If ranking candidate pairs is the priority, A0/B4 remains strong because it
  keeps the best LOGO AUPRC.
- If a single binary decision threshold is needed, B4 or B5 around threshold
  0.35-0.40 improves MCC and F1.
- If high recall is desired, lower thresholds such as 0.10-0.20 can be used,
  but precision drops substantially.

## Decision for Stage C

Carry forward the top two Stage B setups:

1. B4: no imbalance training + threshold tuning, using threshold around 0.40
2. B5: class weight + threshold tuning, using threshold around 0.40

SMOTE and random oversampling should be recorded as tested but not selected for
the next stage, because they did not improve LOGO AUPRC and did not clearly
outperform threshold tuning on unseen-drug generalization.

For Stage C, new molecular representations should be compared using the same
evaluation structure:

- CV as supporting evidence
- LOGO as the main selection criterion
- AUPRC and MCC as primary metrics
- F1, precision, and recall as secondary operating-point metrics

