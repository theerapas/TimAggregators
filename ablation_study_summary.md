# Ablation Study Summary

This document summarizes the completed ablation study for the
drug-excipient co-aggregation prediction pipeline.

## Goal

The main goal was to improve prediction of drug-excipient co-aggregation for
unseen drugs. Because the intended use is candidate discovery, Leave-One-Drug-
Out (LOGO) performance was treated as the primary validation setting.

Primary metrics:

- Ranking/discovery: LOGO AUPRC
- Thresholded binary decision: LOGO MCC

Secondary metrics:

- AUROC
- F1
- Precision
- Recall

The dataset is highly imbalanced:

- Positive pairs: 94
- Negative pairs: 1346

## Final Decision

Use two final setups depending on the downstream goal.

| Use case | Final setup | Threshold | Main result |
| --- | --- | ---: | --- |
| Candidate ranking / discovery | Morgan only + class-weighted Random Forest | 0.30 if threshold needed | Best LOGO AUPRC = 0.4858 |
| Conservative binary decision | Morgan + ChemBERTa + Logistic Regression | 0.40-0.50 | LOGO MCC = 0.3691, LOGO AUPRC = 0.4754 |

The main discovery model remains:

- Features: Morgan fingerprint only
- Model: Random Forest
- Imbalance handling: class weight
- Evaluation priority: LOGO AUPRC

## Baseline vs Improved Models

The original implementation baseline used the setup described in the README
and earlier summary:

- Features: Morgan fingerprint + RDKit descriptors
- Model: Random Forest
- Imbalance handling: none
- Default comparison threshold: 0.50

Baseline LOGO result:

| Setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: Morgan + RDKit + Random Forest | 0.50 | 0.4547 | 0.3066 | 0.2660 | 0.5938 | 0.2160 | 0.8055 |

Final improved setups:

| Use case | Setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranking/discovery | Morgan only + class-weighted Random Forest | 0.30 | **0.4858** | 0.3381 | **0.3451** | 0.3926 | **0.4130** | **0.8426** |
| Conservative binary decision | Morgan + ChemBERTa + Logistic Regression | 0.40 | 0.4754 | **0.3691** | 0.3366 | **0.6250** | 0.2780 | 0.8332 |

Improvement over the baseline:

| Improved setup | AUPRC change | MCC change | F1 change | Precision change | Recall change | AUROC change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ranking/discovery model | +0.0311 | +0.0315 | +0.0791 | -0.2012 | +0.1970 | +0.0371 |
| Conservative binary-decision model | +0.0207 | +0.0625 | +0.0706 | +0.0312 | +0.0620 | +0.0277 |

Interpretation:

- The final ranking model improves LOGO AUPRC, AUROC, MCC, F1, and recall
  over the baseline, but trades away precision because it uses a lower
  operating threshold.
- The conservative binary-decision model improves every listed LOGO metric over
  the baseline, including precision.
- The strongest scientific improvement is not from adding more descriptors or
  embeddings, but from selecting a cleaner representation, using LOGO-oriented
  thresholding, and comparing model families.

## Stage A: Baseline

Baseline setup:

- Features: Morgan fingerprint + RDKit descriptors
- Model: Random Forest
- Imbalance handling: none
- Evaluation: CV and LOGO

Baseline LOGO performance:

| Setup | Threshold | AUPRC | AUROC | MCC |
| --- | ---: | ---: | ---: | ---: |
| Morgan + RDKit + Random Forest | 0.50 | 0.4547 | 0.8055 | 0.3066 |

The baseline was reasonable, but the default threshold was not optimal for
MCC/F1.

## Stage B: Imbalance Handling

Tested methods:

| Stage | Method |
| --- | --- |
| A0 | No imbalance handling |
| B1 | Class weight |
| B2 | Random oversampling |
| B3 | SMOTE |
| B4 | Threshold tuning only |
| B5 | Class weight + threshold tuning |

Main LOGO ranking result:

| Method | LOGO AUROC | LOGO AUPRC |
| --- | ---: | ---: |
| A0: no imbalance | 0.8055 | **0.4547** |
| B1: class weight | **0.8306** | 0.4472 |
| B2: random oversampling | 0.8271 | 0.4475 |
| B3: SMOTE | 0.8177 | 0.4439 |
| B5: class weight + threshold tuning | **0.8306** | 0.4472 |

Main conclusion:

- Imbalance handling did not improve LOGO AUPRC.
- Class weighting improved AUROC but not AUPRC.
- SMOTE helped CV but did not transfer to LOGO.
- Threshold tuning improved MCC/F1.

Best Stage B thresholded rows:

| Method | Threshold | MCC | F1 | Precision | Recall | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B5: class weight + threshold tuning | 0.40 | **0.3357** | **0.3203** | 0.5312 | 0.2780 | 0.4472 |
| B4: threshold tuning only | 0.40 | 0.3329 | 0.2994 | **0.5833** | 0.2598 | **0.4547** |

Decision after Stage B:

- Carry forward B4 and B5.
- Treat LOGO as the primary selection criterion.
- Treat SMOTE and random oversampling as tested but not selected.

## Stage C: Representations

Tested representations:

| Stage | Representation |
| --- | --- |
| C1 | Morgan only |
| C2 | RDKit descriptors only |
| C3 | Morgan + RDKit |
| C4 | eos2lm8 |
| C5 | ChemBERTa |
| C6 | Morgan + eos2lm8 |
| C7 | Morgan + ChemBERTa |
| C8 | Morgan + RDKit + eos2lm8 |
| C9 | Morgan + RDKit + ChemBERTa |

UniMAP was considered but skipped because the available repository/checkpoint
route was no longer usable.

Best LOGO ranking result for each representation:

| Representation | Best setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1: Morgan only | B5 | 0.30 | **0.4858** | 0.3381 | **0.3451** | 0.3926 | **0.4130** | **0.8426** |
| C3: Morgan + RDKit | B4 | 0.40 | 0.4547 | 0.3329 | 0.2994 | 0.5833 | 0.2598 | 0.8055 |
| C8: Morgan + RDKit + eos2lm8 | B4 | 0.45 | 0.4380 | 0.3476 | 0.3216 | 0.5833 | 0.2702 | 0.8101 |
| C6: Morgan + eos2lm8 | B5 | 0.40 | 0.4373 | 0.3528 | 0.3220 | **0.6042** | 0.2702 | 0.8152 |
| C4: eos2lm8 only | B4 | 0.45 | 0.4369 | 0.3476 | 0.3216 | 0.5833 | 0.2702 | 0.8025 |
| C7: Morgan + ChemBERTa | B4 | 0.40 | 0.4362 | **0.3709** | 0.3429 | 0.5833 | 0.3171 | 0.8152 |
| C9: Morgan + RDKit + ChemBERTa | B5 | 0.35 | 0.4295 | 0.3433 | 0.3303 | 0.5281 | 0.2988 | 0.8117 |
| C2: RDKit descriptors only | B4 | 0.45 | 0.4162 | 0.2909 | 0.2533 | 0.5521 | 0.2160 | 0.7860 |
| C5: ChemBERTa only | B5 | 0.35 | 0.3956 | 0.3385 | 0.3211 | 0.5354 | 0.2884 | 0.7909 |

Main Stage C conclusions:

- Morgan-only was the best representation for LOGO ranking.
- Morgan + RDKit was second-best but did not beat Morgan-only.
- This differs from the original paper, where combining fingerprint and
  descriptor features performed better.
- RDKit descriptors alone were weak.
- eos2lm8 contained useful signal, but did not improve over Morgan-only.
- ChemBERTa alone was weak for ranking.
- Morgan + ChemBERTa gave the best threshold-dependent MCC.

Stage C decision:

- Ranking: carry forward C1 + B5.
- Thresholded decision: carry forward C7 + B4.

## Stage E: Model Comparison

Stage E compared models on the two Stage C carry-forward setups:

- Ranking setup: C1 Morgan only + B5
- Thresholded-decision setup: C7 Morgan + ChemBERTa + B4

Tested models:

- Random Forest
- ExtraTrees
- Logistic Regression
- Kernel SVM
- HistGradientBoosting
- XGBoost
- LightGBM

Best LOGO ranking result for each model on C1 + B5:

| Model | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.30 | **0.4858** | 0.3381 | 0.3451 | 0.3926 | **0.4130** | **0.8426** |
| XGBoost | 0.85 | 0.4829 | 0.3084 | 0.2823 | 0.5438 | 0.2494 | 0.8242 |
| Logistic Regression | 0.85 | 0.4789 | **0.3602** | **0.3582** | 0.4740 | 0.3661 | 0.8274 |
| ExtraTrees | 0.35 | 0.4623 | 0.3285 | 0.3386 | 0.3830 | 0.4040 | 0.8253 |
| LightGBM | 0.85 | 0.4623 | 0.3193 | 0.2919 | **0.5677** | 0.2464 | 0.8138 |
| Kernel SVM | 0.35 | 0.4620 | 0.3536 | 0.3459 | 0.5156 | 0.3156 | 0.8208 |
| HistGradientBoosting | 0.85 | 0.4445 | 0.2722 | 0.2546 | 0.4479 | 0.2393 | 0.8085 |

Best LOGO thresholded-decision result for each model on C7 + B4:

| Model | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.40 | 0.4362 | **0.3709** | **0.3429** | 0.5833 | **0.3171** | 0.8152 |
| Logistic Regression | 0.40 | **0.4754** | 0.3691 | 0.3366 | **0.6250** | 0.2780 | **0.8332** |
| Kernel SVM | 0.30 | 0.4391 | 0.3691 | 0.3366 | **0.6250** | 0.2780 | 0.8167 |
| ExtraTrees | 0.35 | 0.4334 | 0.3466 | 0.3201 | 0.5573 | 0.2962 | 0.8134 |
| LightGBM | 0.05 | 0.3954 | 0.3046 | 0.3029 | 0.4323 | 0.2947 | 0.8105 |
| XGBoost | 0.20 | 0.4102 | 0.3003 | 0.2922 | 0.4490 | 0.2884 | 0.8224 |
| HistGradientBoosting | 0.05 | 0.3902 | 0.2702 | 0.2854 | 0.3609 | 0.3030 | 0.8173 |

Main Stage E conclusions:

- Random Forest remains the best ranking/discovery model.
- XGBoost came close in AUPRC but did not beat Random Forest.
- Logistic Regression was the best balanced alternative on C1 because it had
  higher MCC/F1 but slightly lower AUPRC.
- Kernel SVM was competitive but did not clearly win.
- For thresholded decisions on C7, Random Forest had the highest MCC, but
  Logistic Regression had nearly the same MCC with better AUPRC, AUROC, and
  precision.

Stage E decision:

- Final ranking model: C1 Morgan only + B5 + Random Forest.
- Final conservative binary-decision model: C7 Morgan + ChemBERTa + B4 +
  Logistic Regression.

## Work Not Done

The following ideas were considered but not included in the main ablation
conclusion.

| Item | Status | Reason |
| --- | --- | --- |
| SMOTE-NC | Not run | The dataset is mostly binary/continuous molecular features, not true mixed categorical + continuous data. Regular SMOTE already failed to improve LOGO. |
| UniMAP | Skipped | The available repository/checkpoint route was expired or unusable. |
| 3D descriptors | Not run | RDKit 3D descriptors require conformer generation and optimization. This is feasible but can introduce failures/noise. |
| Molecular dynamics features | Not run | Too large in scope for this implementation. |
| Graph neural networks | Not run | Dataset is small, pair-GNN setup is more complex, and overfitting risk is high. Better as future work. |
| Probability calibration | Not run | Useful if calibrated probabilities matter, but not required for ranking comparison. |
| External validation | Not run | No independent external test set was available. |

## Future Possibilities

Reasonable next extensions:

- Add a calibration stage for the selected final models.
- Run the final model workflow to create refreshed screening outputs:
  `scripts/train_final_model.py`, then `scripts/run_final_inference.py`.
- Try RDKit 3D descriptors as a small appendix:
  - Morgan only + 3D descriptors
  - Morgan + ChemBERTa + 3D descriptors
  - Morgan + RDKit + 3D descriptors
- Try pretrained graph embeddings before attempting a full GNN.
- Revisit UniMAP if a stable repository/checkpoint becomes available.
- Add bootstrap confidence intervals for the final LOGO AUPRC/MCC comparisons.

## Files Produced

Main result directories:

- `results/imbalance/`
- `results/representations/`
- `results/stage_e_models/`
- `results/final_model/` after running final training
- `results/final_inference/` after running final inference

Embedding/intermediate files:

- `data/processed/stage_c_molecules.csv`
- `data/processed/eos2lm8_input.csv`
- `data/processed/eos2lm8_raw_output.csv`
- `data/processed/eos2lm8_embeddings.csv`
- `data/processed/chemberta_embeddings.csv`

New scripts used for the ablation:

- `scripts/compare_imbalance_methods.py`
- `scripts/compare_representations.py`
- `scripts/make_embedding_input.py`
- `scripts/normalize_embedding_table.py`
- `scripts/generate_chemberta_embeddings.py`
- `scripts/compare_stage_e_models.py`
- `scripts/train_final_model.py`
- `scripts/run_final_inference.py`
