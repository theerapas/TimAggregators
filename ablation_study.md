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

## Stage C: Representations

Stage C compares molecular representations while keeping the selected Stage B
setups fixed:

1. B4: no imbalance training + threshold tuning
2. B5: class weight + threshold tuning

The Stage C runner is:

```bash
python scripts/compare_representations.py
```

Current built-in representations:

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
| C10 | UniMAP |
| C11 | Morgan + UniMAP |
| C12 | Morgan + RDKit + UniMAP |

C1-C3 run from the existing SMILES data. C4-C9 require external embedding
tables and are skipped unless the corresponding files are provided. C10-C12
were added as optional UniMAP branches, but UniMAP was skipped for the final
Stage C comparison because the available repository/checkpoint route was no
longer usable.

Create the molecule input table for external embedding tools:

```bash
python scripts/make_embedding_input.py
```

This writes:

```text
data/processed/stage_c_molecules.csv
```

with columns:

```text
NAME,SMILES,Source
```

The external embedding output must then be normalized to one row per molecule:

```text
NAME,emb_0000,emb_0001,emb_0002,...
```

If an embedding tool outputs molecule names directly, normalize it with:

```bash
python scripts/normalize_embedding_table.py \
  --input path/to/raw_embedding_output.csv \
  --output data/processed/model_embeddings.csv \
  --key-col NAME \
  --prefix emb
```

If an embedding tool outputs SMILES but not names, normalize it with:

```bash
python scripts/normalize_embedding_table.py \
  --input path/to/raw_embedding_output.csv \
  --output data/processed/model_embeddings.csv \
  --smiles-col SMILES \
  --prefix emb
```

Run embedding-based Stage C after the embedding CSVs exist:

```bash
python scripts/compare_representations.py \
  --eos2lm8-file data/processed/eos2lm8_embeddings.csv \
  --chemberta-file data/processed/chemberta_embeddings.csv
```

For the final Stage C rerun, use:

```bash
python scripts/compare_representations.py \
  --representations C1_morgan_only C2_rdkit_descriptors_only C3_morgan_rdkit C4_eos2lm8 C5_chemberta C6_morgan_eos2lm8 C7_morgan_chemberta C8_morgan_rdkit_eos2lm8 C9_morgan_rdkit_chemberta \
  --eos2lm8-file data/processed/eos2lm8_embeddings.csv \
  --chemberta-file data/processed/chemberta_embeddings.csv
```

Use `--representations` to run a subset, for example:

```bash
python scripts/compare_representations.py \
  --representations C4_eos2lm8 C6_morgan_eos2lm8 C8_morgan_rdkit_eos2lm8 \
  --eos2lm8-file data/processed/eos2lm8_embeddings.csv
```

or:

```bash
python scripts/compare_representations.py \
  --representations C10_unimap C11_morgan_unimap C12_morgan_rdkit_unimap \
  --unimap-file data/processed/unimap_embeddings.csv
```

ChemBERTa/ChemBERTa-2 embeddings can be generated locally with:

```bash
pip install torch transformers
python scripts/generate_chemberta_embeddings.py \
  --model-name DeepChem/ChemBERTa-77M-MTR \
  --output data/processed/chemberta_embeddings.csv
```

Then run:

```bash
python scripts/compare_representations.py \
  --representations C5_chemberta C7_morgan_chemberta C9_morgan_rdkit_chemberta \
  --chemberta-file data/processed/chemberta_embeddings.csv
```

Embedding files should be CSV or TSV files with a molecule key column named
`NAME` by default, plus numeric embedding columns. Use `--embedding-key-col` if
the key column has a different name.

Results are written to `results/representations/`:

- `representation_all_folds.csv`
- `representation_summary.csv`
- `representation_leaderboard.csv`
- `representation_skipped.csv`, when embedding-dependent stages are skipped

## Stage C Partial Result: C1-C3

The first representation comparison
was run for the two original paper feature types:

- C1: Morgan fingerprint only
- C2: RDKit descriptors only
- C3: Morgan fingerprint + RDKit descriptors

This comparison is especially important because the original paper reported
that combining fingerprint and descriptor features performed better than either
feature family alone. In this reproduced pipeline, using LOGO as the primary
criterion and including the Stage B threshold-tuning setup, the same pattern
did not hold.

Best LOGO result for each representation:

| Representation | Best setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1: Morgan only | B5: class weight + threshold tuning | 0.30 | **0.4858** | 0.3381 | **0.3451** | 0.3926 | **0.4130** | **0.8426** |
| C2: RDKit descriptors only | B4: threshold tuning only | 0.45 | 0.4162 | 0.2909 | 0.2533 | 0.5521 | 0.2160 | 0.7860 |
| C3: Morgan + RDKit | B4: threshold tuning only | 0.40 | 0.4547 | 0.3329 | 0.2994 | 0.5833 | 0.2598 | 0.8055 |

The strongest interim result is that Morgan-only features currently perform
best for unseen-drug generalization. C1 has the highest LOGO AUPRC, AUROC, F1,
and recall among C1-C3. C3, the combined Morgan + RDKit representation, is
still competitive and has higher precision than the best C1 ranking setup, but
it does not improve LOGO ranking performance over Morgan alone in this run.

If the primary goal is ranking discovery candidates, the current best C1-C3
setup is:

- C1: Morgan only + B5 class weight + threshold tuning, threshold 0.30

If the primary goal is a more conservative thresholded decision with stronger
MCC and precision, the best C1-C3 operating point is:

- C1: Morgan only + B4 threshold tuning only, threshold 0.30-0.40
  - LOGO MCC = 0.3617
  - LOGO precision = 0.6042
  - LOGO recall = 0.2780

Interpretation:

- RDKit descriptors alone are weaker than Morgan fingerprints.
- Adding RDKit descriptors to Morgan fingerprints does not improve LOGO AUPRC
  in this reproduction.
- The descriptor block may be adding noise or overfitting risk in the small
  LOGO setting, even though combined features were beneficial in the original
  paper.
- C1 should be carried forward as the strongest current representation, while
  C3 should remain recorded as the direct comparison to the original paper's
  combined-feature setup.

## Stage C Partial Result: eos2lm8

The eos2lm8 representation was evaluated for:

- C4: eos2lm8 only
- C6: Morgan fingerprint + eos2lm8
- C8: Morgan fingerprint + RDKit descriptors + eos2lm8

Best LOGO result for each eos2lm8 representation:

| Representation | Best setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C4: eos2lm8 only | B4: threshold tuning only | 0.45 | 0.4369 | 0.3476 | 0.3216 | 0.5833 | 0.2702 | 0.8025 |
| C6: Morgan + eos2lm8 | B5: class weight + threshold tuning | 0.40 | 0.4373 | **0.3528** | 0.3220 | **0.6042** | 0.2702 | **0.8152** |
| C8: Morgan + RDKit + eos2lm8 | B4: threshold tuning only | 0.45 | **0.4380** | 0.3476 | 0.3216 | 0.5833 | 0.2702 | 0.8101 |

The best eos2lm8 LOGO AUPRC is C8 at 0.4380, which is slightly higher than
C4 and C6 but still lower than the previous C3 Morgan + RDKit result
of 0.4547 and lower than the C1 Morgan-only result of 0.4858. The best
eos2lm8 MCC is C4/B5 at 0.3549, slightly below the previous best C1/B4 MCC of
0.3617.

Interpretation:

- eos2lm8 contains real signal: eos2lm8-only performs better than RDKit-only
  on LOGO AUPRC from the C1-C3 run.
- Adding eos2lm8 to Morgan features does not improve LOGO AUPRC over Morgan
  alone.
- Adding both RDKit descriptors and eos2lm8 to Morgan gives the best eos2lm8
  AUPRC, but the gain over eos2lm8-only is very small.
- The current best Stage C representation remains C1: Morgan only.
- eos2lm8 should be recorded as tested but should not replace Morgan-only for
  the next stage unless later ChemBERTa or UniMAP results change the overall
  representation decision.

## Stage C Partial Result: ChemBERTa

ChemBERTa embeddings were evaluated for:

- C5: ChemBERTa only
- C7: Morgan fingerprint + ChemBERTa
- C9: Morgan fingerprint + RDKit descriptors + ChemBERTa

The latest files in `results/representations/` contain this ChemBERTa run.
They overwrite the eos2lm8 result files, so the C1-C3 and eos2lm8 values above
are kept in this document for comparison.

Best LOGO result for each ChemBERTa representation:

| Representation | Best setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C5: ChemBERTa only | B5: class weight + threshold tuning | 0.35 | 0.3956 | 0.3385 | 0.3211 | 0.5354 | 0.2884 | 0.7909 |
| C7: Morgan + ChemBERTa | B4: threshold tuning only | 0.40 | **0.4362** | **0.3709** | **0.3429** | **0.5833** | **0.3171** | **0.8152** |
| C9: Morgan + RDKit + ChemBERTa | B5: class weight + threshold tuning | 0.35 | 0.4295 | 0.3433 | 0.3303 | 0.5281 | 0.2988 | 0.8117 |

ChemBERTa alone is weaker than Morgan-only, Morgan + RDKit, and eos2lm8 on
LOGO AUPRC. However, adding ChemBERTa to Morgan fingerprints gives a useful
thresholded-decision improvement. C7 reaches LOGO MCC = 0.3709, which is higher
than the previous best C1 Morgan-only MCC of 0.3617.

The best ChemBERTa LOGO AUPRC is C7 at 0.4362. This is lower than:

- C1: Morgan only, AUPRC = 0.4858
- C3: Morgan + RDKit, AUPRC = 0.4547
- C8: Morgan + RDKit + eos2lm8, AUPRC = 0.4380

Interpretation:

- ChemBERTa contains useful complementary information for thresholded
  classification when combined with Morgan fingerprints.
- ChemBERTa does not improve candidate ranking over Morgan-only in LOGO.
- ChemBERTa-only is relatively weak for this dataset, suggesting that the
  learned SMILES embedding alone does not capture enough of the co-aggregation
  signal.
- Adding RDKit descriptors to Morgan + ChemBERTa does not help; C9 is worse
  than C7 on AUPRC, MCC, F1, recall, and AUROC.
- Current Stage C ranking conclusion remains C1: Morgan only.
- Current Stage C thresholded-decision conclusion is C7: Morgan + ChemBERTa,
  threshold 0.40, because it has the best LOGO MCC observed so far.

## Stage C Final Result

Stage C was rerun as a unified comparison across C1-C9 after eos2lm8 and
ChemBERTa embeddings were generated. UniMAP was excluded because the available
repository/checkpoint route was no longer usable.

The final evaluated representations were:

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

Best LOGO ranking result for each representation, sorted by LOGO AUPRC:

| Representation | Best setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C1: Morgan only | B5: class weight + threshold tuning | 0.30 | **0.4858** | 0.3381 | **0.3451** | 0.3926 | **0.4130** | **0.8426** |
| C3: Morgan + RDKit | B4: threshold tuning only | 0.40 | 0.4547 | 0.3329 | 0.2994 | 0.5833 | 0.2598 | 0.8055 |
| C8: Morgan + RDKit + eos2lm8 | B4: threshold tuning only | 0.45 | 0.4380 | 0.3476 | 0.3216 | 0.5833 | 0.2702 | 0.8101 |
| C6: Morgan + eos2lm8 | B5: class weight + threshold tuning | 0.40 | 0.4373 | 0.3528 | 0.3220 | **0.6042** | 0.2702 | 0.8152 |
| C4: eos2lm8 only | B4: threshold tuning only | 0.45 | 0.4369 | 0.3476 | 0.3216 | 0.5833 | 0.2702 | 0.8025 |
| C7: Morgan + ChemBERTa | B4: threshold tuning only | 0.40 | 0.4362 | **0.3709** | 0.3429 | 0.5833 | 0.3171 | 0.8152 |
| C9: Morgan + RDKit + ChemBERTa | B5: class weight + threshold tuning | 0.35 | 0.4295 | 0.3433 | 0.3303 | 0.5281 | 0.2988 | 0.8117 |
| C2: RDKit descriptors only | B4: threshold tuning only | 0.45 | 0.4162 | 0.2909 | 0.2533 | 0.5521 | 0.2160 | 0.7860 |
| C5: ChemBERTa only | B5: class weight + threshold tuning | 0.35 | 0.3956 | 0.3385 | 0.3211 | 0.5354 | 0.2884 | 0.7909 |

Top LOGO MCC result:

| Representation | Setup | Threshold | MCC | AUPRC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C7: Morgan + ChemBERTa | B4: threshold tuning only | 0.40 | **0.3709** | 0.4362 | 0.3429 | 0.5833 | 0.3171 | 0.8152 |

Final Stage C interpretation:

- Morgan fingerprints remain the strongest representation for LOGO ranking.
  C1 has the highest LOGO AUPRC and AUROC among all C1-C9 setups.
- The original-paper feature combination, C3 Morgan + RDKit, is still the
  second-best ranking representation, but it does not outperform Morgan alone
  in this reproduction.
- RDKit descriptors alone are weak, and adding RDKit descriptors to learned
  embeddings does not clearly help.
- eos2lm8 contains useful signal, but neither eos2lm8 alone nor Morgan +
  eos2lm8 improves over Morgan-only ranking.
- ChemBERTa alone is weak for ranking, but Morgan + ChemBERTa gives the best
  threshold-dependent MCC.

Decision for the next stage:

- Carry forward C1 + B5 when ranking candidate pairs is the priority.
  - Representation: Morgan only
  - Imbalance setup: class weight + threshold tuning
  - Operating threshold for best F1/AUPRC row: 0.30
- Carry forward C7 + B4 when a single binary decision threshold is the priority.
  - Representation: Morgan + ChemBERTa
  - Imbalance setup: threshold tuning only
  - Operating threshold: 0.40

For scientific reporting, the main Stage C conclusion is that learned
SMILES/embedding representations did not improve unseen-drug ranking over
Morgan fingerprints. ChemBERTa was useful mainly as a thresholded-decision
add-on to Morgan fingerprints.

## Stage E: Model Comparison

Stage E compares model families using the two feature setups selected from
Stage C:

1. Ranking setup:
   - C1: Morgan only
   - B5: class weight + threshold tuning
2. Thresholded-decision setup:
   - C7: Morgan + ChemBERTa
   - B4: threshold tuning only

The Stage E runner is:

```bash
python scripts/compare_stage_e_models.py
```

This evaluates:

| Stage | Model |
| --- | --- |
| E1 | Random Forest |
| E2 | ExtraTrees |
| E3 | Logistic Regression |
| E4 | Kernel SVM |
| E5 | HistGradientBoosting |
| E6 | XGBoost, if installed |
| E7 | LightGBM, if installed |

The default run uses:

```bash
python scripts/compare_stage_e_models.py \
  --chemberta-file data/processed/chemberta_embeddings.csv
```

For a faster smoke test:

```bash
python scripts/compare_stage_e_models.py \
  --models RandomForest KernelSVM \
  --n-splits 2 \
  --max-logo-folds 2
```

Results are written to `results/stage_e_models/`:

- `stage_e_model_all_folds.csv`
- `stage_e_model_summary.csv`
- `stage_e_model_leaderboard.csv`

Selection rule:

- Use LOGO AUPRC as the primary criterion for ranking/discovery models.
- Use LOGO MCC as the primary criterion for thresholded binary-decision models.
- Treat CV as supporting evidence only.
- If Kernel SVM has strong LOGO MCC but weak AUPRC, keep it only for
  thresholded decisions.
- If tree ensembles still dominate LOGO AUPRC, carry the best tree model into
  final inference.
