# TimAggregators

Machine learning pipeline for predicting drug-excipient co-aggregation and screening possible nanoparticle candidates.

This repository has two separated parts:

1. **Baseline reproduction** - the original implementation path using Morgan fingerprints + RDKit descriptors with a Random Forest.
2. **Improvement study** - the ablation work that tests imbalance handling, molecular representations, and model families, then defines an improved final screening model.

For the main results and interpretation, see [summary.md](summary.md).

## Setup

```bash
conda env create -f environment.yml
conda activate nanoparticle-env
```

## Part 1: Baseline Reproduction

The baseline reproduces the first version of the pipeline:

- Features: Morgan fingerprint, radius 4, 2048 bits, plus RDKit descriptors
- Model: Random Forest, 500 trees
- Evaluation: 10-fold cross-validation and leave-one-drug-out validation
- Screening threshold: 0.2 in the inference script

Run baseline training and evaluation:

```bash
python scripts/train_baseline.py
```

Run baseline large-scale inference:

```bash
python scripts/run_inference.py
```

Baseline outputs:

- `results/random_forest/`
- `results/inference/`

Baseline performance summary:

- `summary.md`

## Part 2: Improvement Study

The improvement work is the ablation study. It asks which changes improve unseen-drug generalization under leave-one-drug-out validation.

Stages:

- Stage B: imbalance handling and threshold tuning
- Stage C: molecular representation comparison
- Stage E: model family comparison

Main ablation scripts:

```bash
python scripts/compare_imbalance_methods.py
python scripts/compare_representations.py --eos2lm8-file data/processed/eos2lm8_embeddings.csv --chemberta-file data/processed/chemberta_embeddings.csv
python scripts/compare_stage_e_models.py
```

Improvement outputs:

- `results/imbalance/`
- `results/representations/`
- `results/stage_e_models/`

Improvement notes and final ablation summary:

- `ablation_study.md`
- `ablation_study_summary.md`
- `Improvment Plan.txt`

## Final Improved Model

The selected final discovery model is:

- Features: Morgan fingerprint only
- Model: class-weighted Random Forest
- Threshold: 0.30 when a binary cutoff is needed
- Main LOGO result: AUPRC 0.4858, AUROC 0.8426

Train the final improved model on all labeled data:

```bash
python scripts/train_final_model.py
```

Run final improved screening:

```bash
python scripts/run_final_inference.py
```

Create final comparison visualizations:

```bash
python scripts/visualize_final_comparison.py
```

This uses LOGO predictions on the labeled high-throughput screening matrix. The large inference pool is unlabeled, so it cannot prove model quality by itself. To write optional old/new inference score summaries for candidate-list inspection:

```bash
python scripts/visualize_final_comparison.py --include-inference-diagnostics
```

Final improved outputs:

- `results/final_model/improved_morgan_rf_model.pkl`
- `results/final_model/improved_morgan_rf_metadata.json`
- `results/final_inference/improved_all_pair_scores.csv`
- `results/final_inference/improved_predicted_nanoparticle_candidates.csv`
- `results/final_validation_visualizations/`

Main validation visualization:

- `results/final_validation_visualizations/heatmap_logo_actual_old_new_style.png`
- `results/final_validation_visualizations/logo_validation_curves_old_vs_new.png`

The final inference script uses the metadata saved by `train_final_model.py`, so the feature setup and threshold stay tied to the trained model.

## Main Result

Baseline LOGO performance used Morgan + RDKit with Random Forest at threshold 0.50:

| Setup | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline RF, Morgan + RDKit | 0.4547 | 0.3066 | 0.2660 | 0.5938 | 0.2160 | 0.8055 |

Improved recommended setups:

| Use case | Setup | Threshold | AUPRC | MCC | F1 | Precision | Recall | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate ranking / discovery | Morgan only + class-weighted Random Forest | 0.30 | 0.4858 | 0.3381 | 0.3451 | 0.3926 | 0.4130 | 0.8426 |
| Conservative binary decision | Morgan + ChemBERTa + Logistic Regression | 0.40 | 0.4754 | 0.3691 | 0.3366 | 0.6250 | 0.2780 | 0.8332 |

For screening, use the candidate ranking model unless you specifically need a stricter yes/no decision.

## Folder Guide

```text
data/raw/                 Input datasets
data/processed/           Embedding files and processed molecule tables
scripts/                  Runnable training, ablation, embedding, and inference scripts
src/timaggregators/       Shared feature, loading, model, evaluation, and inference code
results/random_forest/    Baseline Random Forest results and model
results/inference/        Baseline screening output
results/imbalance/        Stage B ablation output
results/representations/  Stage C ablation output
results/stage_e_models/   Stage E ablation output
results/final_model/      Final improved model artifacts, created after training
results/final_inference/  Final improved screening output, created after inference
```

## Notes

- This is an independent implementation inspired by Reker et al., Nature Nanotechnology 2021.
- The original paper included molecular dynamics features; this repository currently uses chemical structure features and optional learned SMILES embeddings.
- UniMAP was skipped because the available repository/checkpoint route was not usable during this study.
- 3D descriptors and graph neural networks are future extensions, not part of the completed main ablation.

## Citation

```text
Reker, D., et al. (2021).
Computationally guided high-throughput design of self-assembling drug nanoparticles.
Nature Nanotechnology.
https://doi.org/10.1038/s41565-021-00870-y
```
