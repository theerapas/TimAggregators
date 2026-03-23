# TimAggregators (Tim's implementation of CoAggregators)

Machine learning pipeline to predict drug–excipient co-aggregation and screen nanoparticle candidates.

---

## Overview

This project reproduces and extends the computational pipeline from:

**Reker et al., Nature Nanotechnology (2021)**  
[_Computationally guided high-throughput design of self-assembling drug nanoparticles_ ](https://doi.org/10.1038/s41565-021-00870-y)

The goal is to:

- Train and evaluate models to predict co-aggregation between drugs and excipients
- Compare multiple machine learning models for both cross-validation and unseen-drug generalization
- Perform large-scale screening (~2.1 million pairs)
- Identify promising nanoparticle candidates for drug delivery

---

## Features

- Morgan fingerprint (radius=4, 2048 bits) + RDKit descriptors
- Random Forest classifier (500 trees)
- Evaluation:
  - 10-fold cross-validation
  - Leave-One-Drug-Out (LOGO)
- Large-scale inference:
  - DrugBank self-aggregating drugs
  - GRAS/IIG excipients
  - Approved DrugBank small molecules
- Screening of ~2.1M candidate pairs

---

## Setup

```bash
conda env create -f environment.yml
conda activate nanoparticle-env
```

## Run

Train model:

```bash
python train.py
```

Run inference (~2.1M pairs):

```bash
python inference.py
```

---

## Scripts

### train.py

- Load screening dataset
- Extract molecular features
- Train Random Forest model
- Run:
  - 10-fold CV
  - Leave-One-Drug-Out
- Save:
  - model (nanoparticle_rf_model.pkl)
  - evaluation results

### inference.py

- Load trained model
- Compute features for:
  - candidate drugs
  - candidate excipients
- Generate all pair combinations
- Predict probability
- Filter by threshold (default = 0.2)
- Save ranked candidates to `predicted_nanoparticle_candidates.csv`

### compare_models.py

- Run 10-fold CV and Leave-One-Drug-Out evaulations on different models
- Save the output in `compare_model_results/`

### visualize.py

- To visualize Random Forest, ExtraTrees, and Logistic Regression model with 10-Fold CV and LOGO
- Plot the model's confidence as heatmap (`heatmap_{method}_{model}.png`) and their probability distributions (`prob_dist_{method}_{model}.png`).
- Save the output in `visualize_model_results/`

---

## Data

Located in `data/`

#### Training data

- `screening_data.tsv`: labeled drug–excipient pairs
- `selected_drugs_smiles.tsv`: training drugs
- `selected_excipients_smiles.tsv`: training excipients

#### Inference data

- `drugbank_selfaggs_smiles.tsv`: candidate drugs
- `gras_iig.tsv`: GRAS/IIG excipients
- `drugbank5_approved_names_smiles.tsv`: additional approved small molecules

---

## Results

*** **_Full performance analysis is in `summary.md`_** ***

#### Random Forest model perfermance results located in `rf_model_results/`

- `cv_results_all_folds_t02.csv`: fold results (threshold 0.2)
- `cv_results_all_folds_t05.csv`: fold results (threshold 0.5)
- `cv_results_summary_t02.csv`: cross validation results summary (threshold 0.2)
- `cv_results_summary_t05.csv`: cross validation results summary (threshold 0.5)
- `logo_results_*`: leave-one-drug-out results
- `threshold_sweep_results.csv`: threshold comparison


#### Inference results located in `inference_results/`

- `all_pair_scores.csv`: all ~2.1M pairs scores
- `predicted_nanoparticle_candidates.csv`: all pairs that cross the 0.2 threshold

#### Model comparison results located in `compare_model_results/`

- `multi_model_overall_summary.csv`
- `multi_model_leaderboard.csv`

#### Visualization for each model located in `visualize_model_results/`

- `heatmap_{method}_{model}.png`
- `prob_dist_{method}_{model}.pmg`
---

## Notes

- This implementation uses chemical descriptors only (no molecular dynamics features from the original paper)
- Results may differ from the published study
- Threshold = 0.2 is used for discovery (higher recall)
- Based on current comparisons:
  - Random Forest is a strong baseline
  - ExtraTrees performs well for candidate ranking
  - Logistic Regression generalizes well in Leave-One-Drug-Out evaluation

## Disclaimer

This repository is an independent implementation inspired by:

`Reker et al., Nature Nanotechnology 2021.`

This project is not the original code from the authors.
All credit for the original methodology and dataset belongs to the authors.

Please refer to the original publication for scientific details and validation.

### Citation

If you use this work, please cite:

```
Reker, D., et al. (2021).
Computationally guided high-throughput design of self-assembling drug nanoparticles.
Nature Nanotechnology.
https://doi.org/10.1038/s41565-021-00870-y
```
