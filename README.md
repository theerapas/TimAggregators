# TimAggregators (Tim's implementation of CoAggregators)

Machine learning pipeline to predict drug–excipient co-aggregation and screen nanoparticle candidates.

Reference: [Computationally guided high-throughput design of self-assembling drug nanoparticles](https://www.nature.com/articles/s41565-021-00870-y)

` Reker et al. Nature Nanotechnology 2021`
`https://doi.org/10.1038/s41565-021-00870-y`

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
- Filter by threshold
- Save ranked candidates to `predicted_nanoparticle_candidates.csv`

## Data

Located in `data/`

- `screening_data.tsv`: labeled drug–excipient pairs

- `selected_drugs_smiles.tsv`: training drugs

- `selected_excipients_smiles.tsv`: training excipients

- `drugbank_selfaggs_smiles.tsv`: candidate drugs

- `gras_iig.tsv`: candidate excipients

## Results

#### Model perfermance results located in `model_results/`

- `cv_results_all_folds_t02.csv`: fold results (threshold 0.2)

- `cv_results_all_folds_t05.csv`: fold results (threshold 0.5)

- `cv_results_summary_t02.csv`: cross validation results summary (threshold 0.2)

- `cv_results_summary_t05.csv`: cross validation results summary (threshold 0.5)

- `logo_results_*`: leave-one-drug-out results

- `threshold_sweep_results.csv`: threshold comparison

**_Full performance analysis: `summary.md`_**

#### Inference results located in `inference_results/`

- `all_pair_scores.csv`: all ~2.1M pairs scores

- `predicted_nanoparticle_candidates.csv`: all pairs that cross the 0.2 threshold
