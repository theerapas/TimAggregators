import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import csv
import joblib
import numpy as np
import pandas as pd

from src.timaggregators.features import get_mol_features
from src.timaggregators.inference import load_feature_table, score_against_pool

# Config
MODEL_PATH = PROJECT_ROOT / "results" / "random_forest" / "nanoparticle_rf_model.pkl"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

SELFAGG_DRUG_FILE = DATA_DIR / "drugbank_selfaggs_smiles.tsv"
GRAS_IIG_FILE = DATA_DIR / "gras_iig.tsv"
APPROVED_DRUGBANK_FILE = DATA_DIR / "drugbank5_approved_names_smiles.tsv"

SELECTED_DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
SELECTED_EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

OUT_DIR = PROJECT_ROOT / "results" / "inference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_HITS_FILE = OUT_DIR / "predicted_nanoparticle_candidates.csv"
OUTPUT_ALL_FILE = OUT_DIR / "all_pair_scores.csv"

THRESHOLD = 0.2
DRUG_BLOCK_SIZE = 8
SAVE_ALL_SCORES = True

def load_training_name_set() -> set[str]:
    selected_drugs_df = pd.read_csv(SELECTED_DRUGS_FILE, sep="\t")
    selected_excipients_df = pd.read_csv(SELECTED_EXCIPIENTS_FILE, sep="\t")

    names = set(selected_drugs_df["NAME"].astype(str).str.strip())
    names.update(selected_excipients_df["NAME"].astype(str).str.strip())
    return names

print("Loading model...")
model = joblib.load(MODEL_PATH)

print("Loading training library names to exclude from approved DrugBank pool...")
training_names = load_training_name_set()
print(f"Training names to exclude from approved DrugBank pool: {len(training_names):,}")

print("\nLoading candidate self-aggregating drugs...")
drug_names, drug_features = load_feature_table(
    SELFAGG_DRUG_FILE,
    name_col="NAME",
    smiles_col="SMILES",
    desc="Self-aggregating drugs",
    describe_func=get_mol_features,
)
print(f"Valid self-aggregating drugs: {len(drug_names):,}")

print("\nLoading GRAS/IIG excipients...")
gras_names, gras_features = load_feature_table(
    GRAS_IIG_FILE,
    name_col="NAME",
    smiles_col="SMILES",
    desc="GRAS/IIG molecules",
    describe_func=get_mol_features,
)
print(f"Valid GRAS/IIG molecules: {len(gras_names):,}")

print("\nLoading additional approved DrugBank small molecules...")
approved_names, approved_features = load_feature_table(
    APPROVED_DRUGBANK_FILE,
    name_col="NAME",
    smiles_col="SMILES",
    exclude_names=training_names,
    desc="Approved DrugBank molecules",
    describe_func=get_mol_features,
)
print(f"Valid approved DrugBank molecules after exclusion: {len(approved_names):,}")

total_pairs_expected = (
    len(drug_names) * len(gras_names)
    + len(drug_names) * len(approved_names)
)
print(f"\nTotal pairs to score: {total_pairs_expected:,}")

with open(OUTPUT_HITS_FILE, "w", newline="", encoding="utf-8") as hits_f:
    hits_writer = csv.writer(hits_f)
    hits_writer.writerow(["DRUG", "EXCIPIENT", "SOURCE", "PROBABILITY"])

    all_f = None
    all_writer = None
    if SAVE_ALL_SCORES:
        all_f = open(OUTPUT_ALL_FILE, "w", newline="", encoding="utf-8")
        all_writer = csv.writer(all_f)
        all_writer.writerow(["DRUG", "EXCIPIENT", "SOURCE", "PROBABILITY"])

    try:
        print("\nScoring against GRAS/IIG pool...")
        pairs_1, hits_1 = score_against_pool(
            model=model,
            drug_names=drug_names,
            drug_features=drug_features,
            candidate_names=gras_names,
            candidate_features=gras_features,
            candidate_source="GRAS_IIG",
            threshold=THRESHOLD,
            hits_writer=hits_writer,
            all_writer=all_writer,
            drug_block_size=DRUG_BLOCK_SIZE,
        )

        print("\nScoring against approved DrugBank pool...")
        pairs_2, hits_2 = score_against_pool(
            model=model,
            drug_names=drug_names,
            drug_features=drug_features,
            candidate_names=approved_names,
            candidate_features=approved_features,
            candidate_source="APPROVED_DRUGBANK",
            threshold=THRESHOLD,
            hits_writer=hits_writer,
            all_writer=all_writer,
            drug_block_size=DRUG_BLOCK_SIZE,
        )

    finally:
        if all_f is not None:
            all_f.flush()
            all_f.close()

# Sort hits file
hits_df = pd.read_csv(OUTPUT_HITS_FILE)
hits_df = hits_df.sort_values(
    by=["PROBABILITY", "DRUG", "EXCIPIENT"],
    ascending=[False, True, True]
)
hits_df.to_csv(OUTPUT_HITS_FILE, index=False)

print("\nDone.")
print(f"Pairs scored (GRAS/IIG): {pairs_1:,}")
print(f"Pairs scored (approved DrugBank): {pairs_2:,}")
print(f"Pairs scored (total): {pairs_1 + pairs_2:,}")
print(f"Hits saved at threshold {THRESHOLD}: {hits_1 + hits_2:,}")
print(f"Thresholded output: {OUTPUT_HITS_FILE}")
if SAVE_ALL_SCORES:
    print(f"All scores output: {OUTPUT_ALL_FILE}")