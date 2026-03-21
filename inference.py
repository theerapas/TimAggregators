import csv
import math
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

# Config
MODEL_PATH = "nanoparticle_rf_model.pkl"
DATA_DIR = "data"

SELFAGG_DRUG_FILE = os.path.join(DATA_DIR, "drugbank_selfaggs_smiles.tsv")
GRAS_IIG_FILE = os.path.join(DATA_DIR, "gras_iig.tsv")
APPROVED_DRUGBANK_FILE = os.path.join(DATA_DIR, "drugbank5_approved_names_smiles.tsv")

SELECTED_DRUGS_FILE = os.path.join(DATA_DIR, "selected_drugs_smiles.tsv")
SELECTED_EXCIPIENTS_FILE = os.path.join(DATA_DIR, "selected_excipients_smiles.tsv")

OUTPUT_HITS_FILE = "predicted_nanoparticle_candidates.csv"
OUTPUT_ALL_FILE = "all_pair_scores.csv"

THRESHOLD = 0.2
DRUG_BLOCK_SIZE = 8
SAVE_ALL_SCORES = True

# Slightly safer than exact max when libraries internally cast to float32
FLOAT32_MAX = np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0)).item()
FLOAT32_MIN = -FLOAT32_MAX


# RDKit descriptor setup
DESCRIPTOR_FUNCS = [func for _, func in Descriptors._descList]
N_FP_BITS = 2048
MORGAN_RADIUS = 4


# Feature extraction
def sanitize_descriptor(value: float) -> float:
    """
    Match the spirit of the author's original code:
    - NaN -> 0.0
    - absurdly large values -> float32 max
    - absurdly small values -> -float32 max
    """
    try:
        value = float(value)
    except Exception:
        return 0.0

    if np.isnan(value):
        return 0.0
    if np.isposinf(value) or value > FLOAT32_MAX:
        return FLOAT32_MAX
    if np.isneginf(value) or value < FLOAT32_MIN:
        return FLOAT32_MIN

    return value


def describe_mol_from_smiles(smiles: str) -> List[float] | None:
    """
    Returns [Morgan FP bits] + [RDKit descriptors]
    using the same feature structure as training.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Match author code
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=N_FP_BITS)
    fp_bits = [float(bit) for bit in fp.ToBitString()]

    desc_values = []
    for func in DESCRIPTOR_FUNCS:
        try:
            raw = func(mol)
            desc_values.append(np.float32(sanitize_descriptor(raw)).item())
        except Exception:
            desc_values.append(np.float32(0.0).item())

    return fp_bits + desc_values


# Data loading
def load_training_name_set() -> set[str]:
    """
    Names used in training libraries; these should be excluded from the
    additional approved-DrugBank pool, following the author logic.
    """
    selected_drugs_df = pd.read_csv(SELECTED_DRUGS_FILE, sep="\t")
    selected_excipients_df = pd.read_csv(SELECTED_EXCIPIENTS_FILE, sep="\t")

    names = set(selected_drugs_df["NAME"].astype(str).str.strip())
    names.update(selected_excipients_df["NAME"].astype(str).str.strip())
    return names


def load_feature_table(
    path: str,
    name_col: str,
    smiles_col: str,
    sep: str = "\t",
    exclude_names: set[str] | None = None,
    desc: str = "Loading molecules",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a NAME/SMILES table and returns:
    - names: ndarray[str]
    - features: ndarray[float64]
    """
    df = pd.read_csv(path, sep=sep)

    names: List[str] = []
    features: List[List[float]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        name = str(row[name_col]).strip()
        smiles = row[smiles_col]

        if exclude_names is not None and name in exclude_names:
            continue

        feats = describe_mol_from_smiles(smiles)
        if feats is None:
            continue

        names.append(name)
        features.append(feats)

    if not features:
        return np.array([], dtype=object), np.empty((0, N_FP_BITS + len(DESCRIPTOR_FUNCS)), dtype=np.float64)

    return np.array(names, dtype=object), np.asarray(features, dtype=np.float64)


# Scoring
def score_against_pool(
    model,
    drug_names: np.ndarray,
    drug_features: np.ndarray,
    candidate_names: np.ndarray,
    candidate_features: np.ndarray,
    candidate_source: str,
    threshold: float,
    hits_writer,
    all_writer=None,
) -> Tuple[int, int]:
    """
    Scores all drug x candidate pairs in blocks.
    Writes rows directly to CSV to avoid holding everything in memory.
    Returns:
    - total_pairs_scored
    - total_hits_written
    """
    if len(drug_names) == 0 or len(candidate_names) == 0:
        return 0, 0

    total_pairs = 0
    total_hits = 0
    n_candidates = candidate_features.shape[0]

    block_iter = range(0, len(drug_names), DRUG_BLOCK_SIZE)
    block_total = math.ceil(len(drug_names) / DRUG_BLOCK_SIZE)

    for start in tqdm(block_iter, total=block_total, desc=f"Scoring vs {candidate_source}"):
        end = min(start + DRUG_BLOCK_SIZE, len(drug_names))

        drug_block_names = drug_names[start:end]
        drug_block_features = drug_features[start:end]

        d = drug_block_features.shape[0]
        c = n_candidates

        # Cartesian expansion for this block
        x_drug = np.repeat(drug_block_features, c, axis=0)
        x_cand = np.tile(candidate_features, (d, 1))
        x_block = np.hstack([x_drug, x_cand])

        # Match train.py, which fit the model with named columns
        if hasattr(model, "feature_names_in_"):
            x_block_input = pd.DataFrame(x_block, columns=model.feature_names_in_)
        else:
            x_block_input = x_block

        probs = model.predict_proba(x_block_input)[:, 1]

        total_pairs += len(probs)

        for idx, prob in enumerate(probs):
            local_drug_idx = idx // c
            cand_idx = idx % c

            drug_name = drug_block_names[local_drug_idx]
            excipient_name = candidate_names[cand_idx]
            prob = float(prob)

            if all_writer is not None:
                all_writer.writerow([drug_name, excipient_name, candidate_source, prob])

            if prob >= threshold:
                hits_writer.writerow([drug_name, excipient_name, candidate_source, prob])
                total_hits += 1

    return total_pairs, total_hits


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
)
print(f"Valid self-aggregating drugs: {len(drug_names):,}")

print("\nLoading GRAS/IIG excipients...")
gras_names, gras_features = load_feature_table(
    GRAS_IIG_FILE,
    name_col="NAME",
    smiles_col="SMILES",
    desc="GRAS/IIG molecules",
)
print(f"Valid GRAS/IIG molecules: {len(gras_names):,}")

print("\nLoading additional approved DrugBank small molecules...")
approved_names, approved_features = load_feature_table(
    APPROVED_DRUGBANK_FILE,
    name_col="NAME",
    smiles_col="SMILES",
    exclude_names=training_names,
    desc="Approved DrugBank molecules",
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