import csv
import math
from typing import Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from .features import get_feature_names

def load_feature_table(
    path: str,
    name_col: str,
    smiles_col: str,
    describe_func,
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

        feats = describe_func(smiles)
        if feats is None:
            continue

        names.append(name)
        features.append(feats)

    feature_names = get_feature_names()
    
    if not features:
        return np.array([], dtype=object), np.empty((0, len(feature_names)), dtype=np.float64)

    return np.array(names, dtype=object), np.asarray(features, dtype=np.float64)

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
    drug_block_size: int = 8,
) -> Tuple[int, int]:
    """
    Scores all drug x candidate pairs in blocks.
    Writes rows directly to CSV to avoid holding everything in memory.
    """
    if len(drug_names) == 0 or len(candidate_names) == 0:
        return 0, 0

    total_pairs = 0
    total_hits = 0
    n_candidates = candidate_features.shape[0]

    block_iter = range(0, len(drug_names), drug_block_size)
    block_total = math.ceil(len(drug_names) / drug_block_size)

    for start in tqdm(block_iter, total=block_total, desc=f"Scoring vs {candidate_source}"):
        end = min(start + drug_block_size, len(drug_names))

        drug_block_names = drug_names[start:end]
        drug_block_features = drug_features[start:end]

        d = drug_block_features.shape[0]
        c = n_candidates

        # Cartesian expansion for this block
        x_drug = np.repeat(drug_block_features, c, axis=0)
        x_cand = np.tile(candidate_features, (d, 1))
        x_block = np.hstack([x_drug, x_cand])

        # Match train_baseline.py, which fit the model with named columns
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
