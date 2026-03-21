import os
import math
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Config
MODEL_PATH = "nanoparticle_rf_model.pkl"
DATA_DIR = "data"

DRUG_FILE = os.path.join(DATA_DIR, "drugbank_selfaggs_smiles.tsv")
EXCIPIENT_FILE = os.path.join(DATA_DIR, "gras_iig.tsv")

OUTPUT_FILE = "predicted_nanoparticle_candidates.csv"

# Practical screening threshold
# Use 0.2 for discovery-oriented screening
# Use 0.5 for paper-style hard classification
THRESHOLD = 0.5

# Number of drugs processed at once to avoid huge memory use
DRUG_BLOCK_SIZE = 8

# Load model
print("Loading model...")
rf_model = joblib.load(MODEL_PATH)

# Feature setup
descriptor_funcs = [func for _, func in Descriptors._descList]
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)


def get_mol_features(smiles: str) -> np.ndarray | None:
    """
    Returns:
        2048-bit Morgan fingerprint + RDKit descriptors
    Returns None if SMILES is invalid.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = morgan_gen.GetFingerprint(mol)
    fp_features = np.array(list(fp), dtype=np.float32)

    desc_features = []
    for func in descriptor_funcs:
        try:
            v = func(mol)
            if isinstance(v, float) and math.isnan(v):
                v = 0.0
            desc_features.append(float(v))
        except Exception:
            desc_features.append(0.0)

    desc_features = np.array(desc_features, dtype=np.float32)
    return np.concatenate([fp_features, desc_features])


print("Loading candidate data...")
drugs_df = pd.read_csv(DRUG_FILE, sep="\t")
excipients_df = pd.read_csv(EXCIPIENT_FILE, sep="\t")


def find_name_column(df: pd.DataFrame, preferred=("NAME")) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    return df.columns[0]


def find_smiles_column(df: pd.DataFrame, preferred=("SMILES")) -> str:
    for col in preferred:
        if col in df.columns:
            return col
    # fallback: choose the first column whose values look most like SMILES strings
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue
        valid_count = sum(Chem.MolFromSmiles(x) is not None for x in sample)
        if valid_count >= max(3, len(sample) // 2):
            return col
    raise ValueError(f"Could not detect SMILES column in columns: {list(df.columns)}")


drug_name_col = find_name_column(drugs_df)
drug_smiles_col = find_smiles_column(drugs_df)

exc_name_col = find_name_column(excipients_df)
exc_smiles_col = find_smiles_column(excipients_df)


print(f"Calculating features for {len(drugs_df)} candidate drugs...")
drug_names = []
drug_features = []

for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df)):
    name = row[drug_name_col]
    smiles = row[drug_smiles_col]
    feats = get_mol_features(smiles)
    if feats is not None:
        drug_names.append(name)
        drug_features.append(feats)

drug_names = np.array(drug_names, dtype=object)
drug_features = np.vstack(drug_features).astype(np.float32)

print(f"Valid candidate drugs: {len(drug_names)}")

print(f"Calculating features for {len(excipients_df)} candidate excipients...")
exc_names = []
exc_features = []

for _, row in tqdm(excipients_df.iterrows(), total=len(excipients_df)):
    name = row[exc_name_col]
    smiles = row[exc_smiles_col]
    feats = get_mol_features(smiles)
    if feats is not None:
        exc_names.append(name)
        exc_features.append(feats)

exc_names = np.array(exc_names, dtype=object)
exc_features = np.vstack(exc_features).astype(np.float32)

print(f"Valid candidate excipients: {len(exc_names)}")

total_pairs = len(drug_names) * len(exc_names)
print(f"Total candidate pairs to score: {total_pairs:,}")

# Predict in memory-safe blocks
print("Running inference in blocks...")
results = []

# For each block of drugs, pair with all excipients
for start in tqdm(
    range(0, len(drug_names), DRUG_BLOCK_SIZE),
    total=math.ceil(len(drug_names) / DRUG_BLOCK_SIZE),
):
    end = min(start + DRUG_BLOCK_SIZE, len(drug_names))

    drug_block_names = drug_names[start:end]
    drug_block_feats = drug_features[start:end]  # shape: [D, F]
    D = drug_block_feats.shape[0]
    E = exc_features.shape[0]

    # Repeat drug features and tile excipient features
    # Result pair matrix shape: [D*E, 2F]
    X_drug = np.repeat(drug_block_feats, E, axis=0)
    X_exc = np.tile(exc_features, (D, 1))
    X_block = np.hstack([X_drug, X_exc]).astype(np.float32)

    probs = rf_model.predict_proba(X_block)[:, 1]
    keep_mask = probs >= THRESHOLD

    if np.any(keep_mask):
        kept_indices = np.where(keep_mask)[0]
        for idx in kept_indices:
            local_drug_idx = idx // E
            exc_idx = idx % E
            results.append(
                {
                    "DRUG": drug_block_names[local_drug_idx],
                    "EXCIPIENT": exc_names[exc_idx],
                    "PROBABILITY": float(probs[idx]),
                }
            )

# Save results
print("Saving predictions...")
results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values(by="PROBABILITY", ascending=False).reset_index(
        drop=True
    )
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(results_df):,} candidates to {OUTPUT_FILE}")
    print("\nTop 10 predictions:")
    print(results_df.head(10).to_string(index=False))
else:
    print(f"No candidates found at threshold {THRESHOLD}. Try lowering the threshold.")
