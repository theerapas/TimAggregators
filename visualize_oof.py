import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

RDLogger.DisableLog("rdApp.*")

# Config
DATA_DIR = "data"
SCREENING_FILE = os.path.join(DATA_DIR, "screening_data.tsv")
DRUGS_FILE = os.path.join(DATA_DIR, "selected_drugs_smiles.tsv")
EXCIPIENTS_FILE = os.path.join(DATA_DIR, "selected_excipients_smiles.tsv")

OUTPUT_FIG = "screening_vs_oof_prediction_heatmaps.png"

DRUG_COL = "DRUG"
EXCIPIENT_COL = "EXCIPIENT"
LABEL_COL = "CLASS"

N_ESTIMATORS = 500
RANDOM_STATE = 42

# Load data
print("Loading data...")

screening_data = pd.read_csv(SCREENING_FILE, sep="\t")
screening_data = screening_data[["DRUG", "EXCIPIENT", "CLASS"]].dropna(subset=["CLASS"])

drugs_smiles = pd.read_csv(DRUGS_FILE, sep="\t")
excipients_smiles = pd.read_csv(EXCIPIENTS_FILE, sep="\t")

print(f"Loaded {len(screening_data)} screening records.")
print(f"Loaded {len(drugs_smiles)} selected drugs.")
print(f"Loaded {len(excipients_smiles)} selected excipients.")

# Feature extraction
descriptor_funcs = [func for _, func in Descriptors._descList]
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)

fp_feature_names = [f"fp_{i}" for i in range(2048)]
desc_feature_names = [name for name, _ in Descriptors._descList]
single_feature_names = fp_feature_names + desc_feature_names


def get_mol_features(smiles: str):
    """
    Returns:
        2048 Morgan fingerprint bits + RDKit descriptors
    """
    total_len = 2048 + len(descriptor_funcs)

    try:
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            return [np.nan] * total_len

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * total_len

        fp = morgan_gen.GetFingerprint(mol)
        fp_features = list(fp)

        desc_features = []
        for func in descriptor_funcs:
            try:
                v = float(func(mol))
                if np.isnan(v) or np.isinf(v):
                    v = 0.0
                desc_features.append(v)
            except Exception:
                desc_features.append(0.0)

        features = np.array(fp_features + desc_features, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features.tolist()

    except Exception:
        return [np.nan] * total_len


# Build molecular feature tables
print("Extracting drug features...")
drug_feature_df = pd.DataFrame(
    drugs_smiles["SMILES"].apply(get_mol_features).tolist(),
    columns=[f"Drug_{name}" for name in single_feature_names],
)
drugs_df = pd.concat([drugs_smiles["NAME"], drug_feature_df], axis=1)

print("Extracting excipient features...")
exc_feature_df = pd.DataFrame(
    excipients_smiles["SMILES"].apply(get_mol_features).tolist(),
    columns=[f"Exc_{name}" for name in single_feature_names],
)
excipients_df = pd.concat([excipients_smiles["NAME"], exc_feature_df], axis=1)

# Merge into training dataset
print("Merging features into training dataset...")

dataset = pd.merge(
    screening_data,
    drugs_df,
    left_on="DRUG",
    right_on="NAME",
    how="left",
).drop(columns=["NAME"])

dataset = pd.merge(
    dataset,
    excipients_df,
    left_on="EXCIPIENT",
    right_on="NAME",
    how="left",
).drop(columns=["NAME"])

X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"])
y = dataset["CLASS"].astype(int)

print(f"Final dataset shape for training: {X.shape}")


def make_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# Out-of-fold prediction
print(f"Running 10-fold out-of-fold prediction...")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
oof_probs = np.zeros(len(dataset), dtype=float)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"  CV fold {fold}/10")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = make_model()
    model.fit(X_train, y_train)

    fold_probs = model.predict_proba(X_test)[:, 1]
    oof_probs[test_idx] = fold_probs

dataset["OOF_PRED_CONFIDENCE_PERCENT"] = oof_probs * 100.0
dataset["SCREENING_VALUE"] = dataset[LABEL_COL].astype(float)

# Save OOF table
# dataset[[DRUG_COL, EXCIPIENT_COL, LABEL_COL, "OOF_PRED_CONFIDENCE_PERCENT"]].to_csv(
#     "oof_table.csv", index=False
# )

# Ordering
drug_order = list(drugs_smiles["NAME"])
excipient_order = list(excipients_smiles["NAME"])

drug_order = [d for d in drug_order if d in dataset[DRUG_COL].unique()]
excipient_order = [e for e in excipient_order if e in dataset[EXCIPIENT_COL].unique()]

# Pivot to matrices
screening_mat = dataset.pivot(
    index=DRUG_COL,
    columns=EXCIPIENT_COL,
    values="SCREENING_VALUE",
).reindex(index=drug_order, columns=excipient_order)

oof_pred_mat = dataset.pivot(
    index=DRUG_COL,
    columns=EXCIPIENT_COL,
    values="OOF_PRED_CONFIDENCE_PERCENT",
).reindex(index=drug_order, columns=excipient_order)

# screening_mat.to_csv("screening_heatmap_matrix.csv")
# oof_pred_mat.to_csv("oof_prediction_heatmap_matrix.csv")

# Plot
n_drugs = len(drug_order)
n_excipients = len(excipient_order)

# Make figure taller and cells closer to squares
cell_size = 0.24
fig_w = max(7, n_drugs * cell_size * 1.8)
fig_h = max(14, n_excipients * cell_size * 1.15)

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), constrained_layout=True)

# Left: actual screening
im1 = axes[0].imshow(
    screening_mat.T.values,
    cmap="Greys",
    vmin=0,
    vmax=1,
    aspect="equal",
    interpolation="nearest",
)
axes[0].set_title("High-throughput screening")
axes[0].set_xlabel("Drugs")
axes[0].set_ylabel("Excipients")
axes[0].set_xticks(np.arange(n_drugs))
axes[0].set_xticklabels(drug_order, rotation=90, fontsize=7)
axes[0].set_yticks(np.arange(n_excipients))
axes[0].set_yticklabels(excipient_order, fontsize=7)
cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.03, pad=0.02)
cbar1.set_label("Screening hit (0/1)")

# Right: out-of-fold prediction
im2 = axes[1].imshow(
    oof_pred_mat.T.values,
    cmap="Greys",
    vmin=0,
    vmax=60,  # for better contrast
    aspect="equal",
    interpolation="nearest",
)
axes[1].set_title("Out-of-fold model prediction")
axes[1].set_xlabel("Drugs")
axes[1].set_ylabel("Excipients")
axes[1].set_xticks(np.arange(n_drugs))
axes[1].set_xticklabels(drug_order, rotation=90, fontsize=7)
axes[1].set_yticks(np.arange(n_excipients))
axes[1].set_yticklabels(excipient_order, fontsize=7)
cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.03, pad=0.02)
cbar2.set_label("Confidence (%)")

plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
print(f"Saved figure to {OUTPUT_FIG}")
