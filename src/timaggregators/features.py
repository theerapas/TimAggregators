import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FP_RADIUS = 4
FP_SIZE = 2048

def get_feature_names():
    descriptor_funcs = [func for _, func in Descriptors._descList]
    fp_feature_names = [f"fp_{i}" for i in range(FP_SIZE)]
    desc_feature_names = [name for name, _ in Descriptors._descList]
    return fp_feature_names + desc_feature_names

def get_mol_features(smiles: str, radius=FP_RADIUS, fp_size=FP_SIZE):
    descriptor_funcs = [func for _, func in Descriptors._descList]
    total_len = fp_size + len(descriptor_funcs)
    try:
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            return [np.nan] * total_len

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * total_len

        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
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

def build_features(
    drugs_smiles: pd.DataFrame,
    excipients_smiles: pd.DataFrame,
    screening_data: pd.DataFrame,
):
    print("Extracting features...")
    single_feature_names = get_feature_names()

    print(f"Processing {len(drugs_smiles)} drugs...")
    drug_feature_df = pd.DataFrame(
        drugs_smiles["SMILES"].apply(get_mol_features).tolist(),
        columns=[f"Drug_{name}" for name in single_feature_names],
    )
    if "NAME" in drugs_smiles.columns:
        drugs_df = pd.concat([drugs_smiles[["NAME"]], drug_feature_df], axis=1)
    else:
        drugs_df = pd.concat([drugs_smiles, drug_feature_df], axis=1)

    print(f"Processing {len(excipients_smiles)} excipients...")
    excipient_feature_df = pd.DataFrame(
        excipients_smiles["SMILES"].apply(get_mol_features).tolist(),
        columns=[f"Exc_{name}" for name in single_feature_names],
    )
    if "NAME" in excipients_smiles.columns:
        excipients_df = pd.concat([excipients_smiles[["NAME"]], excipient_feature_df], axis=1)
    else:
        excipients_df = pd.concat([excipients_smiles, excipient_feature_df], axis=1)

    print("Merging features into training dataset...")
    
    # Try using 'DRUG' in screening_data and 'NAME' in drugs_df if it exists
    left_on_drug = "DRUG"
    right_on_drug = "NAME" if "NAME" in drugs_df.columns else "DRUG"
    
    dataset = pd.merge(
        screening_data,
        drugs_df,
        left_on=left_on_drug,
        right_on=right_on_drug,
        how="left",
    )
    if "NAME" in dataset.columns and right_on_drug == "NAME":
        dataset.drop(columns=["NAME"], inplace=True)

    left_on_exc = "EXCIPIENT"
    right_on_exc = "NAME" if "NAME" in excipients_df.columns else "EXCIPIENT"
    
    dataset = pd.merge(
        dataset,
        excipients_df,
        left_on=left_on_exc,
        right_on=right_on_exc,
        how="left",
    )
    if "NAME" in dataset.columns and right_on_exc == "NAME":
        dataset.drop(columns=["NAME"], inplace=True)

    # Some scripts expect CLASS others LABEL_COL. Just return dataset as is and let the caller drop columns
    return dataset
