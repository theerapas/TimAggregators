import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

FP_RADIUS = 4
FP_SIZE = 2048

FLOAT32_MAX = np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0)).item()
FLOAT32_MIN = -FLOAT32_MAX
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
DEFAULT_COMPONENTS = ("morgan", "rdkit")


def normalize_components(components=None):
    if components is None:
        components = DEFAULT_COMPONENTS
    if isinstance(components, str):
        components = (components,)

    components = tuple(components)
    valid_components = {"morgan", "rdkit"}
    unknown = sorted(set(components) - valid_components)
    if unknown:
        raise ValueError(f"Unknown molecular feature component(s): {unknown}")
    return components


def get_morgan_feature_names(fp_size=FP_SIZE):
    return [f"fp_{i}" for i in range(fp_size)]


def get_rdkit_descriptor_names():
    return [name for name, _ in Descriptors._descList]


def get_feature_names(components=None):
    feature_names = []
    for component in normalize_components(components):
        if component == "morgan":
            feature_names.extend(get_morgan_feature_names())
        elif component == "rdkit":
            feature_names.extend(get_rdkit_descriptor_names())
    return feature_names

def sanitize_value(v):
    try:
        v = float(v)
    except Exception:
        return 0.0

    if np.isnan(v):
        return 0.0
    if np.isposinf(v) or v > FLOAT32_MAX:
        return FLOAT32_MAX
    if np.isneginf(v) or v < FLOAT32_MIN:
        return FLOAT32_MIN
    return v

def get_mol_features(smiles: str, radius=FP_RADIUS, fp_size=FP_SIZE, components=None):
    components = normalize_components(components)
    descriptor_funcs = [func for _, func in Descriptors._descList]
    total_len = 0
    if "morgan" in components:
        total_len += fp_size
    if "rdkit" in components:
        total_len += len(descriptor_funcs)

    try:
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            return [np.nan] * total_len

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * total_len

        feature_blocks = []
        if "morgan" in components:
            fp = MORGAN_GEN.GetFingerprint(mol)
            feature_blocks.extend(list(fp))

        if "rdkit" in components:
            desc_features = []
            for func in descriptor_funcs:
                try:
                    v = sanitize_value(func(mol))
                    desc_features.append(v)
                except Exception:
                    desc_features.append(0.0)
            feature_blocks.extend(desc_features)

        features = np.array(feature_blocks, dtype=np.float64)
        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=FLOAT32_MAX,
            neginf=FLOAT32_MIN,
        )
        features = np.clip(features, FLOAT32_MIN, FLOAT32_MAX)

        return features.tolist()

    except Exception:
        return [np.nan] * total_len

def build_features(
    drugs_smiles: pd.DataFrame,
    excipients_smiles: pd.DataFrame,
    screening_data: pd.DataFrame,
    components=None,
):
    print("Extracting features...")
    components = normalize_components(components)
    single_feature_names = get_feature_names(components)

    print(f"Processing {len(drugs_smiles)} drugs...")
    drug_feature_df = pd.DataFrame(
        drugs_smiles["SMILES"].apply(lambda smiles: get_mol_features(smiles, components=components)).tolist(),
        columns=[f"Drug_{name}" for name in single_feature_names],
    )
    if "NAME" in drugs_smiles.columns:
        drugs_df = pd.concat([drugs_smiles[["NAME"]], drug_feature_df], axis=1)
    else:
        drugs_df = pd.concat([drugs_smiles, drug_feature_df], axis=1)

    print(f"Processing {len(excipients_smiles)} excipients...")
    excipient_feature_df = pd.DataFrame(
        excipients_smiles["SMILES"].apply(lambda smiles: get_mol_features(smiles, components=components)).tolist(),
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
