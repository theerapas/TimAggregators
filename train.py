import os
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

RDLogger.DisableLog("rdApp.*")

# Config
DATA_DIR = "data"
SCREENING_FILE = os.path.join(DATA_DIR, "screening_data.tsv")
DRUGS_FILE = os.path.join(DATA_DIR, "selected_drugs_smiles.tsv")
EXCIPIENTS_FILE = os.path.join(DATA_DIR, "selected_excipients_smiles.tsv")

MODEL_OUTPUT = "nanoparticle_rf_model.pkl"

N_ESTIMATORS = 500
RANDOM_STATE = 42
BEST_THRESHOLD = 0.2
# 0.2 gives the best F1 score, better for discovery
# the paper uses 0.5; this model can also be compared there if needed

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
print("Extracting features...")

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


# Build feature tables
print(f"Processing {len(drugs_smiles)} drugs...")
drug_feature_df = pd.DataFrame(
    drugs_smiles["SMILES"].apply(get_mol_features).tolist(),
    columns=[f"Drug_{name}" for name in single_feature_names],
)
drugs_df = pd.concat([drugs_smiles["NAME"], drug_feature_df], axis=1)

print(f"Processing {len(excipients_smiles)} excipients...")
excipient_feature_df = pd.DataFrame(
    excipients_smiles["SMILES"].apply(get_mol_features).tolist(),
    columns=[f"Exc_{name}" for name in single_feature_names],
)
excipients_df = pd.concat([excipients_smiles["NAME"], excipient_feature_df], axis=1)

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


# Metrics
def compute_metrics(y_true, y_pred, y_prob):
    result = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }

    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        result["AUROC"] = np.nan
        result["AUPRC"] = (
            np.nan if np.sum(y_true) == 0 else average_precision_score(y_true, y_prob)
        )
    else:
        result["AUROC"] = roc_auc_score(y_true, y_prob)
        result["AUPRC"] = average_precision_score(y_true, y_prob)

    return result


# Evaluation
def make_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_stratified_cv(X, y, n_splits=10, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  CV fold {fold}/{n_splits}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = make_model()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Fold"] = fold
        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)


def evaluate_leave_one_drug_out(X, y, groups):
    logo = LeaveOneGroupOut()
    fold_metrics = []

    total_folds = groups.nunique()

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        held_out_drug = groups.iloc[test_idx].iloc[0]
        print(f"  LOGO fold {fold}/{total_folds} - held out: {held_out_drug}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = make_model()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["HeldOutDrug"] = held_out_drug
        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)


print("\nRunning 10-fold stratified cross-validation...")
cv_results = evaluate_stratified_cv(X, y)

print("\n10-Fold CV Mean Performance:")
print(cv_results.mean(numeric_only=True).round(4))
print("\n10-Fold CV Std:")
print(cv_results.std(numeric_only=True).round(4))

print("\nRunning leave-one-drug-out evaluation...")
groups = dataset["DRUG"]
logo_results = evaluate_leave_one_drug_out(X, y, groups)

print("\nLeave-One-Drug-Out Mean Performance:")
print(logo_results.mean(numeric_only=True).round(4))
print("\nLeave-One-Drug-Out Std:")
print(logo_results.std(numeric_only=True).round(4))


# Save evaluation outputs
os.makedirs("rf_model_results", exist_ok=True)
cv_results.to_csv("rf_model_results/cv_results_all_folds.csv", index=False)
logo_results.to_csv("rf_model_results/logo_results_all_folds.csv", index=False)

cv_summary = pd.DataFrame(
    {
        "mean": cv_results.mean(numeric_only=True),
        "std": cv_results.std(numeric_only=True),
    }
).round(4)
cv_summary.to_csv("rf_model_results/cv_results_summary.csv")

logo_summary = pd.DataFrame(
    {
        "mean": logo_results.mean(numeric_only=True),
        "std": logo_results.std(numeric_only=True),
    }
).round(4)
logo_summary.to_csv("rf_model_results/logo_results_summary.csv")


# Train final model
print("\nTraining final model on full dataset...")
rf_model = make_model()
rf_model.fit(X, y)

joblib.dump(rf_model, MODEL_OUTPUT)
print(f"Model saved as '{MODEL_OUTPUT}'")


# Threshold analysis

# def metrics_at_threshold(y_true, y_prob, threshold):
#     y_pred = (y_prob >= threshold).astype(int)
#     return {
#         "threshold": threshold,
#         "MCC": matthews_corrcoef(y_true, y_pred),
#         "F1": f1_score(y_true, y_pred, zero_division=0),
#         "Precision": precision_score(y_true, y_pred, zero_division=0),
#         "Recall": recall_score(y_true, y_pred, zero_division=0),
#         "Accuracy": accuracy_score(y_true, y_pred),
#     }
#
#
# def evaluate_thresholds_cv(X, y, thresholds, n_splits=10, random_state=RANDOM_STATE):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#
#     all_probs = []
#     all_true = []
#
#     for train_idx, test_idx in skf.split(X, y):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#
#         model = make_model()
#         model.fit(X_train, y_train)
#
#         y_prob = model.predict_proba(X_test)[:, 1]
#         all_probs.extend(y_prob)
#         all_true.extend(y_test)
#
#     all_probs = np.array(all_probs)
#     all_true = np.array(all_true)
#
#     results = []
#     for t in thresholds:
#         results.append(metrics_at_threshold(all_true, all_probs, t))
#
#     return pd.DataFrame(results)
#
#
# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
# threshold_results = evaluate_thresholds_cv(X, y, thresholds)
# print("\nThreshold sweep results:")
# print(threshold_results.round(4))
# threshold_results.to_csv("rf_model_results/threshold_sweep_results.csv", index=False)
