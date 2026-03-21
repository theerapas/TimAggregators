import pandas as pd
import numpy as np
import math
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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

# Step 1: Data Ingestion (ETL)
print("Loading data...")
data_dir = "data"
screening_data = pd.read_csv(os.path.join(data_dir, "screening_data.tsv"), sep="\t")
# Keep only relevant columns
if "CLASS" in screening_data.columns:
    screening_data = screening_data[["DRUG", "EXCIPIENT", "CLASS"]].dropna(
        subset=["CLASS"]
    )
else:
    raise ValueError("Target column 'CLASS' missing from screening_data.tsv")

drugs_smiles = pd.read_csv(
    os.path.join(data_dir, "selected_drugs_smiles.tsv"), sep="\t"
)
excipients_smiles = pd.read_csv(
    os.path.join(data_dir, "selected_excipients_smiles.tsv"), sep="\t"
)

print(f"Loaded {len(screening_data)} screening records.")

# Step 2: Feature Extraction
print("Extracting features...")

descriptor_funcs = [func for _, func in Descriptors._descList]

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)


def get_mol_features(smiles):
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
                v = func(mol)
                if isinstance(v, float) and np.isnan(v):
                    v = 0.0
                desc_features.append(float(v))
            except Exception:
                desc_features.append(0.0)

        return fp_features + desc_features

    except Exception:
        return [np.nan] * total_len


# Get all available descriptor names
fp_feature_names = [f"fp_{i}" for i in range(2048)]
desc_feature_names = [name for name, _ in Descriptors._descList]
feature_names = fp_feature_names + desc_feature_names

# Extract features for drugs
print(f"Processing {len(drugs_smiles)} drugs...")
drugs_features = pd.DataFrame(
    drugs_smiles["SMILES"].apply(get_mol_features).tolist(),
    columns=[f"Drug_{name}" for name in feature_names],
)
drugs_df = pd.concat([drugs_smiles["NAME"], drugs_features], axis=1)

# Extract features for excipients
print(f"Processing {len(excipients_smiles)} excipients...")
excipients_features = pd.DataFrame(
    excipients_smiles["SMILES"].apply(get_mol_features).tolist(),
    columns=[f"Exc_{name}" for name in feature_names],
)
excipients_df = pd.concat([excipients_smiles["NAME"], excipients_features], axis=1)

# Step 3: Feature Concatenation
print("Merging features into training dataset...")

# Merge drug features
dataset = pd.merge(
    screening_data, drugs_df, left_on="DRUG", right_on="NAME", how="left"
)
dataset = dataset.drop("NAME", axis=1)

# Merge excipient features
dataset = pd.merge(
    dataset, excipients_df, left_on="EXCIPIENT", right_on="NAME", how="left"
)
dataset = dataset.drop("NAME", axis=1)

# Prepare features (X) and target (y)
X = dataset.drop(["DRUG", "EXCIPIENT", "CLASS"], axis=1)
y = dataset["CLASS"]
print(f"Final dataset shape for training: {X.shape}")


# Evaluations
def compute_metrics(y_true, y_pred, y_prob):
    out = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }

    unique_classes = np.unique(y_true)

    if len(unique_classes) < 2:
        out["AUROC"] = np.nan
        # AUPRC is also not meaningful when there are no positives
        if np.sum(y_true) == 0:
            out["AUPRC"] = np.nan
        else:
            out["AUPRC"] = average_precision_score(y_true, y_prob)
    else:
        out["AUROC"] = roc_auc_score(y_true, y_prob)
        out["AUPRC"] = average_precision_score(y_true, y_prob)

    return out


BEST_THRESHOLD = 0.5
# 0.2 gives the best F1 score, better for discovery
# the paper use 0.5 which this model also got the similar result
# the threshold testing result is in threshold_sweep_result.csv


def evaluate_stratified_cv(X, y, n_splits=10, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)


def evaluate_leave_one_drug_out(X, y, groups):
    logo = LeaveOneGroupOut()
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= BEST_THRESHOLD).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["HeldOutDrug"] = groups.iloc[test_idx].iloc[0]
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

cv_results.to_csv("cv_results_all_folds.csv", index=False)
logo_results.to_csv("logo_results_all_folds.csv", index=False)
cv_summary = pd.DataFrame(
    {
        "mean": cv_results.mean(numeric_only=True),
        "std": cv_results.std(numeric_only=True),
    }
).round(4)
cv_summary.to_csv("cv_results_summary.csv")
logo_summary = pd.DataFrame(
    {
        "mean": logo_results.mean(numeric_only=True),
        "std": logo_results.std(numeric_only=True),
    }
).round(4)
logo_summary.to_csv("logo_results_summary.csv")

print("\nTraining final model on full dataset...")
rf_model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

joblib.dump(rf_model, "nanoparticle_rf_model.pkl")
print("Model saved as 'nanoparticle_rf_model.pkl'")

# **** This is for threshold analysis ****
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


# def evaluate_thresholds_cv(X, y, thresholds, n_splits=10, random_state=42):
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#     all_probs = []
#     all_true = []

#     for train_idx, test_idx in skf.split(X, y):
#         X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#         y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#         model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
#         model.fit(X_train, y_train)

#         y_prob = model.predict_proba(X_test)[:, 1]

#         all_probs.extend(y_prob)
#         all_true.extend(y_test)

#     all_probs = np.array(all_probs)
#     all_true = np.array(all_true)

#     results = []
#     for t in thresholds:
#         row = metrics_at_threshold(all_true, all_probs, t)
#         results.append(row)

#     return pd.DataFrame(results)


# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
# threshold_results = evaluate_thresholds_cv(X, y, thresholds)

# print("\nThreshold sweep results:")
# print(threshold_results.round(4))

# threshold_results.to_csv("threshold_sweep_results.csv", index=False)
