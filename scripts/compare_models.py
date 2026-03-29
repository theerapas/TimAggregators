import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from src.timaggregators.loaders import load_data
from src.timaggregators.features import build_features
from src.timaggregators.models import make_model_builders
from src.timaggregators.evaluation import compute_metrics, get_probabilities

warnings.filterwarnings("ignore")

# Config
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

RANDOM_STATE = 42
THRESHOLDS = [0.2, 0.5]
N_SPLITS = 10

def evaluate_stratified_cv(X, y, model_name, builder, thresholds):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  [CV] {model_name} fold {fold}/{N_SPLITS}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = builder(y_train)
        model.fit(X_train, y_train)
        y_prob = get_probabilities(model, X_test)

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics.update(
                {
                    "Model": model_name,
                    "Fold": fold,
                    "Threshold": threshold,
                    "Evaluation": "CV",
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)

def evaluate_leave_one_drug_out(X, y, groups, model_name, builder, thresholds):
    logo = LeaveOneGroupOut()
    rows = []
    total_folds = groups.nunique()

    for fold, (train_idx, test_idx) in enumerate(
        logo.split(X, y, groups=groups), start=1
    ):
        held_out_drug = groups.iloc[test_idx].iloc[0]
        print(
            f"  [LOGO] {model_name} fold {fold}/{total_folds} - held out: {held_out_drug}"
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = builder(y_train)
        model.fit(X_train, y_train)
        y_prob = get_probabilities(model, X_test)

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics.update(
                {
                    "Model": model_name,
                    "HeldOutDrug": held_out_drug,
                    "Fold": fold,
                    "Threshold": threshold,
                    "Evaluation": "LOGO",
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)

def summarize_results(df: pd.DataFrame):
    metric_cols = ["MCC", "F1", "Precision", "Accuracy", "Recall", "AUROC", "AUPRC"]
    summary = (
        df.groupby(["Evaluation", "Model", "Threshold"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()

screening_data, drugs_smiles, excipients_smiles = load_data(
    SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
)
dataset = build_features(drugs_smiles, excipients_smiles, screening_data)

X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"]).copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
y = dataset["CLASS"].astype(int).copy()
groups = dataset["DRUG"].copy()

print(f"Final dataset shape for training: {X.shape}")

builders = make_model_builders(random_state=RANDOM_STATE)
print("\nModels to compare:")
for name in builders:
    print(f"- {name}")

cv_all = []
logo_all = []

for model_name, builder in builders.items():
    print(f"\nRunning model: {model_name}")
    cv_df = evaluate_stratified_cv(X, y, model_name, builder, THRESHOLDS)
    logo_df = evaluate_leave_one_drug_out(X, y, groups, model_name, builder, THRESHOLDS)
    cv_all.append(cv_df)
    logo_all.append(logo_df)

cv_results = pd.concat(cv_all, ignore_index=True)
logo_results = pd.concat(logo_all, ignore_index=True)
all_results = pd.concat([cv_results, logo_results], ignore_index=True)

cv_summary = summarize_results(cv_results)
logo_summary = summarize_results(logo_results)
overall_summary = summarize_results(all_results)

leaderboard = overall_summary.sort_values(
    by=["Evaluation", "Threshold", "AUPRC_mean", "AUROC_mean", "MCC_mean"],
    ascending=[True, True, False, False, False],
).reset_index(drop=True)

out_dir = PROJECT_ROOT / "results" / "compare_models"
out_dir.mkdir(parents=True, exist_ok=True)

overall_summary.to_csv(out_dir / "multi_model_overall_summary.csv", index=False)
leaderboard.to_csv(out_dir / "multi_model_leaderboard.csv", index=False)

print("\nSaved:")
print("- multi_model_overall_summary.csv")
print("- multi_model_leaderboard.csv")

print("\nTop rows of leaderboard:")
print(leaderboard.head(12).to_string(index=False))
