import sys
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

from src.timaggregators.loaders import load_data
from src.timaggregators.features import build_features
from src.timaggregators.models import make_model_builders
from src.timaggregators.evaluation import compute_metrics

# Config
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

MODEL_OUTPUT = PROJECT_ROOT / "results" / "random_forest" / "nanoparticle_rf_model.pkl"

N_ESTIMATORS = 500
RANDOM_STATE = 42
BEST_THRESHOLD = 0.2

print("Loading data...")
screening_data, drugs_smiles, excipients_smiles = load_data(
    SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
)

dataset = build_features(drugs_smiles, excipients_smiles, screening_data)

X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"])
y = dataset["CLASS"].astype(int)

print(f"Final dataset shape for training: {X.shape}")

def make_model():
    # Helper from models.py returns a dict of builders
    builders = make_model_builders(random_state=RANDOM_STATE)
    return builders["RandomForest"]()

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
rf_out_dir = PROJECT_ROOT / "results" / "random_forest"
rf_out_dir.mkdir(parents=True, exist_ok=True)

cv_results.to_csv(rf_out_dir / "cv_results_all_folds.csv", index=False)
logo_results.to_csv(rf_out_dir / "logo_results_all_folds.csv", index=False)

cv_summary = pd.DataFrame(
    {
        "mean": cv_results.mean(numeric_only=True),
        "std": cv_results.std(numeric_only=True),
    }
).round(4)
cv_summary.to_csv(rf_out_dir / "cv_results_summary.csv")

logo_summary = pd.DataFrame(
    {
        "mean": logo_results.mean(numeric_only=True),
        "std": logo_results.std(numeric_only=True),
    }
).round(4)
logo_summary.to_csv(rf_out_dir / "logo_results_summary.csv")

# Train final model
print("\nTraining final model on full dataset...")
rf_model = make_model()
rf_model.fit(X, y)

joblib.dump(rf_model, MODEL_OUTPUT)
print(f"Model saved as '{MODEL_OUTPUT}'")
