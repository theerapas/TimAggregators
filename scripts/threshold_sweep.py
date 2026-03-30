import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.timaggregators.loaders import load_data
from src.timaggregators.features import build_features
from src.timaggregators.models import make_model_builders

# Config
DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "random_forest"
N_ESTIMATORS = 500
RANDOM_STATE = 42

print("Loading data...")
screening_data, drugs_smiles, excipients_smiles = load_data(
    SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
)

dataset = build_features(drugs_smiles, excipients_smiles, screening_data)

X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"])
y = dataset["CLASS"].astype(int)

print(f"Final dataset shape for training: {X.shape}")

def make_model():
    builders = make_model_builders(random_state=RANDOM_STATE)
    return builders["RandomForest"]()

def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }

def evaluate_thresholds_cv(X, y, thresholds, n_splits=10, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_probs = []
    all_true = []

    print(f"Running {n_splits}-fold CV to collect probabilities...")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"  CV fold {fold}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = make_model()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        all_probs.extend(y_prob)
        all_true.extend(y_test)

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    results = []
    for t in thresholds:
        results.append(metrics_at_threshold(all_true, all_probs, t))

    return pd.DataFrame(results)

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"\nEvaluating thresholds: {thresholds}")
threshold_results = evaluate_thresholds_cv(X, y, thresholds)

print("\nThreshold sweep results:")
print(threshold_results.round(4))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_file = OUTPUT_DIR / "threshold_sweep_results.csv"
threshold_results.to_csv(out_file, index=False)
print(f"\nResults saved to '{out_file}'")
