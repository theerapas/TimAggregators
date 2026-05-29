import argparse
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.timaggregators.features import build_features
from src.timaggregators.loaders import load_data


DATA_DIR = PROJECT_ROOT / "data" / "raw"
SCREENING_FILE = DATA_DIR / "screening_data.tsv"
DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"
OUT_DIR = PROJECT_ROOT / "results" / "final_model"

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.30
FEATURE_COMPONENTS = ("morgan",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final improved discovery model on all labeled data."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=OUT_DIR,
        help="Directory for the final model artifact and metadata.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=FINAL_THRESHOLD,
        help="Probability threshold to store in metadata for final screening.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading labeled baseline data...")
    screening_data, drugs_smiles, excipients_smiles = load_data(
        SCREENING_FILE, DRUGS_FILE, EXCIPIENTS_FILE
    )

    print("Building final improved features: Morgan fingerprint only.")
    dataset = build_features(
        drugs_smiles,
        excipients_smiles,
        screening_data,
        components=FEATURE_COMPONENTS,
    )

    X = dataset.drop(columns=["DRUG", "EXCIPIENT", "CLASS"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = dataset["CLASS"].astype(int)

    print(f"Training rows: {len(X):,}")
    print(f"Feature columns: {X.shape[1]:,}")
    print(f"Class counts: {dict(y.value_counts().sort_index())}")

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "improved_morgan_rf_model.pkl"
    metadata_path = args.output_dir / "improved_morgan_rf_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "name": "improved_morgan_rf",
        "purpose": "final_discovery_ranking",
        "model": "RandomForestClassifier",
        "n_estimators": 500,
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        "feature_components": list(FEATURE_COMPONENTS),
        "threshold": args.threshold,
        "training_rows": int(len(X)),
        "feature_columns": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "selection_rationale": (
            "Stage E selected this as the best LOGO ranking/discovery model: "
            "Morgan-only features, class-weighted Random Forest, LOGO AUPRC 0.4858."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved final improved model:")
    print(f"- {model_path}")
    print(f"- {metadata_path}")


if __name__ == "__main__":
    main()
