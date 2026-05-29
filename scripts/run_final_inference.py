import argparse
import csv
import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.timaggregators.features import get_mol_features
from src.timaggregators.inference import load_feature_table, score_against_pool


DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "results" / "final_model"
OUT_DIR = PROJECT_ROOT / "results" / "final_inference"

SELFAGG_DRUG_FILE = DATA_DIR / "drugbank_selfaggs_smiles.tsv"
GRAS_IIG_FILE = DATA_DIR / "gras_iig.tsv"
APPROVED_DRUGBANK_FILE = DATA_DIR / "drugbank5_approved_names_smiles.tsv"
SELECTED_DRUGS_FILE = DATA_DIR / "selected_drugs_smiles.tsv"
SELECTED_EXCIPIENTS_FILE = DATA_DIR / "selected_excipients_smiles.tsv"

DEFAULT_MODEL_PATH = MODEL_DIR / "improved_morgan_rf_model.pkl"
DEFAULT_METADATA_PATH = MODEL_DIR / "improved_morgan_rf_metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run final improved screening with the Morgan-only RF model."
    )
    parser.add_argument("--model-path", type=pathlib.Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--metadata-path", type=pathlib.Path, default=DEFAULT_METADATA_PATH
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the threshold stored in the final model metadata.",
    )
    parser.add_argument("--drug-block-size", type=int, default=8)
    parser.add_argument(
        "--no-save-all-scores",
        action="store_true",
        help="Only save thresholded hits, not the full pair score table.",
    )
    return parser.parse_args()


def load_training_name_set() -> set[str]:
    selected_drugs_df = pd.read_csv(SELECTED_DRUGS_FILE, sep="\t")
    selected_excipients_df = pd.read_csv(SELECTED_EXCIPIENTS_FILE, sep="\t")

    names = set(selected_drugs_df["NAME"].astype(str).str.strip())
    names.update(selected_excipients_df["NAME"].astype(str).str.strip())
    return names


def main():
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Final model not found: {args.model_path}. "
            "Run scripts/train_final_model.py first."
        )
    if not args.metadata_path.exists():
        raise FileNotFoundError(
            f"Final model metadata not found: {args.metadata_path}. "
            "Run scripts/train_final_model.py first."
        )

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    feature_components = tuple(metadata.get("feature_components", ["morgan"]))
    threshold = args.threshold if args.threshold is not None else metadata["threshold"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_hits_file = args.output_dir / "improved_predicted_nanoparticle_candidates.csv"
    output_all_file = args.output_dir / "improved_all_pair_scores.csv"

    print("Loading final improved model...")
    model = joblib.load(args.model_path)
    describe_func = lambda smiles: get_mol_features(
        smiles, components=feature_components
    )

    print("Loading training library names to exclude from approved DrugBank pool...")
    training_names = load_training_name_set()
    print(f"Training names to exclude from approved DrugBank pool: {len(training_names):,}")

    print("\nLoading candidate self-aggregating drugs...")
    drug_names, drug_features = load_feature_table(
        SELFAGG_DRUG_FILE,
        name_col="NAME",
        smiles_col="SMILES",
        desc="Self-aggregating drugs",
        describe_func=describe_func,
        feature_components=feature_components,
    )
    print(f"Valid self-aggregating drugs: {len(drug_names):,}")

    print("\nLoading GRAS/IIG excipients...")
    gras_names, gras_features = load_feature_table(
        GRAS_IIG_FILE,
        name_col="NAME",
        smiles_col="SMILES",
        desc="GRAS/IIG molecules",
        describe_func=describe_func,
        feature_components=feature_components,
    )
    print(f"Valid GRAS/IIG molecules: {len(gras_names):,}")

    print("\nLoading additional approved DrugBank small molecules...")
    approved_names, approved_features = load_feature_table(
        APPROVED_DRUGBANK_FILE,
        name_col="NAME",
        smiles_col="SMILES",
        exclude_names=training_names,
        desc="Approved DrugBank molecules",
        describe_func=describe_func,
        feature_components=feature_components,
    )
    print(f"Valid approved DrugBank molecules after exclusion: {len(approved_names):,}")

    total_pairs_expected = (
        len(drug_names) * len(gras_names)
        + len(drug_names) * len(approved_names)
    )
    print(f"\nTotal pairs to score: {total_pairs_expected:,}")

    save_all_scores = not args.no_save_all_scores
    with open(output_hits_file, "w", newline="", encoding="utf-8") as hits_f:
        hits_writer = csv.writer(hits_f)
        hits_writer.writerow(["DRUG", "EXCIPIENT", "SOURCE", "PROBABILITY"])

        all_f = None
        all_writer = None
        if save_all_scores:
            all_f = open(output_all_file, "w", newline="", encoding="utf-8")
            all_writer = csv.writer(all_f)
            all_writer.writerow(["DRUG", "EXCIPIENT", "SOURCE", "PROBABILITY"])

        try:
            print("\nScoring against GRAS/IIG pool...")
            pairs_1, hits_1 = score_against_pool(
                model=model,
                drug_names=drug_names,
                drug_features=drug_features,
                candidate_names=gras_names,
                candidate_features=gras_features,
                candidate_source="GRAS_IIG",
                threshold=threshold,
                hits_writer=hits_writer,
                all_writer=all_writer,
                drug_block_size=args.drug_block_size,
            )

            print("\nScoring against approved DrugBank pool...")
            pairs_2, hits_2 = score_against_pool(
                model=model,
                drug_names=drug_names,
                drug_features=drug_features,
                candidate_names=approved_names,
                candidate_features=approved_features,
                candidate_source="APPROVED_DRUGBANK",
                threshold=threshold,
                hits_writer=hits_writer,
                all_writer=all_writer,
                drug_block_size=args.drug_block_size,
            )
        finally:
            if all_f is not None:
                all_f.flush()
                all_f.close()

    hits_df = pd.read_csv(output_hits_file)
    hits_df = hits_df.sort_values(
        by=["PROBABILITY", "DRUG", "EXCIPIENT"],
        ascending=[False, True, True],
    )
    hits_df.to_csv(output_hits_file, index=False)

    print("\nDone.")
    print(f"Feature components: {feature_components}")
    print(f"Threshold: {threshold}")
    print(f"Pairs scored (GRAS/IIG): {pairs_1:,}")
    print(f"Pairs scored (approved DrugBank): {pairs_2:,}")
    print(f"Pairs scored (total): {pairs_1 + pairs_2:,}")
    print(f"Hits saved: {hits_1 + hits_2:,}")
    print(f"Thresholded output: {output_hits_file}")
    if save_all_scores:
        print(f"All scores output: {output_all_file}")


if __name__ == "__main__":
    main()
