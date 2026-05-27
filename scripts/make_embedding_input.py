import argparse
import pathlib

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "stage_c_molecules.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the unique molecule table used to generate Stage C embeddings."
    )
    parser.add_argument(
        "--drugs-file",
        type=pathlib.Path,
        default=DATA_DIR / "selected_drugs_smiles.tsv",
    )
    parser.add_argument(
        "--excipients-file",
        type=pathlib.Path,
        default=DATA_DIR / "selected_excipients_smiles.tsv",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    drugs = pd.read_csv(args.drugs_file, sep="\t")
    excipients = pd.read_csv(args.excipients_file, sep="\t")

    molecules = (
        pd.concat(
            [
                drugs[["NAME", "SMILES"]].assign(Source="drug"),
                excipients[["NAME", "SMILES"]].assign(Source="excipient"),
            ],
            ignore_index=True,
        )
        .dropna(subset=["NAME", "SMILES"])
        .drop_duplicates(subset=["NAME"])
        .sort_values("NAME")
        .reset_index(drop=True)
    )

    duplicated_smiles = molecules["SMILES"].duplicated(keep=False)
    if duplicated_smiles.any():
        print(
            "Warning: duplicate SMILES found for different molecule names. "
            "Stage C joins embeddings by NAME, so this is allowed."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    molecules.to_csv(args.output, index=False)
    print(f"Saved {len(molecules)} molecules to {args.output}")


if __name__ == "__main__":
    main()
