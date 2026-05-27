import argparse
import pathlib

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MOLECULES = PROJECT_ROOT / "data" / "processed" / "stage_c_molecules.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Normalize raw embedding output to the Stage C format: "
            "NAME plus numeric embedding columns."
        )
    )
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--molecules-file",
        type=pathlib.Path,
        default=DEFAULT_MOLECULES,
        help="NAME/SMILES table created by scripts/make_embedding_input.py.",
    )
    parser.add_argument(
        "--key-col",
        default=None,
        help="Column in the raw embedding file that already contains molecule names.",
    )
    parser.add_argument(
        "--smiles-col",
        default=None,
        help="Column in the raw embedding file containing SMILES, used to recover NAME.",
    )
    parser.add_argument(
        "--prefix",
        default="emb",
        help="Prefix for renamed embedding columns.",
    )
    return parser.parse_args()


def read_table(path: pathlib.Path):
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def attach_name(df: pd.DataFrame, args):
    if args.key_col is not None:
        if args.key_col not in df.columns:
            raise ValueError(f"{args.input} does not contain --key-col {args.key_col!r}.")
        return df.rename(columns={args.key_col: "NAME"})

    if args.smiles_col is None:
        raise ValueError("Provide either --key-col or --smiles-col.")
    if args.smiles_col not in df.columns:
        raise ValueError(
            f"{args.input} does not contain --smiles-col {args.smiles_col!r}."
        )

    molecules = read_table(args.molecules_file)[["NAME", "SMILES"]]
    return df.merge(
        molecules,
        left_on=args.smiles_col,
        right_on="SMILES",
        how="left",
    )


def main():
    args = parse_args()
    raw = read_table(args.input)
    named = attach_name(raw, args)

    if named["NAME"].isna().any():
        missing = named.loc[named["NAME"].isna()].head(10)
        raise ValueError(
            "Some embedding rows could not be mapped to NAME. "
            f"First unmatched rows:\n{missing.to_string(index=False)}"
        )

    non_feature_cols = {
        "NAME",
        "SMILES",
        args.key_col,
        args.smiles_col,
        "Source",
        "key",
        "input",
    }
    feature_cols = [
        col
        for col in named.columns
        if col not in non_feature_cols and pd.api.types.is_numeric_dtype(named[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric embedding columns were found.")

    output = named[["NAME"] + feature_cols].drop_duplicates(subset=["NAME"])
    output = output.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    output.columns = ["NAME"] + [f"{args.prefix}_{i:04d}" for i in range(len(feature_cols))]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        f"Saved {len(output)} molecules with {len(feature_cols)} embedding columns "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
