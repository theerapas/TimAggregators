import argparse
import pathlib

import numpy as np
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "stage_c_molecules.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "chemberta_embeddings.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ChemBERTa/ChemBERTa-2 SMILES embeddings for Stage C."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Molecule table with NAME and SMILES columns.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV in Stage C embedding format.",
    )
    parser.add_argument(
        "--model-name",
        default="DeepChem/ChemBERTa-77M-MTR",
        help=(
            "Hugging Face model name. Use DeepChem/ChemBERTa-77M-MTR for "
            "ChemBERTa-2, or DeepChem/ChemBERTa-77M-MLM for the MLM variant."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="How to convert token embeddings into one molecule embedding.",
    )
    return parser.parse_args()


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This script requires torch and transformers. Install them with:\n"
            "  pip install torch transformers\n"
        ) from exc

    molecules = pd.read_csv(args.input)
    required_cols = {"NAME", "SMILES"}
    missing_cols = required_cols - set(molecules.columns)
    if missing_cols:
        raise ValueError(f"{args.input} is missing columns: {sorted(missing_cols)}")

    molecules = (
        molecules[["NAME", "SMILES"]]
        .dropna(subset=["NAME", "SMILES"])
        .drop_duplicates(subset=["NAME"])
        .reset_index(drop=True)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    embeddings = []
    names = molecules["NAME"].tolist()
    smiles = molecules["SMILES"].astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(smiles), args.batch_size):
            batch_smiles = smiles[start : start + args.batch_size]
            encoded = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)

            if args.pooling == "cls":
                pooled = output.last_hidden_state[:, 0, :]
            else:
                pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])

            embeddings.append(pooled.detach().cpu().numpy())
            print(f"Embedded {min(start + args.batch_size, len(smiles))}/{len(smiles)}")

    embedding_array = np.vstack(embeddings)
    feature_cols = [
        f"chemberta_{idx:04d}" for idx in range(embedding_array.shape[1])
    ]
    output_df = pd.DataFrame(embedding_array, columns=feature_cols)
    output_df.insert(0, "NAME", names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(
        f"Saved {len(output_df)} molecules with {embedding_array.shape[1]} "
        f"ChemBERTa dimensions to {args.output}"
    )


if __name__ == "__main__":
    main()
