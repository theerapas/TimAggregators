import pandas as pd

def load_data(screening_file: str, drugs_file: str, excipients_file: str, class_col: str = "CLASS"):
    print("Loading data...")
    screening_data = pd.read_csv(screening_file, sep="\t")
    screening_data = screening_data[["DRUG", "EXCIPIENT", class_col]].dropna(
        subset=[class_col]
    )

    drugs_smiles = pd.read_csv(drugs_file, sep="\t")
    excipients_smiles = pd.read_csv(excipients_file, sep="\t")

    print(f"Loaded {len(screening_data)} screening records.")
    print(f"Loaded {len(drugs_smiles)} selected drugs.")
    print(f"Loaded {len(excipients_smiles)} selected excipients.")
    return screening_data, drugs_smiles, excipients_smiles
