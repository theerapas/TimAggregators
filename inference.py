import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import joblib
import os
import itertools
from tqdm import tqdm

print("Loading models...")
rf_model = joblib.load('nanoparticle_rf_model.pkl')
imputer = joblib.load('nanoparticle_imputer.pkl')

data_dir = "data"
print("Loading candidate data...")
drugs_df = pd.read_csv(os.path.join(data_dir, "drugbank_selfaggs_smiles.tsv"), sep='\t')
excipients_df = pd.read_csv(os.path.join(data_dir, "gras_iig.tsv"), sep='\t')

def get_rdkit_features(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str):
        return [np.nan] * len(Descriptors._descList)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(Descriptors._descList)
        return [func(mol) for _, func in Descriptors._descList]
    except Exception:
        return [np.nan] * len(Descriptors._descList)

feature_names = [name for name, _ in Descriptors._descList]

# Only calculate features for unique smiles to save time
print(f"Calculating features for {len(drugs_df)} drugs...")
drug_feats = np.array([get_rdkit_features(s) for s in tqdm(drugs_df['SMILES'])])

print(f"Calculating features for {len(excipients_df)} excipients...")
exc_feats = np.array([get_rdkit_features(s) for s in tqdm(excipients_df['SMILES'])])

drug_names = drugs_df['NAME'].values
exc_names = excipients_df['NAME'].values

print("Running inference in batches...")
batch_size = 50000
total_pairs = len(drug_names) * len(exc_names)

top_predictions = []

def process_batch(batch_drug_indices, batch_exc_indices):
    # Construct the X matrix for this batch
    d_feats = drug_feats[batch_drug_indices]
    e_feats = exc_feats[batch_exc_indices]
    X_batch = np.hstack([d_feats, e_feats]).astype(np.float32)
    X_batch[np.isinf(X_batch)] = np.nan
    
    # Impute missing values
    X_batch_imputed = imputer.transform(X_batch)
    
    # Predict probabilities (we want class 1)
    probs = rf_model.predict_proba(X_batch_imputed)[:, 1]
    
    # Keep pairs with high probability (e.g. > 0.6)
    high_prob_mask = probs > 0.6
    
    results = []
    if np.any(high_prob_mask):
        for idx in np.where(high_prob_mask)[0]:
            d_idx = batch_drug_indices[idx]
            e_idx = batch_exc_indices[idx]
            results.append({
                'DRUG': drug_names[d_idx],
                'EXCIPIENT': exc_names[e_idx],
                'PROBABILITY': probs[idx]
            })
    return results

all_drug_indices = np.arange(len(drug_names))
all_exc_indices = np.arange(len(exc_names))

# Create pairs iterator
pair_iterator = itertools.product(all_drug_indices, all_exc_indices)

current_batch_d = []
current_batch_e = []
processed = 0

for d_idx, e_idx in tqdm(pair_iterator, total=total_pairs):
    current_batch_d.append(d_idx)
    current_batch_e.append(e_idx)
    
    if len(current_batch_d) >= batch_size:
        batch_results = process_batch(current_batch_d, current_batch_e)
        top_predictions.extend(batch_results)
        processed += len(current_batch_d)
        current_batch_d = []
        current_batch_e = []

# Process remainder
if len(current_batch_d) > 0:
    batch_results = process_batch(current_batch_d, current_batch_e)
    top_predictions.extend(batch_results)

# Save results
print("Sorting and saving predictions...")
results_df = pd.DataFrame(top_predictions)
if len(results_df) > 0:
    results_df = results_df.sort_values(by='PROBABILITY', ascending=False)
    results_df.to_csv("predicted_nanoparticle_candidates.csv", index=False)
    print(f"Saved {len(results_df)} high-probability candidates to predicted_nanoparticle_candidates.csv")
    print("\nTop 10 Predictions:")
    print(results_df.head(10))
else:
    print("No high probability predictions found.")
