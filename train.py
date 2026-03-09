import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# Step 1: Data Ingestion (ETL)
print("Loading data...")
data_dir = "data"
screening_data = pd.read_csv(os.path.join(data_dir, "screening_data.tsv"), sep='\t')
# Keep only relevant columns
if 'CLASS' in screening_data.columns:
    screening_data = screening_data[['DRUG', 'EXCIPIENT', 'CLASS']].dropna(subset=['CLASS'])
else:
    raise ValueError("Target column 'CLASS' missing from screening_data.tsv")

drugs_smiles = pd.read_csv(os.path.join(data_dir, "selected_drugs_smiles.tsv"), sep='\t')
excipients_smiles = pd.read_csv(os.path.join(data_dir, "selected_excipients_smiles.tsv"), sep='\t')

print(f"Loaded {len(screening_data)} screening records.")

# Step 2: Feature Extraction
print("Extracting features using RDKit...")

def get_rdkit_features(smiles):
    """Calculates 200+ molecular properties from a SMILES string."""
    try:
        # Check if SMILES is a string and not empty or nan
        if pd.isna(smiles) or not isinstance(smiles, str):
            return [np.nan] * len(Descriptors._descList)
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [np.nan] * len(Descriptors._descList)
        
        features = []
        for name, func in Descriptors._descList:
            features.append(func(mol))
        return features
    except Exception as e:
        return [np.nan] * len(Descriptors._descList)

# Get all available RDKit descriptor names
feature_names = [name for name, _ in Descriptors._descList]

# Extract features for drugs
print(f"Processing {len(drugs_smiles)} drugs...")
drugs_features = pd.DataFrame(
    drugs_smiles['SMILES'].apply(get_rdkit_features).tolist(),
    columns=[f"Drug_{name}" for name in feature_names]
)
drugs_df = pd.concat([drugs_smiles['NAME'], drugs_features], axis=1)

# Extract features for excipients
print(f"Processing {len(excipients_smiles)} excipients...")
excipients_features = pd.DataFrame(
    excipients_smiles['SMILES'].apply(get_rdkit_features).tolist(),
    columns=[f"Exc_{name}" for name in feature_names]
)
excipients_df = pd.concat([excipients_smiles['NAME'], excipients_features], axis=1)

# Step 3: Feature Concatenation
print("Merging features into training dataset...")

# Merge drug features
dataset = pd.merge(screening_data, drugs_df, left_on='DRUG', right_on='NAME', how='left')
dataset = dataset.drop('NAME', axis=1)

# Merge excipient features
dataset = pd.merge(dataset, excipients_df, left_on='EXCIPIENT', right_on='NAME', how='left')
dataset = dataset.drop('NAME', axis=1)

# Check for missing values before training
initial_len = len(dataset)
dataset = dataset.dropna()
print(f"Dropped {initial_len - len(dataset)} rows due to RDKit feature extraction failures.")
print(f"Final dataset shape for training: {dataset.shape}")

# Prepare features (X) and target (y)
X = dataset.drop(['DRUG', 'EXCIPIENT', 'CLASS'], axis=1)
y = dataset['CLASS']

# Step 4: Model Training and Validation
print("\nTraining Random Forest model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use class_weight='balanced_subsample' to handle the severe class imbalance
rf_model = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    class_weight='balanced_subsample',
    max_depth=10,
    min_samples_split=5
)
rf_model.fit(X_train, y_train)

# Evaluate
print("Evaluating model on 20% validation split...")
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n" + "=" * 30)
print("Model Performance:")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("=" * 30 + "\n")

print("Data preparation and training complete!")

# Code to save the model for later use (optional):
# import joblib
# joblib.dump(rf_model, 'nanoparticle_rf_model.pkl')
# print("Model saved as 'nanoparticle_rf_model.pkl'")
