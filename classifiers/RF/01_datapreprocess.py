import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warning messages
RDLogger.DisableLog('rdApp.*')

def extract_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Extract Morgan fingerprints from SMILES"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ['/' for _ in range(n_bits)]
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp).tolist()
    except:
        return ['/' for _ in range(n_bits)]

def process_subset(subset_name):
    """Process training and validation data for a single fold"""
    if data_type == "train":
        raw_path = f"./data/base_train/{subset_name}/{subset_name}.csv"
        save_path = f"./data/base_train/{subset_name}/{subset_name}_processed.csv"        
    elif data_type == "test":
        raw_path = f"./data/{subset_name}/{subset_name}.csv"
        save_path = f"./data/{subset_name}/{subset_name}_processed.csv"        
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.read_csv(raw_path)
    has_label = "label" in df.columns

    # Extract fingerprints
    df['fingerprint'] = df['smiles'].apply(extract_morgan_fingerprint)
    
    # Expand fingerprints
    fp_length = len(df['fingerprint'].iloc[0])
    fp_columns = [f'fp_{i}' for i in range(fp_length)]
    fp_df = pd.DataFrame(df['fingerprint'].tolist(), columns=fp_columns)
    
    # Merge data
    result_df = pd.concat([df[['ID']], fp_df], axis=1)
    if has_label:
        result_df['label'] = df['label']
    
    # Save training data
    result_df.to_csv(save_path, index=False)
    print(f"{subset_name} processed: {len(result_df)} molecules -> {save_path}")

if __name__ == "__main__":
    data_type = "test" # train
    subsets = ["QSIDB", "DrugBank", "MiMeDB"] # subsets = ["subset_01","subset_02","subset_03","subset_04","subset_05","subset_06","subset_07","subset_08","subset_09","subset_10"]
    for subset in subsets:
        process_subset(subset)