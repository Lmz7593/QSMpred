import os
import pandas as pd
from rdkit import Chem

def sanitize_smiles(smiles):
    """Check if SMILES is valid, return boolean value"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol) 
    except:
        return None

def process_subset(subset_name):
    """Process a single subset of data"""
    if data_type == "train":
        raw_path = f"./data/base_train/{subset_name}/{subset_name}.csv"
        save_path = f"./data/base_train/{subset_name}/{subset_name}_clean.csv"        
    elif data_type == "test":
        raw_path = f"./data/{subset_name}/{subset_name}.csv"
        save_path = f"./data/{subset_name}/{subset_name}_clean.csv"        
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.read_csv(raw_path)
    
    # Clean SMILES
    df["clean_smiles"] = df["smiles"].apply(sanitize_smiles)
    clean_df = df.dropna(subset=["clean_smiles"])

    # Check if label column exists, if not, add default value 0
    has_label = "label" in clean_df.columns
    if not has_label:
        clean_df["label"] = 0

    # Save results
    clean_df.to_csv(save_path, index=False)
    print(f"{subset_name} processed: {len(clean_df)} molecules -> {save_path}")

if __name__ == "__main__":
    data_type = "test" # train
    subsets = ["QSIDB", "DrugBank", "MiMeDB"] # subsets = ["subset_01","subset_02","subset_03","subset_04","subset_05","subset_06","subset_07","subset_08","subset_09","subset_10"]
    for subset in subsets:
        process_subset(subset)
