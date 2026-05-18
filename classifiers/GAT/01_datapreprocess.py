import os
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

def get_atom_features(atom):
    """Atomic Feature Extraction (14-dimensional)"""
    features = [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetTotalValence()),
        float(atom.GetFormalCharge()),
        float(atom.GetHybridization().real),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetMass() * 0.01),
        float(atom.GetNumImplicitHs()),
        float(atom.GetNumExplicitHs()),
        float(atom.GetChiralTag().real),
        float(atom.GetTotalNumHs()),
        float(atom.HasProp('_CIPCode') if atom.HasProp('_CIPCode') else 0.0),
        float(atom.GetIsotope() * 0.1)
    ]
    return features

def get_bond_features(bond):
    """Bond Feature Extraction (6-dimensional)"""
    features = [
        float(bond.GetBondTypeAsDouble()),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
        float(bond.GetStereo().real),
        float(bond.GetBondDir().real),
        float(bond.GetIsAromatic())
    ]
    return features

def smiles_to_graph(smiles, ID, label=None):  
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        Chem.SanitizeMol(mol)
    except:
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception as e:
            return None
    
    mol = Chem.AddHs(mol)
    
    node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])
        bond_feats = get_bond_features(bond)
        edge_attr.extend([bond_feats, bond_feats])
    
    graph_data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        ID=str(ID)
    )
    
    if label is not None:
        graph_data.y = torch.tensor(label, dtype=torch.float)
    
    return graph_data

def process_subset(subset_name):
    """Process a single subset of data and perform 5-fold cross-validation"""
    if data_type == "train":
        raw_root = f"./data/base_train/{subset_name}"
        save_path = f"./data/base_train/{subset_name}/{subset_name}.pt"        
    elif data_type == "test":
        raw_root = f"./data/{subset_name}"
        save_path = f"./data/{subset_name}/{subset_name}.pt"        

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.read_csv(f"{raw_root}/{subset_name}.csv")
    data_list = []
    has_label = "label" in df.columns

    for idx, row in df.iterrows():
        smiles = row['smiles']
        ID = row['ID']
        label = row['label'] if has_label else None
        
        graph_data = smiles_to_graph(smiles, ID, label)
        if graph_data is not None:
            data_list.append(graph_data)
    
    torch.save(data_list, save_path)
    print(f"{subset_name} processed: {len(data_list)} molecules -> {save_path}")

if __name__ == "__main__":
    subsets = ["QSIDB", "DrugBank", "MiMeDB"] # subsets = ["subset_01","subset_02","subset_03","subset_04","subset_05","subset_06","subset_07","subset_08","subset_09","subset_10"]
    data_type = "test"   # train
    for subset in subsets:
        process_subset(subset)