import os
import pandas as pd
from rdkit import Chem

def sanitize_smiles(smiles):
    """清洗并验证SMILES有效性"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)  # 规范化SMILES
    except:
        return None

def process_subset(subset_name):
    """处理单个子集数据"""
    if data_type == "train":
        raw_path = f"./data/base_train/{subset_name}/{subset_name}.csv"
        save_path = f"./data/base_train/{subset_name}/{subset_name}_clean.csv"        
    elif data_type == "test":
        raw_path = f"./data/{subset_name}/{subset_name}.csv"
        save_path = f"./data/{subset_name}/{subset_name}_clean.csv"        
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.read_csv(raw_path)
    
    # 清洗SMILES
    df["clean_smiles"] = df["smiles"].apply(sanitize_smiles)
    clean_df = df.dropna(subset=["clean_smiles"])
    
    # 检查是否有label列，如果没有则添加默认值0
    has_label = "label" in clean_df.columns
    if not has_label:
        clean_df["label"] = 0
    
    # 保存结果
    clean_df.to_csv(save_path, index=False)
    print(f"{subset_name} processed: {len(clean_df)} molecules -> {save_path}")

if __name__ == "__main__":
    # 处理训练子集
    data_type = "test"
    subsets = ["QSIDB"]
    for subset in subsets:
        process_subset(subset)
