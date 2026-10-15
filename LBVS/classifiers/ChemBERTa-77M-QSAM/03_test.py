import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class SmilesDataset(Dataset):
    """自定义数据集类（用于预测）"""
    def __init__(self, df, tokenizer, max_length=128):
        self.smiles = df["clean_smiles"].tolist()
        self.ids = df["ID"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "ID": self.ids[idx]
        }

def get_trained_models():
    """获取所有训练好的模型路径及对应的子集名称"""
    model_info = []
    for i in range(1, 11):
        subset = f"subset_{i:02d}"
        model_path = os.path.join("models", subset)
        if os.path.exists(model_path):
            model_info.append((model_path, subset))
    return model_info

def predict_and_average(dataset_name, model_info_list):
    """对指定数据集使用所有模型预测并计算平均值"""
    processed_path = os.path.join("data", dataset_name, f"{dataset_name}_clean.csv")
    if not os.path.exists(processed_path):
        print(f"错误：{processed_path} 不存在")
        return

    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        print(f"加载 {dataset_name} 失败: {e}")
        return

    all_predictions = []
    for model_path, subset_name in model_info_list:
        try:
            # 加载模型和分词器
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            model.eval()
            
            # 创建数据加载器
            dataset = SmilesDataset(df, tokenizer)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # 进行预测
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    probs = torch.softmax(outputs.logits, dim=1)
                    pos_probs = probs[:, 1].cpu().numpy()
                    predictions.extend(pos_probs)
            
            all_predictions.append(predictions)
            print(f"模型 {subset_name} 预测完成")
            
        except Exception as e:
            print(f"模型 {subset_name} 处理失败: {e}")
            continue

    if not all_predictions:
        print(f"{dataset_name} 无有效预测结果")
        return

    # 计算平均预测概率
    avg_pred_proba = np.mean(all_predictions, axis=0)
    
    # 保存结果
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"{dataset_name}.csv")
    
    result_df = pd.DataFrame({
        'ID': df['ID'],
        'y_pred_proba_avg': avg_pred_proba
    })
    result_df.to_csv(output_path, index=False)
    print(f"保存 {dataset_name} 结果至 {output_path}")

if __name__ == "__main__":
    model_info_list = get_trained_models()
    
    datasets_to_predict = ["QSIDB"]
    for dataset in datasets_to_predict:
        predict_and_average(dataset, model_info_list)
    
    print("所有处理完成")