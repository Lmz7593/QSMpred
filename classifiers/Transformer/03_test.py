import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SmilesDataset(Dataset):
    """Custom dataset class (for prediction)"""
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
    """Get all trained model paths and corresponding subset names"""
    model_info = []
    for i in range(1, 11):
        subset = f"subset_{i:02d}"
        model_path = os.path.join("models", subset)
        if os.path.exists(model_path):
            model_info.append((model_path, subset))
    return model_info

def predict_and_average(dataset_name, model_info_list):
    """Use all models to predict on the specified dataset and calculate the average"""
    processed_path = os.path.join("data", dataset_name, f"{dataset_name}_clean.csv")
    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} does not exist")
        return

    try:
        df = pd.read_csv(processed_path)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return

    all_predictions = []
    for model_path, subset_name in model_info_list:
        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            model.eval()
            
            # Create data loader
            dataset = SmilesDataset(df, tokenizer)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Make predictions
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
            print(f"Model {subset_name} prediction completed")
            
        except Exception as e:
            print(f"Model {subset_name} processing failed: {e}")
            continue

    if not all_predictions:
        print(f"{dataset_name} has no valid predictions")
        return

    # Calculate average prediction probabilities
    avg_pred_proba = np.mean(all_predictions, axis=0)
    
    # Save results
    result_dir = "result"
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f"{dataset_name}.csv")
    
    result_df = pd.DataFrame({
        'ID': df['ID'],
        'y_pred_proba_avg': avg_pred_proba
    })
    result_df.to_csv(output_path, index=False)
    print(f"Save {dataset_name} results to {output_path}")

if __name__ == "__main__":
    model_info_list = get_trained_models()
    
    datasets_to_predict = ["QSIDB", "DrugBank", "MiMeDB"]
    for dataset in datasets_to_predict:
        predict_and_average(dataset, model_info_list)
    
    print("All processing completed")