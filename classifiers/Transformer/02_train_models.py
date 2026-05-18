import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SmilesDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.smiles = df["clean_smiles"].tolist()
        self.labels = df["label"].tolist()
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_pretrained_model():
    """Load pre-trained model and tokenizer"""
    local_pretrained_path = ".models/pretrained/ChemBERTa-77M-MLM"
    
    if not os.path.exists(local_pretrained_path):
        raise FileNotFoundError(f"Local model directory does not exist: {local_pretrained_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(local_pretrained_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_pretrained_path,
        num_labels=2
    ).to(device)
    
    return model, tokenizer

# Training function
def train_model(train_data_path, model_save_dir, log_path, epochs=30, batch_size=32, lr=1e-5):
    os.makedirs(model_save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(train_data_path)
    print(f"Loading training data: {len(df)} samples")

    # Initialize model
    model, tokenizer = load_pretrained_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Data loader
    train_dataset = SmilesDataset(df, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    # Save model parameters
    model_params = {
        'batch_size': batch_size,
        'learning_rate': lr,
        'epochs': epochs,
        'train_samples': len(df)
    }
    with open(os.path.join(model_save_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=4)
    
    log_history = []
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        total_loss = 0
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}: Average Loss: {avg_loss:.4f}")
        
        log_history.append({
            'epoch': epoch,
            'train_loss': avg_loss
        })

    # Save training log
    log_df = pd.DataFrame(log_history)
    log_df.to_csv(log_path, index=False)

    # Save final model and tokenizer
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    
    print(f"Training completed! Model saved to {model_save_dir}")

if __name__ == "__main__":
    subsets = ["subset_01", "subset_02", "subset_03", "subset_04", "subset_05", 
               "subset_06", "subset_07", "subset_08", "subset_09", "subset_10"]
    
    for subset in subsets:
        print(f"\n===== Starting training for {subset} =====")
        train_data_path = os.path.join("data", "base_train", subset, f"{subset}_clean.csv")
        model_save_dir = os.path.join("models",  subset)
        log_path = os.path.join(model_save_dir, "train_log.csv")
        
        # Check if training data exists
        if not os.path.exists(train_data_path):
            print(f"Training file does not exist: {train_data_path}")
            continue
        
        train_model(
            train_data_path=train_data_path,
            model_save_dir=model_save_dir,
            log_path=log_path,
            epochs=30,
            batch_size=32,
            lr=1e-5 
        )
    print("\nAll datasets training completed!")