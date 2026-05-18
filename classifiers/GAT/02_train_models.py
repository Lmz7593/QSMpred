import os
import torch
from torch_geometric.loader import DataLoader
from model import MolecularGAT

def train_base_model(subset_name):
    MODEL_DIR = f"./save_models/{subset_name}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log_file = os.path.join(MODEL_DIR, "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,loss\n")
    
    train_data = torch.load(f"./data/base_train/{subset_name}/{subset_name}.pt", weights_only=False)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    model = MolecularGAT(num_layers=6).to(device)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(1,81):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch.to(device))
            targets = batch.y.to(device).view(-1, 1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Calculate loss
        avg_loss = total_loss / len(train_loader)
        print(f"[{subset_name}] Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")
    
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final.pt"))

if __name__ == "__main__":
    subsets = ["subset_01","subset_02","subset_03","subset_04","subset_05",
               "subset_06","subset_07","subset_08","subset_09","subset_10"]
    for subset in subsets:
        train_base_model(subset)