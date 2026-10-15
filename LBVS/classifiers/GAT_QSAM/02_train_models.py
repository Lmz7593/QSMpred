# 02_train_models.py
import os
import torch
from torch_geometric.loader import DataLoader
from model import MolecularGAT

def train_base_model(subset_name):
    MODEL_DIR = f"./save_models/{subset_name}"
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 日志文件仅记录epoch和loss
    log_file = os.path.join(MODEL_DIR, "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,loss\n")
    
    # 加载全部数据用于训练，不划分验证集
    train_data = torch.load(f"./data/base_train/{subset_name}/{subset_name}.pt", weights_only=False)
    
    # 对齐batch_size为32
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    model = MolecularGAT(num_layers=6).to(device)  # 使用6层
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 使用二元交叉熵损失函数
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # 对齐训练轮次为50轮
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
        
        # 计算平均loss
        avg_loss = total_loss / len(train_loader)
        
        # 打印平均loss信息
        print(f"[{subset_name}] Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")
        
        # 日志记录平均loss
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")
    
    # 仅保存最终模型
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final.pt"))

if __name__ == "__main__":
    subsets = ["subset_01","subset_02","subset_03","subset_04","subset_05",
               "subset_06","subset_07","subset_08","subset_09","subset_10"]
    for subset in subsets:
        train_base_model(subset)