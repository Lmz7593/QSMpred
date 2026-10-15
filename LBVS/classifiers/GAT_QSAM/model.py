# model.py
import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class MolecularGAT(torch.nn.Module):
    def __init__(self, num_layers=6, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 输入层
        self.conv1 = GATConv(14, hidden_dim, heads=heads, edge_dim=6)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        
        # 中间层（6层总深度）
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # 中间层输入输出维度处理
        for i in range(num_layers - 2):  # 减去输入层和输出层
            # 中间层的输入维度是前一层的输出维度
            in_channels = hidden_dim * heads
            self.convs.append(
                GATConv(in_channels, hidden_dim, heads=heads, edge_dim=6)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        
        # 输出层（单头注意力）
        self.conv_out = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=6)
        self.bn_out = torch.nn.BatchNorm1d(hidden_dim)
        
        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        edge_index = data.edge_index.to(torch.long)
        edge_attr = data.edge_attr
        
        # 输入层
        x = self.conv1(data.x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 中间层
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        x = self.conv_out(x, edge_index, edge_attr)
        x = self.bn_out(x)
        x = F.relu(x)
        
        # 全局池化
        x = global_mean_pool(x, data.batch)
        
        # 分类
        return self.classifier(x)