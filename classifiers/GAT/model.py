import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F

class MolecularGAT(torch.nn.Module):
    def __init__(self, num_layers=6, hidden_dim=64, heads=2, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.conv1 = GATConv(14, hidden_dim, heads=heads, edge_dim=6)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        
        # Hidden layers (6 layers total depth)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # Hidden layer input-output dimension processing
        for i in range(num_layers - 2): 
            in_channels = hidden_dim * heads
            self.convs.append(
                GATConv(in_channels, hidden_dim, heads=heads, edge_dim=6)
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim * heads))
        
        # Output layer (single-head attention)
        self.conv_out = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=6)
        self.bn_out = torch.nn.BatchNorm1d(hidden_dim)
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        edge_index = data.edge_index.to(torch.long)
        edge_attr = data.edge_attr
        
        # Input layer
        x = self.conv1(data.x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.conv_out(x, edge_index, edge_attr)
        x = self.bn_out(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        return self.classifier(x)