"""
GNN分子性质预测模型集合
基于综述文献实现从2017到2024年的代表性方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, global_add_pool, 
    global_mean_pool, global_max_pool, MessagePassing
)
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.utils import softmax


# ==================== 1. 经典方法 (2017-2019) ====================

class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, ICLR 2017)
    基础图卷积网络，使用谱图卷积的简化版本
    """
    def __init__(self, in_channels, hidden=128, dropout=0.3, num_classes=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn3 = BatchNorm1d(hidden)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_add_pool(x, batch)
        return self.lin(x)


class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., ICLR 2018)
    引入注意力机制，允许节点关注不同邻居的重要性
    """
    def __init__(self, in_channels, hidden=128, heads=4, dropout=0.3, num_classes=1):
        super().__init__()
        # 第一层：多头注意力，输出维度 = hidden * heads
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden * heads)
        # 第二层：单头注意力
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.bn2 = BatchNorm1d(hidden)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = global_add_pool(x, batch)
        return self.lin(x)


class GIN(nn.Module):
    """
    Graph Isomorphism Network (Xu et al., ICLR 2019)
    理论上与WL图同构测试一样强大，可区分不同图结构
    """
    def __init__(self, in_channels, hidden=128, dropout=0.3, num_classes=1):
        super().__init__()
        # GIN使用MLP作为消息传递函数
        nn1 = Sequential(Linear(in_channels, hidden), ReLU(), Linear(hidden, hidden))
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(hidden)
        
        nn2 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(hidden)
        
        nn3 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(hidden)
        
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = global_add_pool(x, batch)
        return self.lin(x)


# ==================== 2. 改进方法 (2019-2021) ====================

class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network (Yang et al., J. Chem. Inf. Model. 2019)
    Chemprop的核心，在边上进行消息传递而非节点
    更适合化学分子，因为化学键有方向性（从供体到受体）
    """
    def __init__(self, in_channels, hidden=128, dropout=0.3, num_classes=1):
        super().__init__()
        # 原子特征嵌入
        self.atom_encoder = Linear(in_channels, hidden)
        # 边消息传递
        self.edge_message = Sequential(
            Linear(hidden * 2, hidden),  # 连接两个节点的特征
            ReLU(),
            Linear(hidden, hidden)
        )
        # 消息更新
        self.gru = nn.GRUCell(hidden, hidden)
        # 输出层
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = x.float()
        # 初始化节点特征
        h = self.atom_encoder(x)
        
        # 消息传递（简化版，3轮）
        for _ in range(3):
            # 收集邻居消息
            row, col = edge_index
            # 边特征：连接源节点和目标节点
            edge_feat = torch.cat([h[row], h[col]], dim=-1)
            message = self.edge_message(edge_feat)
            
            # 聚合消息（按目标节点）
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, col, message)
            
            # GRU更新
            h = self.gru(aggr, h)
            h = self.dropout(h)
        
        # 全局池化
        h = global_add_pool(h, batch)
        return self.lin(h)


class EGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network (Satorras et al., ICML 2021)
    对3D坐标等变，保持分子的空间对称性
    这里使用简化的2D版本，保持核心思想
    """
    def __init__(self, in_channels, hidden=128, dropout=0.3, num_classes=1):
        super().__init__()
        self.hidden = hidden
        # 边模型：计算消息
        self.edge_mlp = Sequential(
            Linear(hidden * 2 + 1, hidden),  # +1 for distance
            ReLU(),
            Linear(hidden, hidden)
        )
        # 节点模型
        self.node_mlp = Sequential(
            Linear(hidden * 2, hidden),
            ReLU(),
            Linear(hidden, hidden)
        )
        # 初始嵌入
        self.atom_encoder = Linear(in_channels, hidden)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = x.float()
        h = self.atom_encoder(x)
        
        row, col = edge_index
        
        # 消息传递（3轮）
        for _ in range(3):
            # 计算边的平方距离（使用随机坐标模拟）
            # 实际应用中应该使用真实的3D坐标
            dist_sq = torch.ones(row.size(0), 1, device=x.device)
            
            # 边消息
            edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1)
            message = self.edge_mlp(edge_input)
            
            # 聚合
            aggr = torch.zeros_like(h)
            aggr.index_add_(0, col, message)
            
            # 更新节点
            h_new = self.node_mlp(torch.cat([h, aggr], dim=-1))
            h = h + h_new  # 残差连接
            h = self.dropout(h)
        
        h = global_add_pool(h, batch)
        return self.lin(h)


# ==================== 3. 池化变体 ====================

class GINWithAttentionPool(nn.Module):
    """
    GIN + 注意力池化
    结合GIN的表达能力与注意力机制的选择性
    """
    def __init__(self, in_channels, hidden=128, dropout=0.3, num_classes=1):
        super().__init__()
        nn1 = Sequential(Linear(in_channels, hidden), ReLU(), Linear(hidden, hidden))
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(hidden)
        
        nn2 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(hidden)
        
        nn3 = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(hidden)
        
        # 注意力池化
        self.attention = Sequential(
            Linear(hidden, hidden // 2),
            ReLU(),
            Linear(hidden // 2, 1)
        )
        
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 注意力权重
        attn = self.attention(x)
        attn = softmax(attn, batch)
        
        # 加权池化
        x = x * attn
        x = global_add_pool(x, batch)
        
        return self.lin(x)


# ==================== 4. 模型字典 ====================

def get_model(model_name, in_channels, hidden=128, dropout=0.3, num_classes=1):
    """
    获取指定名称的模型
    """
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GIN': GIN,
        'DMPNN': DMPNN,
        'EGNN': EGNN,
        'GIN-AttnPool': GINWithAttentionPool,
    }
    
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(models.keys())}")
    
    return models[model_name](in_channels, hidden, dropout, num_classes)


def get_all_models(in_channels, hidden=128, dropout=0.3, num_classes=1):
    """
    获取所有模型实例
    """
    return {
        name: get_model(name, in_channels, hidden, dropout, num_classes)
        for name in ['GCN', 'GAT', 'GIN', 'DMPNN', 'EGNN', 'GIN-AttnPool']
    }
