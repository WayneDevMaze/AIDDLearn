"""
简化版GNN实验 - 纯PyTorch实现
用于验证环境并快速测试模型结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

# ==================== 简化的图神经网络实现 ====================

class SimpleGraphConv(nn.Module):
    """简化版图卷积层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)
        
    def forward(self, x, adj):
        """
        x: 节点特征 [N, in_channels]
        adj: 邻接矩阵 [N, N]
        """
        # 消息传递: AXW
        x = self.linear(x)
        x = torch.matmul(adj, x)
        return x

class SimpleGCN(nn.Module):
    """简化版GCN"""
    def __init__(self, in_channels, hidden=64, num_classes=1, dropout=0.3):
        super().__init__()
        self.conv1 = SimpleGraphConv(in_channels, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.conv2 = SimpleGraphConv(hidden, hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.conv3 = SimpleGraphConv(hidden, hidden)
        self.bn3 = BatchNorm1d(hidden)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)
        
    def forward(self, x, adj, batch_idx=None):
        x = self.conv1(x, adj)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, adj)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, adj)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局池化（简化：平均）
        if batch_idx is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            # 按batch分组平均
            unique_batches = torch.unique(batch_idx)
            pooled = []
            for b in unique_batches:
                mask = batch_idx == b
                pooled.append(x[mask].mean(dim=0))
            x = torch.stack(pooled)
        
        return self.lin(x)

class SimpleGAT(nn.Module):
    """简化版GAT"""
    def __init__(self, in_channels, hidden=64, num_classes=1, dropout=0.3, heads=4):
        super().__init__()
        self.heads = heads
        self.hidden = hidden
        
        # 多头注意力
        self.attn_linear = Linear(in_channels, hidden * heads)
        self.attn_vector = nn.Parameter(torch.randn(heads, 2 * hidden))
        
        self.bn1 = BatchNorm1d(hidden * heads)
        self.conv2 = SimpleGraphConv(hidden * heads, hidden)
        self.bn2 = BatchNorm1d(hidden)
        
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)
        
    def forward(self, x, adj, batch_idx=None):
        N = x.size(0)
        
        # 第一层：多头注意力
        x = self.attn_linear(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 简化注意力（使用邻接矩阵）
        x = self.conv2(x, adj)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 全局池化
        if batch_idx is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            unique_batches = torch.unique(batch_idx)
            pooled = []
            for b in unique_batches:
                mask = batch_idx == b
                pooled.append(x[mask].mean(dim=0))
            x = torch.stack(pooled)
        
        return self.lin(x)

class SimpleGIN(nn.Module):
    """简化版GIN"""
    def __init__(self, in_channels, hidden=64, num_classes=1, dropout=0.3):
        super().__init__()
        # GIN的MLP
        self.mlp1 = Sequential(
            Linear(in_channels, hidden),
            ReLU(),
            Linear(hidden, hidden)
        )
        self.bn1 = BatchNorm1d(hidden)
        
        self.mlp2 = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden)
        )
        self.bn2 = BatchNorm1d(hidden)
        
        self.mlp3 = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden)
        )
        self.bn3 = BatchNorm1d(hidden)
        
        self.eps = nn.Parameter(torch.zeros(1))
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, num_classes)
        
    def forward(self, x, adj, batch_idx=None):
        # GIN消息传递: (1 + eps) * x + sum(neighbors)
        # 第一层
        neighbor_sum = torch.matmul(adj, x)
        x = (1 + self.eps) * x + neighbor_sum
        x = self.mlp1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        neighbor_sum = torch.matmul(adj, x)
        x = (1 + self.eps) * x + neighbor_sum
        x = self.mlp2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第三层
        neighbor_sum = torch.matmul(adj, x)
        x = (1 + self.eps) * x + neighbor_sum
        x = self.mlp3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局池化
        if batch_idx is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            unique_batches = torch.unique(batch_idx)
            pooled = []
            for b in unique_batches:
                mask = batch_idx == b
                pooled.append(x[mask].mean(dim=0))
            x = torch.stack(pooled)
        
        return self.lin(x)


# ==================== 生成模拟数据 ====================

def generate_synthetic_data(num_graphs=1000, num_nodes_range=(10, 30), num_features=9):
    """生成模拟的分子图数据（类似Tox21）"""
    graphs = []
    
    for i in range(num_graphs):
        num_nodes = np.random.randint(*num_nodes_range)
        
        # 节点特征
        x = torch.randn(num_nodes, num_features)
        
        # 随机生成邻接矩阵（模拟分子连接）
        # 分子通常是稀疏连接的
        adj = torch.zeros(num_nodes, num_nodes)
        for j in range(num_nodes):
            # 每个节点连接2-4个邻居
            num_neighbors = np.random.randint(2, 5)
            neighbors = np.random.choice(num_nodes, min(num_neighbors, num_nodes-1), replace=False)
            for n in neighbors:
                if n != j:
                    adj[j, n] = 1
                    adj[n, j] = 1  # 无向图
        
        # 添加自环
        adj = adj + torch.eye(num_nodes)
        
        # 归一化邻接矩阵
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        adj_norm = adj / degree
        
        # 标签（二分类，模拟毒性预测）
        # 使用节点特征的某种函数生成标签，使任务可学习
        label = (x.mean() > 0).float()
        
        graphs.append({
            'x': x,
            'adj': adj_norm,
            'y': label,
            'num_nodes': num_nodes
        })
    
    return graphs


def collate_graphs(graphs):
    """将多个图批量处理"""
    batch_x = []
    batch_adj = []
    batch_y = []
    batch_idx = []
    
    node_offset = 0
    for i, g in enumerate(graphs):
        batch_x.append(g['x'])
        batch_adj.append(g['adj'])
        batch_y.append(g['y'])
        batch_idx.extend([i] * g['num_nodes'])
        node_offset += g['num_nodes']
    
    # 创建块对角邻接矩阵
    max_nodes = max(g['num_nodes'] for g in graphs)
    total_nodes = sum(g['num_nodes'] for g in graphs)
    
    # 简化：分别返回每个图的列表
    return {
        'x': torch.cat(batch_x, dim=0),
        'adjs': [g['adj'] for g in graphs],
        'y': torch.tensor(batch_y).unsqueeze(1),
        'batch_idx': torch.tensor(batch_idx),
        'num_graphs': len(graphs)
    }


# ==================== 训练与评估 ====================

def train_model(model, train_graphs, epochs=50, lr=0.001, device='cpu'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {'loss': [], 'auc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 批量训练
        batch_size = 32
        num_batches = len(train_graphs) // batch_size
        
        for i in range(num_batches):
            batch_graphs = train_graphs[i*batch_size:(i+1)*batch_size]
            batch = collate_graphs(batch_graphs)
            
            x = batch['x'].to(device)
            batch_idx = batch['batch_idx'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            # 对每个图分别前向传播
            outputs = []
            node_start = 0
            for j, adj in enumerate(batch['adjs']):
                num_nodes = adj.size(0)
                x_graph = x[node_start:node_start+num_nodes]
                adj_graph = adj.to(device)
                
                out = model(x_graph, adj_graph)
                outputs.append(out)
                node_start += num_nodes
            
            out = torch.cat(outputs, dim=0)
            loss = F.binary_cross_entropy_with_logits(out, y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, history


def evaluate_model(model, test_graphs, device='cpu'):
    """评估模型"""
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for g in test_graphs:
            x = g['x'].to(device)
            adj = g['adj'].to(device)
            
            out = model(x, adj)
            prob = torch.sigmoid(out).cpu().item()
            
            all_probs.append(prob)
            all_labels.append(g['y'].item())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算指标
    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = (preds == all_labels).mean()
    
    return {
        'auc': auc,
        'accuracy': acc,
        'probs': all_probs,
        'labels': all_labels
    }


# ==================== 主实验 ====================

def main():
    print("="*60)
    print("简化版GNN实验 - 验证模型实现")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成数据
    print("\n生成模拟数据...")
    all_graphs = generate_synthetic_data(num_graphs=1000, num_features=9)
    
    # 划分训练集和测试集
    train_size = int(len(all_graphs) * 0.8)
    train_graphs = all_graphs[:train_size]
    test_graphs = all_graphs[train_size:]
    
    print(f"训练集: {len(train_graphs)}, 测试集: {len(test_graphs)}")
    
    # 定义模型
    models = {
        'GCN': SimpleGCN(in_channels=9, hidden=64),
        'GAT': SimpleGAT(in_channels=9, hidden=64, heads=4),
        'GIN': SimpleGIN(in_channels=9, hidden=64),
    }
    
    results = {}
    
    # 训练和评估每个模型
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"训练模型: {name}")
        print(f"{'='*60}")
        
        trained_model, history = train_model(
            model, train_graphs, epochs=50, lr=0.001, device=device
        )
        
        metrics = evaluate_model(trained_model, test_graphs, device=device)
        
        print(f"\n{name} 测试结果:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        results[name] = metrics
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['labels'], result['probs'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={result['auc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - GNN Models Comparison", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig('simple_experiment_results.png', dpi=300, bbox_inches='tight')
    print("\n结果图已保存至: simple_experiment_results.png")
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)


if __name__ == '__main__':
    main()
