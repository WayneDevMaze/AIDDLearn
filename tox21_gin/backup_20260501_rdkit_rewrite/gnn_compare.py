# 导入必要的库
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# -------------------------- 1. 数据加载与预处理 --------------------------
dataset = MoleculeNet(root='data', name='Tox21')
dataset = dataset.shuffle()

# 划分训练集和测试集（8:2）
train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):]
print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

# 【关键修改1】计算训练集第一个任务的正负样本比例，用于处理类别不平衡
y_train = []
for data in train_dataset:
    y = data.y[:, 0]
    mask = ~torch.isnan(y)
    y_train.append(y[mask])
y_train = torch.cat(y_train)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # 负样本数/正样本数
print(f"正样本比例: {(y_train == 1).sum()/len(y_train):.4f}, 正样本权重: {pos_weight:.4f}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# -------------------------- 2. 模型定义（加了BatchNorm、Dropout、加深层数） --------------------------
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()
        # 【关键修改2】加深到3层，每层加BatchNorm和Dropout
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = BatchNorm1d(hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.bn3 = BatchNorm1d(hidden)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        # 第1层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        # 第2层
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        # 第3层
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 全局池化
        x = global_add_pool(x, batch)
        return self.lin(x)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, dropout=0.3):
        super().__init__()
        # 【关键修改3】GIN的MLP保持不变，但每层卷积后加BatchNorm和Dropout
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
        self.lin = Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        # 第1层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        # 第2层
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        # 第3层
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 全局池化
        x = global_add_pool(x, batch)
        return self.lin(x)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4, dropout=0.3):
        super().__init__()
        # 【关键修改4】GAT每层加BatchNorm和Dropout，注意力层也加dropout
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.bn1 = BatchNorm1d(hidden * heads)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=True, dropout=dropout)
        self.bn2 = BatchNorm1d(hidden)
        self.dropout = Dropout(dropout)
        self.lin = Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        # 第1层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        # 第2层
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        # 全局池化
        x = global_add_pool(x, batch)
        return self.lin(x)

# -------------------------- 3. 训练与测试函数 --------------------------
def train(model, loader, optimizer, pos_weight):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        
        y = data.y[:, 0].float()
        mask = ~torch.isnan(y)
        
        # 【关键修改5】损失函数加pos_weight处理类别不平衡
        loss = F.binary_cross_entropy_with_logits(
            out[:, 0][mask],
            y[mask],
            pos_weight=pos_weight.to(device)
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model, loader):
    model.eval()
    ys, preds = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, 0]
        mask = ~torch.isnan(y)
        ys.append(y[mask].cpu())
        preds.append(out[:, 0][mask].cpu())
    y = torch.cat(ys).numpy()
    pred = torch.cat(preds).numpy()
    return roc_auc_score(y, pred)

# -------------------------- 4. 模型初始化与训练 --------------------------
in_channels = dataset.num_features
print(f"输入特征维度: {in_channels}")

models = {
    "GCN": GCN(in_channels).to(device),
    "GIN": GIN(in_channels).to(device),
    "GAT": GAT(in_channels).to(device)
}

# 训练和测试每个模型
for name, model in models.items():
    print(f"\n==== 训练 {name} ====")
    # 【关键修改6】学习率调到0.001，用AdamW加权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_auc = 0
    for epoch in range(150):  # 【关键修改7】训练轮次增加到150
        loss = train(model, train_loader, optimizer, pos_weight)
        test_auc = test(model, test_loader)
        
        # 保存最佳AUC
        if test_auc > best_auc:
            best_auc = test_auc
        
        if (epoch + 1) % 20 == 0:  # 每20轮打印一次
            print(f"{name} 第 {epoch+1:03d} 轮 | 损失 {loss:.4f} | 测试AUC {test_auc:.4f} | 最佳AUC {best_auc:.4f}")

# -------------------------- 5. 画图（加了sigmoid） --------------------------
@torch.no_grad()
def get_roc(model, loader):
    model.eval()
    y_true, y_score = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, 0]
        mask = ~torch.isnan(y)
        y_true.append(y[mask].cpu().numpy())
        # 【关键修改8】用sigmoid转成概率
        y_score.append(torch.sigmoid(out[:, 0][mask]).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)

def plot_roc(models, loader):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        fpr, tpr, auc_score = get_roc(model, loader)
        plt.plot(fpr, tpr, label=f"{name} AUC={auc_score:.3f}", linewidth=2)
    plt.plot([0,1],[0,1],'--', color='gray', linewidth=1.5)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Curve - GCN vs GIN vs GAT (Modified)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

plot_roc(models, test_loader)