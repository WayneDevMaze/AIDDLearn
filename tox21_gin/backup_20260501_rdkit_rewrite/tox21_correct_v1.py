#!/usr/bin/env python3
"""
Tox21 分子性质预测 — 正确实现
1. 用 MoleculeNet 加载独立分子图（非拼合大图）
2. 实现 GIN / GCN / GraphSAGE 三种 GNN + readout 池化
3. 12任务多标签二分类，NaN标签自动mask
"""
import warnings; warnings.filterwarnings('ignore')
import json, time, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GINConv, GCNConv, SAGEConv, global_mean_pool, global_add_pool
)
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 配置 ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = os.path.expanduser('~/Desktop/AIDDLearn/tox21_gin/data')
OUT_DIR   = os.path.expanduser('~/Desktop/AIDDLearn/tox21_gin')
HIDDEN    = 128
NUM_LAYERS = 3
EPOCHS     = 200
BATCH_SIZE = 128
LR         = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE   = 30

print("=" * 65)
print("Tox21 分子性质预测 — GNN 正确实现")
print("=" * 65)
print(f"Device: {DEVICE}")

# ============ 1. 数据加载 ============
print("\n[1] 加载 Tox21 数据集 (MoleculeNet)...")
dataset = MoleculeNet(root=DATA_ROOT, name='Tox21')
print(f"    分子总数: {len(dataset)}")
print(f"    节点特征维度: {dataset.num_node_features}")
print(f"    边特征维度: {dataset.num_edge_features}")
print(f"    标签任务数: {dataset.num_classes}  (12个毒性任务)")

# 检查单个样本
sample = dataset[0]
print(f"    样例: {sample.num_nodes} 节点, {sample.num_edges} 边, y={sample.y.shape}")

# 随机划分 80/10/10
n = len(dataset)
perm = torch.randperm(n)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train_idx = perm[:n_train]
val_idx   = perm[n_train:n_train + n_val]
test_idx  = perm[n_train + n_val:]

train_ds = dataset[train_idx.tolist()]
val_ds   = dataset[val_idx.tolist()]
test_ds  = dataset[test_idx.tolist()]

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,   batch_size=BATCH_SIZE)

print(f"    训练/验证/测试 = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

# 统计标签缺失率
all_y = torch.cat([d.y for d in dataset], dim=0)
nan_rate = all_y.isnan().float().mean().item()
print(f"    标签缺失率: {nan_rate:.1%}")

# ============ 2. 模型定义 ============

class GINModel(nn.Module):
    """GIN + Jumping Knowledge + Readout"""
    def __init__(self, in_dim, hidden, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            in_c = in_dim if i == 0 else hidden
            # GIN: MLP 聚合
            mlp = nn.Sequential(
                nn.Linear(in_c, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        xs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            xs.append(x)
        # Jumping Knowledge: 取最后一层
        x = xs[-1]
        # Readout: 全局均值池化
        x = global_mean_pool(x, batch)
        # MLP 预测头
        x = self.dropout(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return x


class GCNModel(nn.Module):
    """GCN + Readout"""
    def __init__(self, in_dim, hidden, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            in_c = in_dim if i == 0 else hidden
            self.convs.append(GCNConv(in_c, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return x


class SAGEModel(nn.Module):
    """GraphSAGE + Readout"""
    def __init__(self, in_dim, hidden, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for i in range(n_layers):
            in_c = in_dim if i == 0 else hidden
            self.convs.append(SAGEConv(in_c, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.lin1(x)))
        x = self.lin2(x)
        return x


# ============ 3. 训练/评估函数 ============

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # [B, 12]
        y = data.y.squeeze(-1)  # [B, 12] 或 [B]
        if y.dim() == 1:
            y = y.unsqueeze(1)
        # mask NaN 标签
        mask = ~y.isnan()
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(
            out[mask], y[mask]
        )
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred, all_y = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y = data.y.squeeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        all_pred.append(out.cpu())
        all_y.append(y.cpu())

    all_pred = torch.cat(all_pred, dim=0)  # [N, 12]
    all_y    = torch.cat(all_y, dim=0)

    # 逐任务计算 AUC
    aucs = []
    for t in range(all_y.shape[1]):
        yt = all_y[:, t]
        pt = all_pred[:, t]
        mask = ~yt.isnan()
        yt_clean = yt[mask]
        pt_clean = pt[mask]
        if len(yt_clean.unique()) > 1:
            aucs.append(roc_auc_score(yt_clean.numpy(), pt_clean.numpy()))
    
    return sum(aucs) / len(aucs) if aucs else 0.0, aucs


def run_experiment(model_cls, name, train_loader, val_loader, test_loader, device):
    print(f"\n{'─' * 55}")
    print(f"  训练 {name}")
    print(f"{'─' * 55}")

    model = model_cls(
        in_dim=dataset.num_node_features,
        hidden=HIDDEN,
        out_dim=12,
        n_layers=NUM_LAYERS
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=15, factor=0.5, min_lr=1e-5
    )

    best_val_auc = 0
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, opt, device)
        val_auc, _ = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={loss:.4f} | val_AUC={val_auc:.4f} | best={best_val_auc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"  早停 @ Epoch {epoch}")
            break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model = model.to(device)

    # 最终测试评估
    test_auc, task_aucs = evaluate(model, test_loader, device)
    print(f"\n  ✅ {name} 结果:")
    print(f"     验证AUC = {best_val_auc:.4f}")
    print(f"     测试AUC = {test_auc:.4f}")
    print(f"     训练时间 = {elapsed:.1f}s")
    print(f"     逐任务AUC: {[f'{a:.3f}' for a in task_aucs]}")

    return {
        "name": name,
        "params": n_params,
        "val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "task_aucs": [round(a, 4) for a in task_aucs],
        "time": round(elapsed, 1),
    }


# ============ 4. 运行实验 ============
print(f"\n[2] 模型配置: hidden={HIDDEN}, layers={NUM_LAYERS}, lr={LR}")
print(f"[3] 开始训练 (早停patience={PATIENCE})...")

results = []
for cls, name in [(GINModel, "GIN"), (GCNModel, "GCN"), (SAGEModel, "GraphSAGE")]:
    r = run_experiment(cls, name, train_loader, val_loader, test_loader, DEVICE)
    results.append(r)

# ============ 5. 可视化 ============
print(f"\n{'=' * 65}")
print("最终结果汇总")
print(f"{'=' * 65}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 5a. 柱状图
names  = [r["name"] for r in results]
val_aucs = [r["val_auc"] for r in results]
test_aucs = [r["test_auc"] for r in results]
x = np.arange(len(names))
w = 0.3
bars1 = axes[0].bar(x - w/2, val_aucs, w, label='Val AUC', color='#3498db', edgecolor='white', linewidth=1.5)
bars2 = axes[0].bar(x + w/2, test_aucs, w, label='Test AUC', color='#e74c3c', edgecolor='white', linewidth=1.5)
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{bar.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{bar.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names, fontsize=12)
axes[0].set_ylabel('AUC-ROC', fontsize=12)
axes[0].set_ylim(0.4, 1.0)
axes[0].legend(fontsize=10)
axes[0].set_title('模型性能对比 (Tox21)', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, label='随机基线')

# 5b. 逐任务AUC热力图
task_names = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
n_tasks = len(task_names)
task_matrix = np.zeros((len(results), n_tasks))
for i, r in enumerate(results):
    aucs = r["task_aucs"]
    for j in range(min(len(aucs), n_tasks)):
        task_matrix[i, j] = aucs[j]

im = axes[1].imshow(task_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
axes[1].set_xticks(range(n_tasks))
axes[1].set_xticklabels(task_names, rotation=45, ha='right', fontsize=8)
axes[1].set_yticks(range(len(results)))
axes[1].set_yticklabels(names, fontsize=11)
axes[1].set_title('逐任务 AUC 热力图', fontsize=13, fontweight='bold')
for i in range(len(results)):
    for j in range(n_tasks):
        v = task_matrix[i, j]
        if v > 0:
            axes[1].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if v < 0.65 else 'black')
fig.colorbar(im, ax=axes[1], shrink=0.8)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/tox21_gnn_correct.png', dpi=150, bbox_inches='tight')
print(f"图表保存: {OUT_DIR}/tox21_gnn_correct.png")

# 保存结果
with open(f'{OUT_DIR}/tox21_gnn_correct_results.json', 'w') as f:
    json.dump({
        "dataset": "Tox21 (MoleculeNet)",
        "n_molecules": len(dataset),
        "split": f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}",
        "features": f"node_features({dataset.num_node_features}-dim)",
        "models": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n结果JSON: {OUT_DIR}/tox21_gnn_correct_results.json")

# 打印汇总表
print(f"\n{'模型':<12} {'参数量':>8} {'Val AUC':>8} {'Test AUC':>9} {'时间':>6}")
print("─" * 50)
for r in results:
    print(f"{r['name']:<12} {r['params']:>8,} {r['val_auc']:>8.4f} {r['test_auc']:>9.4f} {r['time']:>5.1f}s")

print(f"\n✅ 实验完成！")
