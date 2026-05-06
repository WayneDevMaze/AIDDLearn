#!/usr/bin/env python3
"""
Tox21 分子性质预测 — 从CSV+RDKit正确构建分子图
1. 读取本地 tox21.csv，用 RDKit 从 SMILES 构建分子图
2. 实现 GIN / GCN / GraphSAGE + readout 池化
3. 12任务多标签分类，NaN标签自动mask
"""
import warnings; warnings.filterwarnings('ignore')
import json, time, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import (
    GINConv, GCNConv, SAGEConv, global_mean_pool, global_add_pool
)
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# ============ 配置 ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.expanduser('~/Desktop/AIDDLearn/tox21_gin')
CSV_PATH = os.path.join(OUT_DIR, 'data/tox21/raw/tox21.csv')
HIDDEN = 128
NUM_LAYERS = 3
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30

print("=" * 65)
print("Tox21 分子性质预测 — GNN 正确实现 (RDKit)")
print("=" * 65)
print(f"Device: {DEVICE}")
print(f"RDKit: {'✓' if HAS_RDKIT else '✗'}")


# ============ 1. 从CSV构建分子图 ============
def atom_features(atom):
    """原子特征: 类型、度数、氢数、隐式价、芳香性"""
    return [
        # 原子类型 one-hot (C,N,O,F,P,S,Cl,Br,I,other)
        int(atom.GetSymbol() == s) for s in
        ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    ] + [
        int(atom.GetSymbol() not in ['C','N','O','F','P','S','Cl','Br','I']),  # other
        atom.GetDegree(),           # 度数 0-5
        atom.GetTotalNumHs(),       # 氢数 0-4
        atom.GetImplicitValence(),  # 隐式价 0-6
        int(atom.GetIsAromatic()),  # 芳香性
    ]  # 共 15 维


def mol_to_graph(smiles):
    """SMILES -> PyG Data 对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 原子特征
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # 边 (双向)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        # 单原子分子
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


class Tox21Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        super().__init__(transform=transform)
        self.df = pd.read_csv(csv_path)
        # 标签列: 以 NR- 或 SR- 开头
        self.label_cols = [c for c in self.df.columns
                           if c.startswith('NR-') or c.startswith('SR-')]
        print(f"  CSV: {len(self.df)} 行, 标签列: {len(self.label_cols)}")
        print(f"  任务: {self.label_cols}")
        self._graphs = None

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        # 缓存已构建的图
        if self._graphs is None:
            self._graphs = [None] * len(self.df)

        if self._graphs[idx] is None:
            smiles = row['smiles']
            g = mol_to_graph(smiles)
            if g is None:
                # 无效SMILES: 占位图
                g = Data(x=torch.zeros(1, 15), edge_index=torch.zeros(2, 0, dtype=torch.long))
            self._graphs[idx] = g
        else:
            g = self._graphs[idx]

        # 标签
        y = torch.tensor([row[c] if pd.notna(row[c]) else float('nan')
                          for c in self.label_cols], dtype=torch.float)
        g.y = y
        return g

    @property
    def num_node_features(self):
        return 15  # atom_features 输出维度

    @property
    def num_classes(self):
        return len(self.label_cols)


print("\n[1] 从 CSV 构建分子图 (RDKit)...")
dataset = Tox21Dataset(CSV_PATH)
print(f"    分子总数: {len(dataset)}")
print(f"    节点特征维度: {dataset.num_node_features}")
print(f"    标签任务数: {dataset.num_classes}")

# 预构建所有图
print("    构建分子图...")
n_invalid = 0
for i in range(len(dataset)):
    _ = dataset[i]
    if dataset._graphs[i].num_nodes == 1 and dataset._graphs[i].num_edges == 0:
        n_invalid += 1
print(f"    完成! 无效SMILES: {n_invalid}")

# 统计
sizes = [g.num_nodes for g in dataset._graphs]
print(f"    原子数: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")

# 随机划分
n = len(dataset)
perm = torch.randperm(n)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train_ds = Subset(dataset, perm[:n_train].tolist())
val_ds = Subset(dataset, perm[n_train:n_train + n_val].tolist())
test_ds = Subset(dataset, perm[n_train + n_val:].tolist())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print(f"    训练/验证/测试 = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")


# ============ 2. 模型 ============
class GINModel(nn.Module):
    """GIN + Readout"""
    def __init__(self, in_dim, hidden, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(n_layers):
            in_c = in_dim if i == 0 else hidden
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
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.lin1(x)))
        return self.lin2(x)


class GCNModel(nn.Module):
    """GCN + Readout"""
    def __init__(self, in_dim, hidden, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
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
        return self.lin2(x)


class SAGEModel(nn.Module):
    """GraphSAGE + Readout"""
    def __init__(self, in_dim, hidden, out_dim, n_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
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
        return self.lin2(x)


# ============ 3. 训练/评估 ============
def train_epoch(model, loader, opt, device):
    model.train()
    total_loss, n_batches = 0, 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        y = data.y
        if y.dim() == 1:
            y = y.unsqueeze(1)
        mask = ~y.isnan()
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
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
        y = data.y
        if y.dim() == 1:
            y = y.unsqueeze(1)
        all_pred.append(out.cpu())
        all_y.append(y.cpu())

    all_pred = torch.cat(all_pred, dim=0)
    all_y = torch.cat(all_y, dim=0)

    aucs = []
    for t in range(all_y.shape[1]):
        yt, pt = all_y[:, t], all_pred[:, t]
        mask = ~yt.isnan()
        yt_c, pt_c = yt[mask], pt[mask]
        if len(yt_c.unique()) > 1:
            aucs.append(roc_auc_score(yt_c.numpy(), pt_c.numpy()))

    return sum(aucs) / len(aucs) if aucs else 0.0, aucs


def run_experiment(model_cls, name):
    print(f"\n{'─' * 55}")
    print(f"  训练 {name}")
    print(f"{'─' * 55}")

    model = model_cls(
        in_dim=15, hidden=HIDDEN, out_dim=12, n_layers=NUM_LAYERS
    ).to(device=DEVICE)

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
        loss = train_epoch(model, train_loader, opt, DEVICE)
        val_auc, _ = evaluate(model, val_loader, DEVICE)
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
    model = model.to(DEVICE)

    test_auc, task_aucs = evaluate(model, test_loader, DEVICE)
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


# ============ 4. 运行 ============
print(f"\n[2] 模型: hidden={HIDDEN}, layers={NUM_LAYERS}, lr={LR}")
print(f"[3] 开始训练...")

results = []
for cls, name in [(GINModel, "GIN"), (GCNModel, "GCN"), (SAGEModel, "GraphSAGE")]:
    results.append(run_experiment(cls, name))

# ============ 5. 可视化 ============
print(f"\n{'=' * 65}")
print("最终结果汇总")
print(f"{'=' * 65}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

names = [r["name"] for r in results]
val_aucs = [r["val_auc"] for r in results]
test_aucs = [r["test_auc"] for r in results]
x = np.arange(len(names))
w = 0.3
bars1 = axes[0].bar(x - w/2, val_aucs, w, label='Val AUC', color='#3498db',
                     edgecolor='white', linewidth=1.5)
bars2 = axes[0].bar(x + w/2, test_aucs, w, label='Test AUC', color='#e74c3c',
                     edgecolor='white', linewidth=1.5)
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
axes[0].set_title('GNN模型性能对比 (Tox21)', fontsize=13, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)

# 逐任务热力图
task_names = dataset.label_cols
n_tasks = len(task_names)
task_matrix = np.zeros((len(results), n_tasks))
for i, r in enumerate(results):
    for j in range(min(len(r["task_aucs"]), n_tasks)):
        task_matrix[i, j] = r["task_aucs"][j]

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
print(f"\n图表: {OUT_DIR}/tox21_gnn_correct.png")

with open(f'{OUT_DIR}/tox21_gnn_correct_results.json', 'w') as f:
    json.dump({
        "dataset": "Tox21 (RDKit processed)",
        "n_molecules": len(dataset),
        "split": f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}",
        "features": "atom_features(15-dim): type,degree,Hs,valence,aromatic",
        "models": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n{'模型':<12} {'参数量':>8} {'Val AUC':>8} {'Test AUC':>9} {'时间':>6}")
print("─" * 50)
for r in results:
    print(f"{r['name']:<12} {r['params']:>8,} {r['val_auc']:>8.4f} {r['test_auc']:>9.4f} {r['time']:>5.1f}s")

print(f"\n✅ 全部完成!")
