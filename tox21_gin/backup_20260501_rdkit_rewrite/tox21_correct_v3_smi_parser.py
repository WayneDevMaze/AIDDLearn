#!/usr/bin/env python3
"""
Tox21 分子性质预测 — 从CSV纯Python构建分子图（无需RDKit）
1. 解析SMILES构建原子节点和键边
2. 实现 GIN / GCN / GraphSAGE + readout 池化
3. 12任务多标签分类
"""
import warnings; warnings.filterwarnings('ignore')
import json, time, os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import (
    GINConv, GCNConv, SAGEConv, global_mean_pool
)
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
print("Tox21 GNN — 纯Python SMILES解析 (无需RDKit)")
print("=" * 65)
print(f"Device: {DEVICE}")


# ============ 1. 轻量SMILES解析器 ============
# 只提取原子和键（不含立体化学），足够构建分子图

# 元素周期表（常见有机分子元素）
ATOMIC_SYMBOLS = {
    'Br', 'Cl', 'Si', 'Se', 'Na', 'Li', 'Al', 'Mg', 'Ca',
    'C', 'N', 'O', 'F', 'P', 'S', 'I', 'B', 'K',
}

def parse_smiles_graph(smiles):
    """
    解析SMILES，返回(原子列表, 边列表)
    原子列表: [原子符号, ...]
    边列表: [[i,j,bond_type], ...]  bond_type: 1=single, 2=double, 3=triple, 4=aromatic
    """
    atoms = []
    edges = []
    stack = []  # 支链栈
    ring_open = {}  # 环开环: 编号 -> (原子idx, bond_type)
    i = 0
    prev_atom = -1
    bond_type = 1  # 默认单键

    while i < len(smiles):
        c = smiles[i]

        if c == '(':
            stack.append(prev_atom)
            i += 1
            continue
        elif c == ')':
            prev_atom = stack.pop()
            bond_type = 1
            i += 1
            continue
        elif c in '-=:':
            bond_type = {'-': 1, '=': 2, '#': 3}[c]
            i += 1
            continue
        elif c == '/':  # 忽略立体化学
            i += 1
            continue
        elif c == '\\':
            i += 1
            continue
        elif c == '.':  # 断键
            prev_atom = -1
            bond_type = 1
            i += 1
            continue
        elif c == '[':
            # 方括号原子 [NH2], [C@H], etc.
            j = smiles.index(']', i)
            bracket_content = smiles[i+1:j]
            # 提取元素符号
            elem = ''
            for k, ch in enumerate(bracket_content):
                if ch.isalpha():
                    if not elem:
                        elem += ch
                    elif ch.isupper():
                        break
                    else:
                        elem += ch
                elif elem:
                    break
            if not elem or elem not in ATOMIC_SYMBOLS:
                elem = 'C'  # 默认碳
            atoms.append(elem)
            curr_atom = len(atoms) - 1
            if prev_atom >= 0:
                bt = bond_type
                edges.append([prev_atom, curr_atom, bt])
                edges.append([curr_atom, prev_atom, bt])
            prev_atom = curr_atom
            bond_type = 1
            i = j + 1
            continue
        elif c.isdigit() or c == '%':
            # 环闭合
            if c == '%':
                ring_num = int(smiles[i+1:i+3])
                i += 3
            else:
                ring_num = int(c)
                i += 1
            if ring_num in ring_open:
                other_idx, other_bond = ring_open.pop(ring_num)
                bt = max(bond_type, other_bond) if bond_type > 1 or other_bond > 1 else 1
                edges.append([other_idx, prev_atom, bt])
                edges.append([prev_atom, other_idx, bt])
                bond_type = 1
            else:
                ring_open[ring_num] = (prev_atom, bond_type)
                bond_type = 1
            continue
        elif c.isalpha():
            # 普通原子
            # 检查双字母元素
            if i + 1 < len(smiles) and smiles[i:i+2] in ATOMIC_SYMBOLS:
                elem = smiles[i:i+2]
                i += 2
            elif c in ATOMIC_SYMBOLS:
                elem = c
                i += 1
            else:
                # 未知元素，跳过
                i += 1
                continue

            atoms.append(elem)
            curr_atom = len(atoms) - 1

            # 芳香性: 小写字母表示芳香原子
            is_aromatic = c.islower()

            if prev_atom >= 0:
                bt = bond_type
                # 芳香键处理: 两个芳香原子之间默认是芳香键(4)
                if is_aromatic and bt == 1:
                    bt = 4  # aromatic
                edges.append([prev_atom, curr_atom, bt])
                edges.append([curr_atom, prev_atom, bt])
            prev_atom = curr_atom
            bond_type = 1
            continue
        else:
            i += 1
            continue

    return atoms, edges


# 原子特征编码
ATOM_TYPES = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Other']
BOND_TYPES = {1: 0, 2: 1, 3: 2, 4: 3}  # single, double, triple, aromatic

def smiles_to_graph(smiles):
    """SMILES -> PyG Data"""
    atoms, edges = parse_smiles_graph(smiles)
    if not atoms:
        # 空图: 用一个碳原子占位
        x = torch.zeros(1, len(ATOM_TYPES))
        x[0, 0] = 1  # C
        return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

    # 原子特征: one-hot atom type
    x = torch.zeros(len(atoms), len(ATOM_TYPES))
    for i, atom in enumerate(atoms):
        if atom in ATOM_TYPES:
            x[i, ATOM_TYPES.index(atom)] = 1
        else:
            x[i, -1] = 1  # Other

    # 边索引 - 过滤非法边
    if edges:
        valid_edges = [(e[0], e[1]) for e in edges
                       if 0 <= e[0] < len(atoms) and 0 <= e[1] < len(atoms)]
        if valid_edges:
            ei = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)

    return Data(x=x, edge_index=ei)


# ============ 2. 数据集 ============
class Tox21Dataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.label_cols = [c for c in self.df.columns
                           if c.startswith('NR-') or c.startswith('SR-')]
        self._graphs = [None] * len(self.df)
        self._build_all()

    def _build_all(self):
        print(f"  构建 {len(self.df)} 个分子图...")
        n_fail = 0
        for i in range(len(self.df)):
            smiles = self.df.iloc[i]['smiles']
            g = smiles_to_graph(smiles)
            self._graphs[i] = g
            if g.num_nodes <= 1 and g.num_edges == 0:
                n_fail += 1
        print(f"  完成! 解析失败: {n_fail}")

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def len(self): return len(self.df)

    def get(self, idx):
        g = self._graphs[idx]
        row = self.df.iloc[idx]
        y = torch.tensor([[row[c] if pd.notna(row[c]) else float('nan')
                           for c in self.label_cols]], dtype=torch.float)  # [1, 12]
        g = g.clone()
        g.y = y
        return g

    @property
    def num_node_features(self):
        return len(ATOM_TYPES)

    @property
    def num_classes(self):
        return len(self.label_cols)


print("\n[1] 构建分子图...")
dataset = Tox21Dataset(CSV_PATH)
print(f"    分子数: {len(dataset)}")
print(f"    节点特征: {dataset.num_node_features}-dim")
print(f"    任务数: {dataset.num_classes}")

# 统计
sizes = [g.num_nodes for g in dataset._graphs]
edges = [g.num_edges for g in dataset._graphs]
print(f"    原子数: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
print(f"    键数: min={min(edges)}, max={max(edges)}, avg={np.mean(edges):.1f}")

# 验证前3个
for i in range(3):
    g = dataset._graphs[i]
    smi = dataset.df.iloc[i]['smiles']
    print(f"    [{i}] {smi[:40]:40s} → {g.num_nodes} atoms, {g.num_edges} edges")

# 划分
n = len(dataset)
perm = torch.randperm(n)
n_train, n_val = int(0.8 * n), int(0.1 * n)

train_ds = Subset(dataset, perm[:n_train].tolist())
val_ds = Subset(dataset, perm[n_train:n_train + n_val].tolist())
test_ds = Subset(dataset, perm[n_train + n_val:].tolist())

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print(f"    训练/验证/测试 = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")


# ============ 3. 模型 ============
class GINModel(nn.Module):
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


# ============ 4. 训练 ============
def train_epoch(model, loader, opt, device):
    model.train()
    total_loss, n_batches = 0, 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)  # [B, 12]
        y = data.y  # [B, 12] (stored as [1,12] per sample, DataLoader concatenates to [B,12])
        if y.dim() == 1:
            y = y.reshape(-1, 12)
        # Flatten both for masked loss
        out_flat = out.reshape(-1)
        y_flat = y.reshape(-1)
        mask = ~y_flat.isnan()
        if mask.sum() == 0:
            continue
        loss = F.binary_cross_entropy_with_logits(out_flat[mask], y_flat[mask])
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
        out = model(data.x, data.edge_index, data.batch)  # [B, 12]
        y = data.y
        if y.dim() == 1:
            y = y.reshape(-1, 12)
        all_pred.append(out.cpu())
        all_y.append(y.cpu())

    all_pred = torch.cat(all_pred, dim=0)  # [N, 12]
    all_y = torch.cat(all_y, dim=0)  # [N, 12]

    # Per-task AUC
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
        in_dim=len(ATOM_TYPES), hidden=HIDDEN, out_dim=12, n_layers=NUM_LAYERS
    ).to(DEVICE)

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
    print(f"\n  ✅ {name}:")
    print(f"     Val AUC = {best_val_auc:.4f}")
    print(f"     Test AUC = {test_auc:.4f}")
    print(f"     时间 = {elapsed:.1f}s")
    print(f"     逐任务: {[f'{a:.3f}' for a in task_aucs]}")

    return {
        "name": name,
        "params": n_params,
        "val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "task_aucs": [round(a, 4) for a in task_aucs],
        "time": round(elapsed, 1),
    }


# ============ 5. 运行 ============
print(f"\n[2] hidden={HIDDEN}, layers={NUM_LAYERS}, lr={LR}")
print(f"[3] 开始训练...")

results = []
for cls, name in [(GINModel, "GIN"), (GCNModel, "GCN"), (SAGEModel, "GraphSAGE")]:
    results.append(run_experiment(cls, name))

# ============ 6. 可视化 ============
print(f"\n{'=' * 65}")
print("最终结果")
print(f"{'=' * 65}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

names = [r["name"] for r in results]
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
fig_path = f'{OUT_DIR}/tox21_gnn_correct.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n图表: {fig_path}")

json_path = f'{OUT_DIR}/tox21_gnn_correct_results.json'
with open(json_path, 'w') as f:
    json.dump({
        "dataset": "Tox21 (SMILES parsed, no RDKit)",
        "n_molecules": len(dataset),
        "split": f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)}",
        "features": f"atom_type_onehot({len(ATOM_TYPES)}-dim)",
        "models": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n{'模型':<12} {'参数量':>8} {'Val AUC':>8} {'Test AUC':>9} {'时间':>6}")
print("─" * 50)
for r in results:
    print(f"{r['name']:<12} {r['params']:>8,} {r['val_auc']:>8.4f} {r['test_auc']:>9.4f} {r['time']:>5.1f}s")

print(f"\n✅ 全部完成!")
