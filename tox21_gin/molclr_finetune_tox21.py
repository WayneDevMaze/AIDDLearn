"""
MolCLR Pre-trained GIN Fine-tuning on Tox21 (Multi-task)
=========================================================
Uses MolCLR pre-trained weights (5-layer GIN, emb=300, feat=512)
Adapted for Tox21 12-task multi-label classification with CPU support.
"""

import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ======================== MolCLR GIN Model ========================

NUM_ATOM_TYPE = 119
NUM_CHIRALITY_TAG = 3
NUM_BOND_TYPE = 5
NUM_BOND_DIRECTION = 3


class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4  # bond type for self-loop
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class MolCLRGIN(nn.Module):
    """MolCLR GIN encoder + multi-task prediction head for Tox21."""

    def __init__(self, num_tasks=12, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.3, pool="mean"):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        # Atom embeddings (same as MolCLR pre-training)
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        # GIN layers
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        self.batch_norms = nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # Pooling
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "max":
            self.pool = global_max_pool
        else:
            self.pool = global_add_pool

        # Feature projection (from pre-training)
        self.feat_lin = nn.Linear(emb_dim, feat_dim)

        # Multi-task prediction head (NEW - not from pre-training)
        self.pred_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Softplus(),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.Softplus(),
            nn.Linear(feat_dim // 4, num_tasks),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Embed atoms
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        # GIN layers
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        # Graph-level pooling
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)

        # Prediction
        pred = self.pred_head(h)
        return h, pred

    def load_pretrained(self, state_dict):
        """Load MolCLR pre-trained weights, skipping prediction head."""
        own_state = self.state_dict()
        loaded, skipped = 0, 0
        for name, param in state_dict.items():
            if name not in own_state:
                skipped += 1
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
            loaded += 1
        print(f"  Loaded {loaded} params, skipped {skipped} (pred_head/out_lin)")

    def freeze_encoder(self):
        """Freeze GNN encoder, only train prediction head."""
        for name, param in self.named_parameters():
            if "pred_head" not in name and "feat_lin" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# ======================== Data Loading ========================

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data with MolCLR-compatible features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # Atom features
    type_idx, chirality_idx = [], []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in ATOM_LIST:
            type_idx.append(ATOM_LIST.index(atomic_num))
        else:
            type_idx.append(0)
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    # Edge features
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        s, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [s, e]
        col += [e, s]
        ef = [
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir()),
        ]
        edge_feat.append(ef)
        edge_feat.append(ef)

    if len(row) == 0:
        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_feat, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def load_tox21_data(data_dir="/Users/user/Desktop/AIDDLearn/tox21_gin/data/tox21"):
    """Load Tox21 dataset from CSV with MolCLR-compatible graph features."""
    import csv

    csv_path = os.path.join(data_dir, "raw", "tox21.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tox21 CSV not found: {csv_path}")

    task_names = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

    graphs = []
    skipped = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("smiles", "")
            if not smiles:
                skipped += 1
                continue

            # Parse labels (12 tasks, may be empty/NaN)
            labels = []
            for t in task_names:
                val = row.get(t, "")
                if val == "" or val is None:
                    labels.append(float("nan"))
                else:
                    labels.append(float(val))
            y = torch.tensor([labels], dtype=torch.float)  # [1, 12]

            # Convert SMILES to graph
            g = smiles_to_graph(smiles)
            if g is not None:
                g.y = y
                graphs.append(g)
            else:
                skipped += 1

    print(f"  Loaded {len(graphs)} molecules (skipped {skipped}) with MolCLR features")
    return graphs


def split_data(graphs, train_ratio=0.8, val_ratio=0.1):
    """Random split."""
    n = len(graphs)
    indices = list(range(n))
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx], [graphs[i] for i in test_idx]


# ======================== Training ========================


def evaluate(model, loader, device, num_tasks=12):
    """Evaluate multi-task AUC."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, pred = model(data)
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()  # [N, 12]
    all_labels = torch.cat(all_labels, dim=0).numpy()  # [N, 12]

    # Handle shape
    if all_labels.ndim == 3:
        all_labels = all_labels.squeeze(1)

    # Handle NaN labels
    aucs = []
    for t in range(num_tasks):
        if all_preds.ndim == 1 or all_preds.shape[1] <= t:
            aucs.append(0.5)
            continue
        task_label = all_labels[:, t]
        task_pred = all_preds[:, t]
        valid = ~np.isnan(task_label)
        if valid.sum() > 10 and len(np.unique(task_label[valid])) > 1:
            auc = roc_auc_score(task_label[valid], task_pred[valid])
            aucs.append(auc)
        else:
            aucs.append(0.5)

    mean_auc = np.mean(aucs) if aucs else 0.0
    return mean_auc, aucs


def _train_phase(model, train_loader, val_loader, test_loader, criterion,
                 epochs, lr, weight_decay, phase_name=""):
    best_val_auc = 0.0
    best_test_auc = 0.0
    best_task_aucs = []
    patience_counter = 0
    patience = 15

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()

            _, pred = model(data)  # [B, 12]

            # Multi-task BCE with NaN mask
            labels = data.y
            if labels.dim() == 3:
                labels = labels.squeeze(1)  # [B, 12]
            labels = labels.float()

            # Handle shape mismatches
            if labels.shape != pred.shape:
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).expand(-1, pred.size(1))

            loss_per_sample = criterion(pred, labels)  # [B, 12]

            # Mask NaN labels
            mask = ~torch.isnan(labels)
            loss = (loss_per_sample * mask).sum() / mask.sum().clamp(min=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_auc, _ = evaluate(model, val_loader, DEVICE)
            test_auc, task_aucs = evaluate(model, test_loader, DEVICE)
            elapsed = time.time() - start_time
            print(f"  [{phase_name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | Time: {elapsed:.0f}s")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = test_auc
                best_task_aucs = task_aucs
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time
    print(f"  [{phase_name}] Best Val AUC: {best_val_auc:.4f} | Test AUC: {best_test_auc:.4f} | Total: {elapsed:.0f}s")
    return best_val_auc, best_test_auc, best_task_aucs


def train_finetune(
    pretrained_path="/Users/user/Desktop/AIDDLearn/MolCLR/ckpt/pretrained_gin/checkpoints/model.pth",
    data_dir="/Users/user/Desktop/AIDDLearn/tox21_gin/data/tox21",
    mode="two_stage",
    epochs_stage1=30,
    epochs_stage2=100,
    batch_size=64,
    lr_head=1e-3,
    lr_encoder=5e-5,
    weight_decay=1e-6,
):
    """
    Fine-tune MolCLR on Tox21.

    Modes:
    - linear_probe: freeze encoder, only train head
    - two_stage: stage1 freeze encoder, stage2 unfreeze all
    - end_to_end: train everything from start
    """
    print("=" * 60)
    print(f"MolCLR Fine-tuning on Tox21 | Mode: {mode}")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading Tox21 data...")
    graphs = load_tox21_data(data_dir)
    train_graphs, val_graphs, test_graphs = split_data(graphs)
    print(f"  Train/Val/Test: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # Build model
    print("\n[2/4] Building MolCLR-GIN model...")
    model = MolCLRGIN(num_tasks=12, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0.3, pool="mean")

    # Load pre-trained weights
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"  Loading pre-trained weights: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_pretrained(state_dict)
    else:
        print("  WARNING: No pre-trained weights found, training from scratch!")

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # Loss
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    results = {}

    if mode == "linear_probe":
        model.freeze_encoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[3/4] Linear probe | Trainable: {trainable:,}")
        best_auc, test_auc, task_aucs = _train_phase(
            model, train_loader, val_loader, test_loader, criterion,
            epochs=epochs_stage2, lr=lr_head, weight_decay=weight_decay, phase_name="linear_probe"
        )
        results["linear_probe"] = {"val_auc": float(best_auc), "test_auc": float(test_auc), "task_aucs": [float(a) for a in task_aucs]}

    elif mode == "two_stage":
        # Stage 1: Freeze encoder, train head
        model.freeze_encoder()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[3a/4] Stage 1: Frozen encoder | Trainable: {trainable:,}")
        _train_phase(
            model, train_loader, val_loader, test_loader, criterion,
            epochs=epochs_stage1, lr=lr_head, weight_decay=weight_decay, phase_name="stage1_frozen"
        )

        # Stage 2: Unfreeze all
        model.unfreeze_all()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[3b/4] Stage 2: Full fine-tune | Trainable: {trainable:,}")
        best_auc, test_auc, task_aucs = _train_phase(
            model, train_loader, val_loader, test_loader, criterion,
            epochs=epochs_stage2, lr=lr_encoder, weight_decay=weight_decay, phase_name="stage2_full"
        )
        results["two_stage"] = {"val_auc": float(best_auc), "test_auc": float(test_auc), "task_aucs": [float(a) for a in task_aucs]}

    elif mode == "end_to_end":
        print("\n[3/4] End-to-end fine-tune")
        best_auc, test_auc, task_aucs = _train_phase(
            model, train_loader, val_loader, test_loader, criterion,
            epochs=epochs_stage2, lr=lr_encoder, weight_decay=weight_decay, phase_name="end_to_end"
        )
        results["end_to_end"] = {"val_auc": float(best_auc), "test_auc": float(test_auc), "task_aucs": [float(a) for a in task_aucs]}

    # Final evaluation
    print("\n[4/4] Final Test Evaluation:")
    val_auc, _ = evaluate(model, val_loader, DEVICE)
    test_auc, task_aucs = evaluate(model, test_loader, DEVICE)
    print(f"  Val AUC:  {val_auc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    task_names = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
                  "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
    for name, auc in zip(task_names, task_aucs):
        print(f"    {name}: {auc:.4f}")

    results["final"] = {"val_auc": float(val_auc), "test_auc": float(test_auc), "task_aucs": [float(a) for a in task_aucs]}
    return results


if __name__ == "__main__":
    # Run all three modes for comparison
    all_results = {}

    for mode in ["linear_probe", "two_stage", "end_to_end"]:
        print(f"\n\n{'='*60}")
        print(f"  Running mode: {mode}")
        print(f"{'='*60}")
        results = train_finetune(mode=mode)
        all_results[mode] = results

    # Save results
    output_path = "/Users/user/Desktop/AIDDLearn/tox21_gin/molclr_finetune_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: MolCLR Fine-tuning vs From-Scratch Baseline")
    print("=" * 60)
    header = f"{'Method':<25} {'Val AUC':>10} {'Test AUC':>10}"
    print(header)
    print("-" * 45)
    baseline_gin = f"{'GIN (from scratch)':<25} {'0.8420':>10} {'0.8500':>10}"
    baseline_gs = f"{'GraphSAGE (from scratch)':<25} {'0.8412':>10} {'0.8502':>10}"
    print(baseline_gin)
    print(baseline_gs)
    for mode, res in all_results.items():
        final = res.get("final", res.get(mode, {}))
        val = final.get("val_auc", 0)
        test = final.get("test_auc", 0)
        line = f"MolCLR-{mode:<18} {val:>10.4f} {test:>10.4f}"
        print(line)
