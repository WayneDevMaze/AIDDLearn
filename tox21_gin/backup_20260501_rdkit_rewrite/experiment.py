"""
GNN分子毒性预测综合实验
复现2017-2024年代表性方法在Tox21数据集上的性能
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from models import get_all_models

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 数据加载 ====================

def load_data():
    """加载Tox21数据集"""
    dataset = MoleculeNet(root='data', name='Tox21')
    dataset = dataset.shuffle()
    
    # 划分训练集和测试集（8:2）
    train_size = int(len(dataset) * 0.8)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    print(f"数据集大小: {len(dataset)}")
    print(f"训练集: {len(train_dataset)}, 测试集: {len(test_dataset)}")
    print(f"节点特征维度: {dataset.num_features}")
    print(f"任务数: {dataset.num_classes}")
    
    return dataset, train_dataset, test_dataset


def compute_pos_weight(train_dataset, task_idx=0):
    """计算正样本权重用于处理类别不平衡"""
    y_train = []
    for data in train_dataset:
        y = data.y[:, task_idx]
        mask = ~torch.isnan(y)
        y_train.append(y[mask])
    
    y_train = torch.cat(y_train)
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    print(f"任务{task_idx} - 正样本: {num_pos}, 负样本: {num_neg}")
    print(f"正样本比例: {num_pos/len(y_train):.4f}, 正样本权重: {pos_weight:.4f}")
    
    return pos_weight


# ==================== 2. 训练与评估函数 ====================

def train_epoch(model, loader, optimizer, pos_weight, task_idx=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, task_idx].float()
        mask = ~torch.isnan(y)
        
        if mask.sum() == 0:
            continue
            
        loss = F.binary_cross_entropy_with_logits(
            out[:, 0][mask],
            y[mask],
            pos_weight=torch.tensor([pos_weight], device=device) if pos_weight != 1.0 else None
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, task_idx=0):
    """评估模型性能"""
    model.eval()
    ys, preds, probs = [], [], []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, task_idx]
        mask = ~torch.isnan(y)
        
        if mask.sum() == 0:
            continue
        
        ys.append(y[mask].cpu())
        preds.append(out[:, 0][mask].cpu())
        probs.append(torch.sigmoid(out[:, 0][mask]).cpu())
    
    y = torch.cat(ys).numpy()
    pred_logits = torch.cat(preds).numpy()
    pred_probs = torch.cat(probs).numpy()
    pred_labels = (pred_probs > 0.5).astype(int)
    
    # 计算各项指标
    metrics = {
        'auc': roc_auc_score(y, pred_probs),
        'accuracy': accuracy_score(y, pred_labels),
        'precision': precision_score(y, pred_labels, zero_division=0),
        'recall': recall_score(y, pred_labels, zero_division=0),
        'f1': f1_score(y, pred_labels, zero_division=0),
    }
    
    return metrics, y, pred_probs


# ==================== 3. 主实验流程 ====================

def run_experiment(model_name, model, train_loader, test_loader, pos_weight, 
                   epochs=100, lr=0.001, task_idx=0):
    """运行单个模型的实验"""
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    best_auc = 0
    best_metrics = None
    history = {'train_loss': [], 'test_auc': []}
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, pos_weight, task_idx)
        test_metrics, _, _ = evaluate(model, test_loader, task_idx)
        test_auc = test_metrics['auc']
        
        history['train_loss'].append(train_loss)
        history['test_auc'].append(test_auc)
        
        if test_auc > best_auc:
            best_auc = test_auc
            best_metrics = test_metrics
            best_state = model.state_dict().copy()
        
        scheduler.step(test_auc)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | "
                  f"Test AUC: {test_auc:.4f} | Best: {best_auc:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(best_state)
    
    print(f"\n{model_name} 最佳性能:")
    for metric, value in best_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return best_metrics, history, model


def plot_results(results, save_path='results_comparison.png'):
    """绘制所有模型的ROC曲线对比"""
    plt.figure(figsize=(12, 10))
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. ROC曲线
    ax1 = axes[0, 0]
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_prob'])
        auc_score = result['metrics']['auc']
        ax1.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})", linewidth=2)
    ax1.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC Curve Comparison", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. AUC对比柱状图
    ax2 = axes[0, 1]
    models = list(results.keys())
    aucs = [results[m]['metrics']['auc'] for m in models]
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    bars = ax2.bar(models, aucs, color=colors, edgecolor='black')
    ax2.set_ylabel("AUC Score", fontsize=12)
    ax2.set_title("AUC Score Comparison", fontsize=14)
    ax2.set_ylim([0.5, 1.0])
    for bar, auc in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. 多指标对比
    ax3 = axes[1, 0]
    metrics_names = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_names))
    width = 0.15
    
    for i, model_name in enumerate(models):
        values = [results[model_name]['metrics'][m] for m in metrics_names]
        ax3.bar(x + i*width, values, width, label=model_name, color=colors[i])
    
    ax3.set_ylabel("Score", fontsize=12)
    ax3.set_title("Multi-Metrics Comparison", fontsize=14)
    ax3.set_xticks(x + width * (len(models)-1) / 2)
    ax3.set_xticklabels([m.upper() for m in metrics_names])
    ax3.legend(fontsize=9)
    ax3.set_ylim([0, 1])
    
    # 4. 训练历史（以GIN为例）
    ax4 = axes[1, 1]
    if 'GIN' in results and 'history' in results['GIN']:
        history = results['GIN']['history']
        epochs = range(1, len(history['test_auc']) + 1)
        ax4.plot(epochs, history['test_auc'], 'b-', label='Test AUC', linewidth=2)
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("AUC Score", fontsize=12)
        ax4.set_title("GIN Training History", fontsize=14)
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n结果图已保存至: {save_path}")
    plt.close()


def save_results_table(results, save_path='results_table.csv'):
    """保存结果表格"""
    data = []
    for model_name, result in results.items():
        row = {'Model': model_name}
        row.update(result['metrics'])
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('auc', ascending=False)
    df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"结果表格已保存至: {save_path}")
    print("\n" + "="*80)
    print("实验结果汇总:")
    print("="*80)
    print(df.to_string(index=False))
    return df


# ==================== 4. 主函数 ====================

def main():
    """主实验函数"""
    print("="*80)
    print("GNN分子毒性预测综合实验")
    print("复现2017-2024年代表性方法")
    print("="*80)
    
    # 加载数据
    dataset, train_dataset, test_dataset = load_data()
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # 计算正样本权重
    pos_weight = compute_pos_weight(train_dataset, task_idx=0)
    
    # 获取所有模型
    models_dict = get_all_models(
        in_channels=dataset.num_features,
        hidden=128,
        dropout=0.3,
        num_classes=1
    )
    
    # 存储结果
    all_results = {}
    
    # 运行每个模型的实验
    for model_name, model in models_dict.items():
        metrics, history, trained_model = run_experiment(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            pos_weight=pos_weight,
            epochs=100,
            lr=0.001,
            task_idx=0
        )
        
        # 获取预测结果用于绘图
        _, y_true, y_prob = evaluate(trained_model, test_loader, task_idx=0)
        
        all_results[model_name] = {
            'metrics': metrics,
            'history': history,
            'y_true': y_true,
            'y_prob': y_prob,
            'model': trained_model
        }
    
    # 绘制结果
    plot_results(all_results, save_path='experiment_results.png')
    
    # 保存结果表格
    results_df = save_results_table(all_results, save_path='experiment_results.csv')
    
    # 保存详细结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {
        'timestamp': timestamp,
        'dataset': 'Tox21',
        'task': 'NR-AR (task 0)',
        'models': {
            name: {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in result['metrics'].items()}
            for name, result in all_results.items()
        }
    }
    
    with open(f'results_summary_{timestamp}.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)
    
    return all_results, results_df


if __name__ == '__main__':
    results, df = main()
