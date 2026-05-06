# GNN分子毒性预测综合实验

## 实验目标
复现2017-2024年图神经网络在分子性质预测领域的代表性方法，在Tox21数据集上进行综合对比实验。

## 已实现的方法

### 1. 经典方法 (2017-2019)
- **GCN** (Kipf & Welling, ICLR 2017): 图卷积网络
- **GAT** (Veličković et al., ICLR 2018): 图注意力网络
- **GIN** (Xu et al., ICLR 2019): 图同构网络

### 2. 改进方法 (2019-2021)
- **D-MPNN** (Yang et al., J. Chem. Inf. Model. 2019): 有向边消息传递网络
- **EGNN** (Satorras et al., ICML 2021): E(n)等变图神经网络

### 3. 注意力池化变体
- **GIN-AttnPool**: GIN + 注意力池化

## 文件说明

### 核心代码
- `models.py`: 所有GNN模型的实现
- `experiment.py`: 完整的训练和评估流程
- `simple_experiment.py`: 简化版实验（纯PyTorch，不依赖torch_geometric）
- `train.py`: 原始GIN训练脚本
- `gnn_compare.py`: 原始GCN/GIN/GAT对比脚本

### 数据集
- Tox21: 分子毒性预测数据集，包含12个毒性任务
- 使用MoleculeNet提供的标准数据集

## 运行方法

### 完整实验（需要torch_geometric）
```bash
python3 experiment.py
```

### 简化实验（纯PyTorch）
```bash
python3 simple_experiment.py
```

### 原始脚本
```bash
python3 train.py          # 训练单个GIN模型
python3 gnn_compare.py    # GCN/GIN/GAT对比
```

## 实验输出
- `experiment_results.png`: ROC曲线和性能对比图
- `experiment_results.csv`: 详细结果表格
- `results_summary_*.json`: 实验摘要

## 技术特点

### 数据处理
- 类别不平衡处理：正样本权重
- 缺失值处理：NaN掩码
- 数据划分：80%训练，20%测试

### 训练策略
- 优化器：AdamW + 权重衰减
- 学习率调度：ReduceLROnPlateau
- 早停机制：保存最佳模型

### 评估指标
- AUC-ROC: 主要指标
- Accuracy, Precision, Recall, F1

## 参考文献

1. Kipf & Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
2. Veličković et al. Graph Attention Networks. ICLR 2018.
3. Xu et al. How Powerful are Graph Neural Networks? ICLR 2019.
4. Yang et al. Analyzing Learned Molecular Representations for Property Prediction. J. Chem. Inf. Model. 2019.
5. Satorras et al. E(n) Equivariant Graph Neural Networks. ICML 2021.

## 待完成

- [ ] 运行完整实验并收集结果
- [ ] 添加更多基线方法（如MPNN）
- [ ] 实现自监督学习方法（MolCLR风格）
- [ ] 添加超参数搜索
- [ ] 生成详细的实验报告
