# AIDD_LearnProject

## 项目简介

本项目是一个用于学习和研究**人工智能药物发现（AIDD, Artificial Intelligence in Drug Discovery）**的学习仓库。主要包含分子表示学习、图神经网络（GNN）在分子性质预测中的应用，以及相关论文复现实践。

## 项目结构

```
├── MolCLR/              # 分子对比学习（Molecular Contrastive Learning）
│   ├── models/          # GCN和GIN模型实现
│   ├── dataset/         # 数据集处理模块
│   ├── utils/           # 工具函数
│   └── ckpt/            # 预训练模型权重
├── tox21_gin/           # Tox21毒性预测任务
│   ├── data/            # Tox21数据集
│   └── models.py        # GNN模型定义
├── 学习论文参考/         # 论文资源库
│   ├── 经典 papers/     # 经典GNN和AIDD论文（35篇）
│   └── 近两年 papers/   # 最新研究进展（40+篇）
└── 黎老师论文/           # 导师研究论文
```

## 主要内容

### 1. MolCLR 分子对比学习

基于论文《MolCLR: Molecular Contrastive Learning of Representations》实现，包含：
- GCN和GIN两种图神经网络架构
- NT-Xent对比损失函数
- 预训练和微调流程

### 2. Tox21 毒性预测

使用图神经网络进行分子毒性预测，支持：
- 多种GNN模型对比实验
- RDKit分子特征提取
- 模型性能可视化

### 3. 论文资源

收集了80+篇AIDD领域的经典和最新论文，涵盖：
- 图神经网络基础（GCN, GAT, GIN）
- 分子表示学习（MolCLR, GraphMAE）
- 药物发现应用（分子生成、性质预测）
- 综述与基准测试

## 环境依赖

- Python 3.8+
- PyTorch 1.8+
- PyTorch Geometric
- RDKit
- numpy, pandas, scikit-learn

## 使用方法

```bash
# 克隆项目
git clone <repository-url>
cd AIDD_LearnProject

# MolCLR预训练
cd MolCLR
python molclr.py --config config.yaml

# Tox21微调
cd ../tox21_gin
python molclr_finetune_tox21.py
```

## 学习路径

1. 阅读经典论文理解GNN基础
2. 运行MolCLR预训练代码
3. 完成Tox21毒性预测任务
4. 对比不同GNN模型性能

## 许可证

MIT License

---

*本项目用于个人学习和研究目的，记录AIDD领域的学习过程与论文复现实践。*