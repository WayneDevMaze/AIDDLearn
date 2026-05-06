# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # 神经网络功能模块
from torch_geometric.datasets import MoleculeNet  # 分子网络数据集
from torch_geometric.loader import DataLoader  # 数据加载器
from torch_geometric.nn import GINConv, global_add_pool  # GIN卷积层和全局池化
from sklearn.metrics import roc_auc_score  # ROC AUC评分指标
from torch.nn import Sequential, Linear, ReLU  # 序列模型、线性层和ReLU激活函数

# 设置设备：优先使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载Tox21数据集（用于分子毒性预测）
dataset = MoleculeNet(root='data', name='Tox21')
dataset = dataset.shuffle()  # 打乱数据集

# 划分训练集和测试集（8:2比例）
train_dataset = dataset[:int(len(dataset)*0.8)]
test_dataset = dataset[int(len(dataset)*0.8):]

# 创建数据加载器，用于批量处理数据
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练集使用打乱
test_loader = DataLoader(test_dataset, batch_size=64)  # 测试集不需要打乱

# 定义GIN（Graph Isomorphism Network）模型类
class GIN(torch.nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类构造函数

        # 第一层GIN卷积的MLP
        nn1 = Sequential(
            Linear(dataset.num_features, 64),  # 输入层：从节点特征维度到64维
            ReLU(),  # ReLU激活函数
            Linear(64, 64)  # 输出层：64维到64维
        )
        self.conv1 = GINConv(nn1)  # 创建第一层GIN卷积层

        # 第二层GIN卷积的MLP
        nn2 = Sequential(
            Linear(64, 64),  # 输入层：64维到64维
            ReLU(),  # ReLU激活函数
            Linear(64, 64)  # 输出层：64维到64维
        )
        self.conv2 = GINConv(nn2)  # 创建第二层GIN卷积层

        # 第三层GIN卷积的MLP
        nn3 = Sequential(
            Linear(64, 64),  # 输入层：64维到64维
            ReLU(),  # ReLU激活函数
            Linear(64, 64)  # 输出层：64维到64维
        )
        self.conv3 = GINConv(nn3)  # 创建第三层GIN卷积层

        # 全连接输出层：将64维特征映射到类别数
        self.lin = Linear(64, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 前向传播函数
        x = self.conv1(x, edge_index).relu()  # 第一层GIN卷积，后跟ReLU激活
        x = self.conv2(x, edge_index).relu()  # 第二层GIN卷积，后跟ReLU激活
        x = self.conv3(x, edge_index).relu()  # 第三层GIN卷积，后跟ReLU激活
        x = global_add_pool(x, batch)  # 全局加法池化，将图中所有节点特征相加
        return self.lin(x)  # 输出层预测

# 实例化模型并移至指定设备
model = GIN().to(device)
# 定义Adam优化器，学习率为0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train():
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 累计损失
    for data in train_loader:  # 遍历训练数据
        data = data.to(device)  # 将数据移至指定设备
        optimizer.zero_grad()  # 清零梯度

        # 模型前向传播
        out = model(data.x, data.edge_index, data.batch)
        y = data.y[:, 0].float()  # 提取第一个任务的标签并转换为浮点型

        # 创建掩码，过滤掉无效标签（NaN值）
        mask = y == y  # NaN != NaN，所以只有非NaN值会为True
        # 计算二分类交叉熵损失，只考虑有效标签
        loss = F.binary_cross_entropy_with_logits(
            out[:, 0][mask],  # 模型预测的第一个任务输出
            y[mask]  # 对应的真实标签
        )

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        total_loss += loss.item()  # 累加损失值

    return total_loss / len(train_loader)  # 返回平均损失

# 测试函数，使用@torch.no_grad()装饰器禁用梯度计算
@torch.no_grad()
def test(loader):
    model.eval()  # 设置模型为评估模式

    ys, preds = [], []  # 存储真实标签和预测值

    for data in loader:  # 遍历测试数据
        data = data.to(device)  # 将数据移至指定设备
        out = model(data.x, data.edge_index, data.batch)  # 模型前向传播

        y = data.y[:, 0]  # 提取第一个任务的标签
        mask = y == y  # 创建掩码，过滤掉无效标签

        ys.append(y[mask].cpu())  # 存储真实标签
        preds.append(out[:, 0][mask].cpu())  # 存储预测值

    y = torch.cat(ys).numpy()  # 拼接所有真实标签
    pred = torch.cat(preds).numpy()  # 拼接所有预测值

    return roc_auc_score(y, pred)  # 计算ROC AUC评分

# 训练60个epoch
for epoch in range(60):
    loss = train()  # 训练一个epoch并获取损失
    auc = test(test_loader)  # 在测试集上评估性能

    # 打印epoch、损失和AUC值
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}')