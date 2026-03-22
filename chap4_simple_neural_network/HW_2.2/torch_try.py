# 如果用.ipynb那个核老是没法正确加载，麻烦得很
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

device = "cuda"

def targetFunc(x):
    """目标函数：x²的平方函数"""
    return x * x

# 生成实际训练数据（更密集的采样）
x = np.arange(0, 5 * np.pi, 0.001)  # 0-5π范围内生成训练数据
y = [targetFunc(i) for i in x]

# 数据集划分（训练：验证：测试 = 7:1:2）
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=1)

class CurveDataset(Dataset):
    """自定义曲线数据集类，继承自PyTorch的Dataset"""
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.tensor(X)  # 输入特征转换为张量
        self.y = torch.tensor(y)  # 目标值转换为张量

    def __getitem__(self, idx):
        """获取单个样本，需要返回(input, target)对"""
        return self.X[idx], self.y[idx]

    def __len__(self):
        """返回数据集总长度"""
        return len(self.y)
    

train_dataset = CurveDataset(X_train, y_train)
val_dataset = CurveDataset(X_val, y_val)
test_dataset = CurveDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 神经网络模型定义
class MLP(nn.Module):
    """两层全连接网络（输入层->隐藏层->输出层）"""
    def __init__(self, in_features=1, out_features=1):
        super().__init__()
        # 第一层：输入维度1，输出维度10
        self.FC1 = nn.Linear(in_features=in_features, out_features=10)
        # 使用ReLU激活函数   引入非线性能力
        self.relu = nn.ReLU()
        # 第二层：输入维度10，输出维度1
        self.FC2 = nn.Linear(in_features=10, out_features=out_features)

    def forward(self, x):
        x = self.FC1(x)
        x = self.relu(x)
        outputs = self.FC2(x)
        return outputs


model = MLP()


# 权重初始化函数
def weights_init(m):
    """自定义权重初始化：使用正态分布初始化全连接层权重"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 均值为0，标准差为0.02

model.apply(weights_init)
loss_fn = nn.L1Loss()


# 验证函数
def val(model, dataloader, loss_fn, device='cpu'):
    """验证阶段函数"""
    model.to(device)
    model.eval()  # 设置评估模式（关闭dropout等）
    with torch.inference_mode():  # 关闭梯度计算
        rec_loss = 0
        for X, y in dataloader:
            # 数据预处理：增加通道维度并转换类型
            X = X.to(device).unsqueeze(-1).float()  # [batch] -> [batch, 1]
            y = y.to(device).unsqueeze(-1).float()
            # 前向计算
            logits = model(X)
            loss = loss_fn(logits, y)
            rec_loss += loss
        print(f"Validation loss:{rec_loss / len(dataloader)}")

# 训练函数
def training(model, dataloader, val_dataloader, loss_fn, lr=0.001, epochs=50, device='cpu', verbose_epoch=10):
    """训练流程主函数"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器
    for epoch in tqdm(range(epochs)):
        model.train()   # 设置训练模式
        rec_loss = 0
        for X, y in dataloader:
            # 数据预处理
            X = X.to(device).unsqueeze(-1).float()
            y = y.to(device).unsqueeze(-1).float()

            # 前向传播
            logits = model(X)
            loss = loss_fn(logits, y)

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

            rec_loss += loss

        if epoch % verbose_epoch == 0:
            print(f"Epoch{epoch}\tLoss{rec_loss / len(dataloader)}")
            val(model, val_dataloader, loss_fn, device)  # 执行验证


# 测试函数
def test(model, ranger, steper, loss_fn, device='cpu'):
    """测试与可视化函数"""
    model.to(device)
    model.eval()  # 设置评估模式（关闭dropout等）
    x = []
    y_pred = []
    with torch.inference_mode():
        rec_loss = 0
        # 生成连续测试点
        for X in range(ranger[0], ranger[1], steper):
            X = torch.tensor(X).to(device).unsqueeze(-1).float()
            y = torch.tensor([targetFunc(i) for i in X]).to(device).unsqueeze(-1).float()

            logits = model(X)
            loss = loss_fn(logits, y)
            rec_loss += loss

            x.extend(X.unsqueeze(1))
            y_pred.extend(logits.unsqueeze(1))
        print(f"Test loss:{rec_loss * steper / (ranger[1] - ranger[0])}")

    x = [i.cpu().numpy() for i in x]
    y_pred = [i.cpu().numpy() for i in y_pred]

    plt.plot(x, [targetFunc(i) for i in x])
    plt.plot(x, y_pred)
    plt.legend(["Ground truth", "Prediction"])
    plt.show()


if __name__ == '__main__':
    training(model, train_dataloader, val_dataloader, loss_fn, lr=0.001, epochs=100, device=device, verbose_epoch=10)
    test(model, [0, 30], 1, loss_fn, device)