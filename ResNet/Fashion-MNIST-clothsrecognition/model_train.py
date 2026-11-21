import copy
import time
from pathlib import Path

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import Residual, ResNet18
import torch.nn as nn
import pandas as pd

batch_size = 64 

# 从官方训练集划分训练与验证数据，官方测试集保留给 model_test.py
def train_val_data_process():
    # 原图是 28*28，这里将图片大小调整为 224*224
    # 将像素转成 0-1 之间的小数，单张形状为 [1, 224, 224]
    train_data = FashionMNIST(
        root=Path(__file__).resolve().parents[2] / "data",
        train=True, 
        transform=transforms.ToTensor(), # ToTensor() 将图片转成张量
        download=True # 没有会自动下载
    )
    # 固定随机种子
    generator = torch.Generator().manual_seed(42)

    train_data, val_data = Data.random_split(
        train_data,
        [48_000, 12_000],
        generator=generator,
    )
    train_dataloader = Data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, # 打乱数据
        num_workers=2, # 多线程读取数据
        persistent_workers=True # 让worker进程在数据加载完成后保持开启状态
    )
    val_dataloader = Data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    return train_dataloader, val_dataloader


# 每轮先用数据集更新参数，再用验证集评估，验证准确率用于选择权重
def train_model_process(model, train_dataloader, val_dataloader, num_epochs=10):
    # 设定训练设别
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001
    )
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型放入到训练设备中
    model = model.to(device)
    # 保存独立的参数和缓冲区快照
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    best_acc = 0.0 # 最高准确度
    train_loss_list = [] # 训练集损失
    val_loss_list = [] # 验证集损失
    train_acc_list = [] # 训练集准确度
    val_acc_list = [] # 验证集准确度
    since = time.time() # 记录开始时间

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0 # 训练集预测正确的样本数
        val_correct = 0
        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据放入到训练设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train() # 开启训练模式，训练模式下 BatchNorm 使用当前批次的统计量，并更新运行统计量

            output = model(b_x) # 前向传播
            pre_lab = torch.argmax(output, dim=1) # 取出预测值最大的索引
            loss = criterion(output, b_y) # 计算损失

            # pytorch 默认累计梯度，反向传播之前清掉上一个批次留下的梯度
            optimizer.zero_grad()
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            train_loss += loss.item() * b_x.size(0) # 累计损失
            train_correct += torch.sum(pre_lab == b_y) # 累计预测正确的样本数
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval() # 开启评估模式，评估模式下 BatchNorm 使用运行统计量，不更新运行统计量

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_correct += torch.sum(pre_lab == b_y)
            val_num += b_x.size(0)

        # 计算平均损失和准确度
        train_loss_list.append(train_loss / train_num)
        val_loss_list.append(val_loss / val_num)
        train_acc = train_correct.item() / train_num
        val_acc = val_correct.item() / val_num
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f'Train Loss: {train_loss / train_num:.4f} Val Loss: {val_loss / val_num:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            # 深拷贝当前的参数和缓冲区
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(f'Epoch {epoch}/{num_epochs - 1} time use: {time_use:.4f} s')
        print('-'*10)

    model.load_state_dict(best_model_wts) # 加载最佳参数
    torch.save(model.state_dict(), 'model.pth') # 保存模型

    # 记录训练过程，供后续绘制学习曲线
    train_process = pd.DataFrame(
        data = {
            "epoch": range(num_epochs),
            "train_loss": train_loss_list,
            "val_loss": val_loss_list,
            "train_acc": train_acc_list,
            "val_acc": val_acc_list
        }
    )

    return train_process

def matplot_acc_loss(train_process):
    # 左图比较训练/验证损失，右图比较训练/验证准确率。
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], "ro-", label="Train loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc"], "ro-", label="Train acc")
    plt.plot(train_process["epoch"], train_process["val_acc"], "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 加载需要的模型
    ResNet18 = ResNet18(Residual)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(ResNet18, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
