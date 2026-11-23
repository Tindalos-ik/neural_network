import time
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils import data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import AlexNet

ROOT = Path(__file__).resolve().parent
SEED = 42


def train_val_data_process(augment=False):
    # 验证集和无增强组使用相同的固定预处理。
    eval_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])
    train_transform = eval_transform
    if augment:
        # 数据增强：水平翻转、随机裁剪。
        # 在 28×28 原图上翻转、补边再裁剪，模拟最多 2 像素的平移。
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转，概率 0.5。
            transforms.RandomCrop(28, padding=2), # 随机裁剪，补边 2 像素。
            eval_transform,
        ])

    train_data = FashionMNIST(
        root=ROOT.parents[1] / "data",
        train=True,
        transform=train_transform,
        download=True,
    )
    # 固定划分种子，两组用完全相同的训练和验证样本。
    train_data, val_data = Data.random_split(
        train_data, [48_000, 12_000], generator=torch.Generator().manual_seed(SEED)
    )
    # random_split 的两个子集共享源数据。浅拷贝验证集的源数据对象，
    # 只更换 transform，不复制图片，也不影响训练集。
    val_source = copy(val_data.dataset)
    val_source.transform = eval_transform
    val_data = Data.Subset(val_source, val_data.indices)

    # 单独固定打乱顺序的种子，使两组每轮看到相同的样本顺序。
    train_dataloader = Data.DataLoader(
        dataset=train_data, 
        batch_size=256, 
        shuffle=True, 
        num_workers=2,
        persistent_workers=True, generator=torch.Generator().manual_seed(SEED),
    )
    val_dataloader = Data.DataLoader(
        dataset=val_data, batch_size=256, shuffle=False, num_workers=2,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs, checkpoint):
    # 有 GPU 就使用 GPU；将模型移到设备后再创建优化器。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 记录每轮的损失和准确率，训练完成后用于绘图。
    best_acc = -1.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.perf_counter()

    for epoch in range(num_epochs):
        train_loss = 0
        train_corrects = 0
        train_num = 0
        val_loss = 0
        val_corrects = 0
        val_num = 0

        # 训练：前向传播、反向传播、更新参数。
        model.train()
        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            optimizer.zero_grad()
            output = model(b_x)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()

            # 交叉熵返回批次均值，按样本数累加后再计算整轮均值。
            train_loss += loss.detach() * b_x.size(0)
            train_corrects += (output.argmax(dim=1) == b_y).sum()
            train_num += b_x.size(0)

        # 验证：关闭 Dropout，只做前向传播，不记录梯度。
        model.eval()
        with torch.inference_mode():
            for b_x, b_y in val_dataloader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                loss = criterion(output, b_y)
                val_loss += loss * b_x.size(0)
                val_corrects += (output.argmax(dim=1) == b_y).sum()
                val_num += b_x.size(0)

        train_loss_all.append(train_loss.item() / train_num)
        train_acc_all.append(train_corrects.item() / train_num)
        val_loss_all.append(val_loss.item() / val_num)
        val_acc_all.append(val_corrects.item() / val_num)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train loss {train_loss_all[-1]:.4f}, acc {train_acc_all[-1]:.2%} | "
            f"val loss {val_loss_all[-1]:.4f}, acc {val_acc_all[-1]:.2%}",
            flush=True,
        )

        # 保留验证准确率最高的一轮；权重副本放在 CPU，不占训练显存。
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = {
                name: tensor.to("cpu", copy=True) for name, tensor in model.state_dict().items()
            }

    # 两组分别保存最佳权重，供 model_test.py 在同一测试集上比较。
    torch.save(best_model_wts, checkpoint)
    print(f"Best val accuracy: {best_acc:.2%} | {time.perf_counter() - since:.1f}s | {checkpoint}")

    train_process = pd.DataFrame({
        "epoch": range(1, num_epochs + 1),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    })
    return train_process


def matplot_acc_loss(train_process, axes, group):
    # 左图比较损失，右图比较准确率；虚线是训练集，实线是验证集。
    epochs = train_process["epoch"]
    axes[0].plot(epochs, train_process["train_loss_all"], "--", label=f"{group} train")
    axes[0].plot(epochs, train_process["val_loss_all"], label=f"{group} val")
    axes[0].set(xlabel="Epoch", ylabel="loss")
    axes[1].plot(epochs, train_process["train_acc_all"], "--", label=f"{group} train")
    axes[1].plot(epochs, train_process["val_acc_all"], label=f"{group} val")
    axes[1].set(xlabel="Epoch", ylabel="acc")


if __name__ == "__main__":
    _, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for group in ("baseline", "augmented"):
        # 每组都重新初始化模型，并使用同一种子，让初始权重一致。
        torch.manual_seed(SEED)
        print(f"Training: {group}", flush=True)
        model = AlexNet()
        train_data, val_data = train_val_data_process(augment=(group == "augmented"))
        train_process = train_model_process(
            model, train_data, val_data, num_epochs=20, checkpoint=ROOT / f"{group}.pth"
        )
        matplot_acc_loss(train_process, axes, group)

    for axis in axes:
        axis.legend()
    plt.show()