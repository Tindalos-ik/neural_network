import argparse
import time
from copy import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import GoogLeNet, Inception

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parents[1] / "data" / "cat-and-dog"
PLOT_DIR = ROOT / "runs"
CHECKPOINT_PATH = ROOT / "best_googlenet.pth"
SEED = 42
VAL_FRACTION = 0.2


def load_dataset(data_dir, split):
    # ImageFolder 按子目录名排序生成标签，默认把图片读成 RGB。
    # ToTensor 把图片转成 [3, 224, 224] 的张量，像素值缩放到 [0, 1]。
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(data_dir / split, transform=transform)
    if len(dataset.classes) != 2:
        raise ValueError(f"{data_dir / split} 必须包含两个类别目录，实际为 {dataset.classes}")
    return dataset


def train_val_data_process(data_dir, batch_size):
    source = load_dataset(data_dir, "train")
    indices_by_class = [[] for _ in source.classes]
    for index, label in enumerate(source.targets):
        indices_by_class[label].append(index)

    # 用固定种子在每类中划出约 20% 做验证，test 不参与训练或选权重。
    generator = torch.Generator().manual_seed(SEED)
    train_indices, val_indices = [], []
    for name, indices in zip(source.classes, indices_by_class):
        if len(indices) < 2:
            raise ValueError(f"train/{name} 至少需要两张图片，才能划分训练集和验证集")
        order = torch.randperm(len(indices), generator=generator).tolist()
        val_count = max(1, round(len(indices) * VAL_FRACTION))
        val_indices.extend(indices[i] for i in order[:val_count])
        train_indices.extend(indices[i] for i in order[val_count:])

    # 浅拷贝只更换训练集的预处理，不复制样本列表；验证集沿用固定预处理。
    train_source = copy(source)
    train_source.transform = transforms.Compose([source.transform, transforms.RandomHorizontalFlip()])
    train_data = Subset(train_source, train_indices)
    val_data = Subset(source, val_indices)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=2, persistent_workers=True,
    )
    return train_loader, val_loader


def _run_epoch(model, loader, criterion, device, description, scaler, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0

    # 验证只做前向传播；训练才计算梯度和更新参数。
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels in tqdm(loader, desc=description, unit="batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            # CUDA 卷积等运算使用混合精度；交叉熵直接接收原始分数，不需要 Softmax。
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                # 缩放损失以减少 FP16 梯度下溢；CPU 上 scaler 禁用，按 FP32 更新。
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # 按样本数加权，最后一个不满的批次不会影响平均损失的权重。
            batch_size = labels.size(0)
            loss_sum += loss.detach() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum()
            total += batch_size

    return loss_sum.item() / total, correct.item() / total


def train_model_process(model, train_loader, val_loader, num_epochs, checkpoint, device):
    model.to(device)
    # 较大的 Adam 步长会让未归一化的 Inception 堆栈在训练初期出现损失尖峰。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_acc = -1.0
    history = []
    since = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        progress = f"Epoch {epoch}/{num_epochs}"
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, f"{progress} Train", scaler, optimizer,
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, device, f"{progress} Val", scaler,
        )
        history.append((train_loss, val_loss, train_acc, val_acc))
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.2%}",
            flush=True,
        )

        # 只用验证集选择最佳轮次；保存标签映射，供测试时核对。
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_loader.dataset.dataset.class_to_idx,
            }, checkpoint)

    print(f"Best val accuracy: {best_acc:.2%} | {time.perf_counter() - since:.1f}s | {checkpoint}")
    return history


def matplot_acc_loss(history, path):
    import matplotlib.pyplot as plt

    # 保存学习曲线，不阻塞命令行；横轴从第 1 轮开始。
    train_loss, val_loss, train_acc, val_acc = zip(*history)
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for axis, train, val, label in (
        (axes[0], train_loss, val_loss, "Loss"),
        (axes[1], train_acc, val_acc, "Accuracy"),
    ):
        axis.plot(epochs, train, label="Train")
        axis.plot(epochs, val, label="Validation")
        axis.set(xlabel="Epoch", ylabel=label)
        axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="训练 GoogLeNet 猫狗分类模型")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="包含 train 的数据目录")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH, help="最佳权重的保存路径")
    parser.add_argument("--output-dir", type=Path, default=PLOT_DIR, help="训练曲线图片的输出目录")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = train_val_data_process(args.data_dir, args.batch_size)
    precision = "AMP" if device.type == "cuda" else "FP32"
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | {device} ({precision})")
    print(f"Labels: {train_loader.dataset.dataset.class_to_idx}", flush=True)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = train_model_process(
        GoogLeNet(Inception), train_loader, val_loader, args.epochs,
        args.checkpoint, device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "training_curves.png"
    matplot_acc_loss(history, plot_path)
    print(f"Training curves: {plot_path}")


if __name__ == "__main__":
    main()