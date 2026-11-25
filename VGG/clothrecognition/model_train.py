import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

if __package__:
    from .model import IMAGE_SIZE, VGG16
else:
    from model import IMAGE_SIZE, VGG16

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parents[1] / "data"
CHECKPOINT = ROOT / "model.pth"
SEED = 42


def load_fashion_mnist(train):
    # 保留灰度单通道，仅缩放为 64×64 并转为张量。
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return FashionMNIST(root=DATA_ROOT, train=train, transform=transform, download=True)


def train_val_data_process(batch_size=512):
    # 只从官方训练集划分 48,000 张训练图和 12,000 张验证图。
    train_data, val_data = random_split(
        load_fashion_mnist(train=True),
        [48_000, 12_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, scaler, optimizer=None):
    # 传入优化器时才更新参数；验证只做前向，不记录梯度。
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            # CUDA 卷积使用 FP16，交叉熵由 autocast 保持在 FP32；CPU 仍用 FP32。
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                scores = model(images)
                loss = criterion(scores, labels)
            if training:
                # 缩放损失以减轻 FP16 梯度下溢；更新前由 scaler 还原梯度。
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # 交叉熵返回批次均值，乘以样本数后累加，避免最后一小批权重过大。
            loss_sum += loss.detach() * labels.size(0)
            correct += (scores.argmax(dim=1) == labels).sum()
            total += labels.size(0)

    return loss_sum.item() / total, correct.item() / total


def train_model_process(model, train_loader, val_loader, device, num_epochs=20, checkpoint=CHECKPOINT):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_history, val_history = [], []
    best_acc = -1.0
    started = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        # 按批次显示当前轮的训练和验证进度，结束后只保留指标汇总。
        epoch_label = f"Epoch {epoch}/{num_epochs}"
        with tqdm(train_loader, desc=f"{epoch_label} Train", unit="batch", leave=False) as progress:
            train_loss, train_acc = run_epoch(model, progress, criterion, device, scaler, optimizer)
        with tqdm(val_loader, desc=f"{epoch_label} Val", unit="batch", leave=False) as progress:
            val_loss, val_acc = run_epoch(model, progress, criterion, device, scaler)
        train_history.append((train_loss, train_acc))
        val_history.append((val_loss, val_acc))
        print(
            f"{epoch_label} | train loss {train_loss:.4f}, acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.2%}",
            flush=True,
        )

        # 按验证准确率选权重，不用测试集挑选模型；同分时保留较早的一轮。
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

    print(f"Best val accuracy: {best_acc:.2%} | {time.perf_counter() - started:.1f}s | {checkpoint}")
    return train_history, val_history


def save_learning_curves(train_history, val_history, path):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    epochs = range(1, len(train_history) + 1)
    for label, history in (("Train", train_history), ("Validation", val_history)):
        losses, accuracies = zip(*history)
        loss_ax.plot(epochs, losses, label=label)
        acc_ax.plot(epochs, accuracies, label=label)

    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    acc_ax.set(xlabel="Epoch", ylabel="Accuracy")
    loss_ax.legend()
    acc_ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = "FP16 AMP" if device.type == "cuda" else "FP32"
    print(f"Training VGG16 on {device} ({precision})", flush=True)
    train_loader, val_loader = train_val_data_process()
    train_history, val_history = train_model_process(VGG16(), train_loader, val_loader, device)
    path = ROOT / "runs" / "learning_curves.png"
    save_learning_curves(train_history, val_history, path)
    print(f"Learning curves: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()