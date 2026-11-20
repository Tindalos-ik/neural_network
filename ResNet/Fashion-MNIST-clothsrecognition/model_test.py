from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import Residual, ResNet18

# 数据、权重和输出都相对于脚本所在目录，不依赖终端的工作目录。
ROOT = Path(__file__).resolve().parent


# 加载 Fashion-MNIST 官方测试集。
def load_test_data():
    dataset = FashionMNIST(
        root=ROOT / "data",
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )
    # 每批送入 128 张图片
    return Data.DataLoader(dataset, batch_size=128, shuffle=False)


# inference_mode() 关闭梯度记录；eval() 让 BatchNorm 使用保存的运行统计量。
@torch.inference_mode()
def evaluate(model, loader, device):
    model.to(device).eval()
    num_classes = len(loader.dataset.classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    # 行是真实类别，列是预测类别；每个样本只给对应的格子加 1。
    for images, labels in loader:
        # 输出形状是 [批大小, 10]。取每行最高分的类别，再转回 CPU 与真实标签一起统计。
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        # 把 (真实类别, 预测类别) 编成一维索引，用 bincount 批量累加后还原成矩阵。
        pairs = labels * num_classes + predictions
        matrix += torch.bincount(pairs, minlength=matrix.numel()).reshape_as(matrix)
    return matrix


# 保存各类别的预测计数图：对角线是正确预测，其他格子标出误判方向。
def save_confusion_matrix(matrix, classes, path):
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Images")
    ax.set(
        title="Fashion-MNIST confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    # 深色格用白字，浅色格用黑字，便于读出计数。
    midpoint = matrix.max() / 2
    for row, counts in enumerate(matrix):
        for col, count in enumerate(counts):
            ax.text(col, row, str(count), ha="center", va="center",
                    color="white" if count > midpoint else "black", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先创建同结构的模型，把权重读到 CPU；评估时再把整个模型移到运行设备。
    model = ResNet18(Residual)
    model.load_state_dict(torch.load(ROOT / "model.pth", map_location="cpu", weights_only=True))
    loader = load_test_data()
    classes = loader.dataset.classes
    matrix = evaluate(model, loader, device)

    # 每类准确率 = 对角线计数 ÷ 对应行的总数；总体准确率按全部样本计算。
    correct_by_class = matrix.diag().tolist()
    total_by_class = matrix.sum(dim=1).tolist()
    correct, total = sum(correct_by_class), sum(total_by_class)
    print(f"Test Accuracy: {correct / total:.4f} ({correct}/{total}, {device})")
    for name, hits, count in zip(classes, correct_by_class, total_by_class):
        print(f"{name:<12} {hits / count:6.2%} ({hits}/{count})")

    path = ROOT / "runs" / "confusion_matrix.png"
    save_confusion_matrix(matrix.numpy(), classes, path)
    print(f"Confusion matrix: {path.relative_to(ROOT)}")


