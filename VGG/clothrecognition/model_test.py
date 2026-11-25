import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

if __package__:
    from .model import VGG16
    from .model_train import CHECKPOINT, ROOT, load_fashion_mnist
else:
    from model import VGG16
    from model_train import CHECKPOINT, ROOT, load_fashion_mnist


def load_test_data(batch_size=16):
    # 官方测试集不参与训练和选权重，预处理与训练、验证一致。
    return DataLoader(load_fashion_mnist(train=False), batch_size=batch_size, shuffle=False)


@torch.inference_mode()
def evaluate_checkpoint(loader, device, checkpoint=CHECKPOINT):
    # 先在 CPU 读取权重，再把模型移到运行设备；也能在 CPU 测试 GPU 训练的权重。
    model = VGG16()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    num_classes = len(loader.dataset.classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            scores = model(images)
        predictions = scores.argmax(dim=1).cpu()
        # 将（真实类别，预测类别）编码成一维索引，累加后还原成混淆矩阵。
        pairs = labels * num_classes + predictions
        matrix += torch.bincount(pairs, minlength=matrix.numel()).reshape_as(matrix)
    return matrix


def print_test_results(matrix, classes):
    # 矩阵对角线是判对的数量，每行总数是该真实类别的样本数。
    correct_by_class = matrix.diag().tolist()
    total_by_class = matrix.sum(dim=1).tolist()
    correct, total = sum(correct_by_class), sum(total_by_class)
    print(f"Test accuracy: {correct / total:.2%} ({correct}/{total})")
    for name, hits, count in zip(classes, correct_by_class, total_by_class):
        print(f"{name:<12} {hits / count:>7.2%} ({hits}/{count})")


def save_confusion_matrix(matrix, classes, path):
    fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Images")
    ax.set(
        title="VGG16 on Fashion-MNIST",
        xlabel="Predicted label",
        ylabel="True label",
    )
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    midpoint = matrix.max().item() / 2
    for row, counts in enumerate(matrix.tolist()):
        for col, count in enumerate(counts):
            ax.text(col, row, str(count), ha="center", va="center",
                    color="white" if count > midpoint else "black", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing VGG16 on {device}", flush=True)
    loader = load_test_data()
    classes = loader.dataset.classes
    matrix = evaluate_checkpoint(loader, device)
    print_test_results(matrix, classes)
    path = ROOT / "runs" / "confusion_matrix.png"
    save_confusion_matrix(matrix, classes, path)
    print(f"Confusion matrix: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()