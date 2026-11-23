from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import AlexNet

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def evaluate(model, loader, device):
    # eval() 关闭 Dropout，inference_mode() 禁用梯度记录。
    model.eval()
    num_classes = len(loader.dataset.classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        # 将「真实类别、预测类别」编码成一个编号，批量累加到混淆矩阵。
        pairs = labels * num_classes + predictions
        matrix += torch.bincount(pairs, minlength=matrix.numel()).reshape_as(matrix)
    return matrix


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 两组都使用同一份官方测试集及固定预处理。
    dataset = FashionMNIST(
        root=ROOT.parents[1] / "data",
        train=False,
        transform=transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()]),
        download=True,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = AlexNet().to(device)
    results = {}
    # 复用同一个模型结构，分别载入两个训练组的最佳权重。
    for group in ("baseline", "augmented"):
        checkpoint = ROOT / f"{group}.pth"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        matrix = evaluate(model, loader, device)
        correct = matrix.diag()
        # 对角线是判对的张数，每行总和是该类的真实样本数。
        results[group] = [correct.sum().item() / matrix.sum().item(),
                          *(correct / matrix.sum(dim=1)).tolist()]

    print(f"Fashion-MNIST: {len(dataset):,} test images ({device})")
    print(f"{'Metric':<16}{'Baseline':>12}{'Augmented':>12}{'Delta (pp)':>12}")
    # 用百分点表示「增强组准确率 − 无增强组准确率」。
    for label, baseline, augmented in zip(
        ["Accuracy", *dataset.classes], results["baseline"], results["augmented"]
    ):
        print(f"{label:<16}{baseline:>12.2%}{augmented:>12.2%}{(augmented - baseline) * 100:>+12.2f}")


if __name__ == "__main__":
    main()