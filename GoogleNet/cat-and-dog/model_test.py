import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import GoogLeNet, Inception
from model_train import CHECKPOINT_PATH, DATA_DIR, load_dataset


@torch.inference_mode()
def evaluate(model, loader, device):
    # 切换到评估模式，只做推理，不记录梯度。
    model.eval()
    num_classes = len(loader.dataset.classes)
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for images, labels in loader:
        # argmax 直接取分数较高的类别，结果转回 CPU 后和真实标签一起统计。
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        # 行是真实类别，列是预测类别；先编码成一维编号，再批量计数。
        pairs = labels * num_classes + predictions
        matrix += torch.bincount(pairs, minlength=matrix.numel()).reshape_as(matrix)
    return matrix


def main():
    parser = argparse.ArgumentParser(description="在独立测试集上评估 GoogLeNet 猫狗分类模型")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="包含 test 的数据目录")
    parser.add_argument(
        "--checkpoint", type=Path, default=CHECKPOINT_PATH, help="训练保存的最佳权重",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # 测试使用与验证集相同的固定预处理，不打乱样本，也不做数据增强。
    dataset = load_dataset(args.data_dir, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先在 CPU 读取权重，使 CUDA 训练的模型也能在 CPU 上测试。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != dataset.class_to_idx:
        raise ValueError("test 的类别目录名必须与训练时一致")
    model = GoogLeNet(Inception)
    model.load_state_dict(checkpoint["model_state_dict"])
    matrix = evaluate(model.to(device), loader, device)

    # 对角线是判对的张数；每行总和是对应真实类别的样本数。
    correct = matrix.trace().item()
    total = len(dataset)
    print(f"Test Accuracy: {correct / total:.2%} ({correct}/{total}, {device})")
    print(f"{'True / Predicted':<18}", *(f"{name:>12}" for name in dataset.classes), f"{'Accuracy':>12}")
    for index, (name, row) in enumerate(zip(dataset.classes, matrix.tolist())):
        print(f"{name:<18}", *(f"{count:>12}" for count in row), f"{row[index] / sum(row):>12.2%}")


if __name__ == "__main__":
    main()