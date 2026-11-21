import argparse
from importlib import import_module
from pathlib import Path
import sys

import torch

from model import LeNet

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RESNET_EXPERIMENT = "Fashion-MNIST-clothsrecognition"


def main():
    parser = argparse.ArgumentParser(description="评估 LeNet 的 Fashion-MNIST 服饰识别表现")
    parser.add_argument("--weights", type=Path, default=ROOT / "model.pth",
                        help="LeNet 权重路径（默认读取实验目录的 model.pth）")
    parser.add_argument("--compare", action="store_true", help="用同一测试集对比已有的 ResNet-18")
    args = parser.parse_args()

    if not args.weights.is_file():
        parser.error(f"找不到 LeNet 权重：{args.weights}。训练结束后用 --weights 指定保存路径。")
    resnet_weights = REPO / "ResNet" / RESNET_EXPERIMENT / "model.pth"
    if args.compare and not resnet_weights.is_file():
        parser.error(f"找不到 ResNet-18 权重：{resnet_weights}")

    # 按模块全名加载，避免两个实验的 model.py、model_test.py 同名冲突。
    sys.path.insert(0, str(REPO))
    resnet_test = import_module(f"ResNet.{RESNET_EXPERIMENT}.model_test")
    loader = resnet_test.load_test_data()
    classes = loader.dataset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fashion-MNIST: {len(loader.dataset):,} test images ({device})")

    experiments = [("LeNet", LeNet(), args.weights)]
    if args.compare:
        experiments.append(("ResNet-18", resnet_test.ResNet18(resnet_test.Residual), resnet_weights))

    results = []
    for name, model, weights in experiments:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        matrix = resnet_test.evaluate(model, loader, device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results.append((matrix, params))

    print(f"{'Metric':<16}" + "".join(f"{name:>14}" for name, _, _ in experiments))
    print(f"{'Accuracy':<16}" + "".join(
        f"{matrix.diag().sum().item() / matrix.sum().item():>14.2%}" for matrix, _ in results
    ))
    print(f"{'Parameters':<16}" + "".join(f"{params:>14,}" for _, params in results))
    for index, name in enumerate(classes):
        print(f"{name:<16}" + "".join(
            f"{matrix[index, index].item() / matrix[index].sum().item():>14.2%}"
            for matrix, _ in results
        ))

    path = ROOT / "runs" / "confusion_matrix.png"
    resnet_test.save_confusion_matrix(results[0][0].numpy(), classes, path)
    print(f"LeNet confusion matrix: {path}")


if __name__ == "__main__":
    main()