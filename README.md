# PyTorch 神经网络实验

这个仓库用于学习 PyTorch 和实践不同类型的神经网络，涵盖数据处理、模型搭建、训练和评估。目前的实验从图像分类入手，后续会加入更多网络架构和应用实验。

## 实验

- **[PyTorch 入门](lab00/pytorch使用.ipynb)**：从张量和自动求导入手，用合成图像走完分类模型的训练与推理流程。
- **[CNN：CIFAR-10 图像分类](CNN/cnn.ipynb)**：搭建 `SimpleCNN`，识别飞机、汽车等 10 类彩色图像。
- **[ResNet-18：Fashion-MNIST 服饰识别](ResNet/Fashion-MNIST-clothsrecognition/服饰识别.md)**：手写残差块和 ResNet-18，识别 10 类灰度服饰图像，测试准确率为 **91.02%**，并用混淆矩阵分析误判。

## 环境

在仓库根目录运行 `uv sync --locked`。`cnn` 和 `yolo` 共用 `.venv`，在 VS Code 选择 `.venv\Scripts\python.exe` 作为解释器或 Notebook 内核。
