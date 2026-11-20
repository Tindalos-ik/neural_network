# PyTorch 神经网络实验

这个仓库用于学习 PyTorch 和实践不同类型的神经网络。`master` 是实验索引，各专题放在独立分支。

## 实验索引

- **[卷积神经网络（`cnn`）](https://github.com/Tindalos-ik/neural_network/tree/cnn)**：PyTorch 入门、CIFAR-10 图像分类、ResNet-18 服饰识别。

本地可用 `git switch cnn` 切换到卷积神经网络专题。

## 安装环境

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在仓库根目录执行：

```sh
uv sync --locked
```

uv 会按 `uv.lock` 创建并同步 Python 3.11 的 `.venv`。Windows 和 Linux 使用 CUDA 12.8 版 PyTorch。

## 主要依赖

- 深度学习：`torch`、`torchvision`、`torchsummary`、`thop`
- 图像处理与评估：`opencv-python`、`Pillow`、`pycocotools`
- 数值计算与绘图：`numpy`、`scipy`、`pandas`、`matplotlib`、`seaborn`
- Notebook 与训练日志：`ipykernel`、`tensorboard`、`tqdm`
- 其他：`PyYAML`、`requests`、`setuptools`
