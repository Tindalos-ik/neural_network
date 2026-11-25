# PyTorch 神经网络实验


## 环境安装

本实验使用`uv`进行环境管理

```bash
uv sync --locked
```

`uv`会按照`uv.lock`创建`.venv`，创建好实验需要的虚拟环境

## 实验索引

### 图像分类（`cnn`）

- [TinyCNN](https://github.com/Tindalos-ik/neural_network/tree/cnn/lab00)：合成图像分类（PyTorch 入门）。
- [SimpleCNN](https://github.com/Tindalos-ik/neural_network/blob/cnn/CNN/cnn.ipynb)：CIFAR-10 图像分类。
- [LeNet-5](https://github.com/Tindalos-ik/neural_network/tree/cnn/LeNet/clothrecognition)、[AlexNet](https://github.com/Tindalos-ik/neural_network/tree/cnn/AlexNet/clothrecognition)、[VGG16](https://github.com/Tindalos-ik/neural_network/tree/cnn/VGG/clothrecognition)、[ResNet-18](https://github.com/Tindalos-ik/neural_network/tree/cnn/ResNet/Fashion-MNIST-clothsrecognition)：Fashion-MNIST 服饰分类。
- [GoogLeNet](https://github.com/Tindalos-ik/neural_network/tree/cnn/GoogleNet/cat-and-dog)：猫狗图像分类。

### 目标检测（`yolo`）

- [YOLOv1](https://github.com/Tindalos-ik/neural_network/blob/yolo/yolov1/YOLOv1.md)：目标检测原理学习（仅理论笔记）。
