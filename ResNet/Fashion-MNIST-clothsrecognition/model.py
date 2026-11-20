import torch
from torch import nn
from torchsummary import summary # summary 是一个用于打印模型信息的库

# ResNet 的基本残差块：残差分支 + 捷径分支
class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1conv=False, stride=1):
        """
        input_channels: 输入通道数
        ouput_channels: 输出通道数
        use_1conv: 决定捷径分支是否使用 1x1 卷积核
        """
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()

        # 残差分支，第一层使用 3x3 的卷积核，stride 为 2
        # 第二层保持尺寸
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels) # 批量归一化
        self.conv2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channels)

        # 相加之前，需要将输入的尺寸与残差分支的尺寸对齐
        # 如果输入通道数与输出通道数不一致，需要使用 1x1 的卷积核进行变换
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, X):
        # 残差分支执行 Conv -> BN -> ReLU -> Conv -> BN ，最后一次 ReLU 放在残差分支与捷径分支相加之后
        y = self.conv1(X)
        y = self.bn1(y)
        y = self.ReLU(y)
        y = self.conv2(y)
        y = self.bn2(y)
        # 有投影时对齐形状，否则捷径直接传递原输入。
        x = self.conv3(X) if self.conv3 is not None else X
        y = self.ReLU(y + x)
        return y

# ResNet18 模型，适配 Fashion-MNIST 的单通道输入和 10 类输出
# 1 层入口卷积，8个残差块，每个残差块包含 2 个卷积层，1 层全连接层
class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        # 形状用 [B, C, H, W] 表示：B 是批大小，C 是通道数，H/W 是高/宽。
        # 以训练脚本缩放后的 224×224 灰度图为例，入口卷积和池化各减半一次：
        # [B, 1, 224, 224] → [B, 64, 112, 112] → [B, 64, 56, 56]。
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 第一个残差块的两层都保持 尺寸不变，可直接传递输入
        self.b2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, stride=1),
            Residual(64, 64, use_1conv=False, stride=1)
        )

        # 后三个都在第一层把高，宽减半并增加通道，第二层保持形状
        # 输出 [B, 128, 28, 28]
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2),
            Residual(128, 128, use_1conv=False, stride=1)
        )
        # 输出 [B, 256, 14, 14]
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, stride=2),
            Residual(256, 256, use_1conv=False, stride=1)
        )
        # 输出 [B, 512, 7, 7]
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, stride=2),
            Residual(512, 512, use_1conv=False, stride=1)
        )
        # 全局平均池化将每个通道的空间特征平均成一个值，然后展开得到 [B, 512]
        # 全连接层映射到 [B, 10]，返回类别分数
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == '__main__':
    # 单独运行打印模型信息
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(Residual).to(device)
    summary(model, (1, 224, 224))

