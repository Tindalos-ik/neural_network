import torch
from torch import nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 第一层卷积：输入通道1，输出通道6，卷积核大小5，padding=2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding = 2)
        self.sig = nn.Sigmoid()
        # 第一层池化：池化核大小2，步长2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二层卷积：输入通道6，输出通道16，卷积核大小5
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 第二层池化：池化核大小2，步长2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        # 展开层：将4*4*16的矩阵展开成1*400的矩阵
        self.flatten = nn.Flatten()
        # 全连接层1：输入400，输出120
        self.f5 = nn.Linear(400, 120)
        # 全连接层2：输入120，输出84
        self.f6 = nn.Linear(120, 84)
        # 全连接层3：输入84，输出10
        self.f7 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.sig(self.f5(x))
        x = self.sig(self.f6(x))
        x = self.f7(x)
        return x

if __name__ == '__main__':
    devcie = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(devcie)
    model = LeNet().to(devcie)
    print(summary(model, (1, 28, 28)))