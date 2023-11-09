import torch.nn as nn


class SE_block(nn.Module):
    """ FFA 的 CA 实现和 RCAN 2018 大体相似"""
    def __init__(self, channel):
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 中间过程类似 SENet(Sequeeze and Excitation Net)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(),  # remove inplace=True
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        # 通道注意力 [1,64,1,1]，大大减少参数量
        y = self.ca(y)
        return x * y


class SE_block1D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE_block1D, self).__init__()
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.ca(x)
        return x * y
