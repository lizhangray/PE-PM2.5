import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary


class DS_Concat_layer(nn.Module):
    """输入为双通道或单通道"""
    def __init__(self, in_channel=2):
        super(DS_Concat_layer, self).__init__()
        Relu = nn.LeakyReLU(0.2, True)
        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 参照 Resnet 下采样，通道数减少
        condition_conv1 = nn.Conv2d(in_channel, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        condition_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        condition_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)

        conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        sift_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        sift_conv2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu, avgpool]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, depth):
        depth_condition = self.condition_conv(depth)
        sifted_feature = self.sift_conv(depth_condition)

        return sifted_feature


if __name__ == '__main__':
    net = DS_Concat_layer()
    summary(net, input_size=(2, 256, 256))

    x1 = torch.rand(1, 2, 256, 256)
    flops, params = profile(net, inputs=(x1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')