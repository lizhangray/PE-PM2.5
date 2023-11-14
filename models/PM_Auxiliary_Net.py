from thop import profile

from models.backbones import *
from torchsummary import summary


class DS_Aux_Branch(nn.Module):
    """输入为双通道或单通道"""
    def __init__(self, in_channel=2):
        super(DS_Aux_Branch, self).__init__()

        # 参照 Resnet 下采样，通道数减少
        condition_conv1 = nn.Conv2d(in_channel, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        condition_conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        condition_conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1)

        conditon_conv = [
            condition_conv1, nn.LeakyReLU(0.2, True),
            condition_conv2, nn.LeakyReLU(0.2, True),
            condition_conv3, nn.LeakyReLU(0.2, True)
        ]
        self.condition_conv = nn.Sequential(*conditon_conv)

        # TODO: 变量命名方式为 SFT 遗留，暂未更改
        sift_conv1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)
        sift_conv2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)
        sift_conv = [
            sift_conv1, nn.LeakyReLU(0.2, True),
            sift_conv2, nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d((1, 1))
        ]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, depth):
        depth_condition = self.condition_conv(depth)
        sifted_feature = self.sift_conv(depth_condition)

        return sifted_feature


class PM_Single_DS_Net(nn.Module):
    """以暗通道、饱和度图作为输入的单网络"""
    def __init__(self, DS_type='DC', useMobile=False):
        super(PM_Single_DS_Net, self).__init__()

        if DS_type == 'DC' or DS_type == 'SM':
            self.Pie = DS_Aux_Branch(in_channel=1)
        else:
            self.Pie = DS_Aux_Branch(in_channel=2)

        self.fcs = nn.Sequential(
            nn.Linear(512, 1)
        )

        # TODO: 替换为 backbone 试试
        if useMobile:
            m = MobileNetV2(True, del_type='avgpool')
            m.model[0][0] = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.Pie = m
            self.fcs = nn.Sequential(
                nn.Linear(1280, 1)
            )

    def forward(self, data):
        depth = data["DS"]
        y2 = self.Pie(depth)
        y2 = torch.flatten(y2, start_dim=1)
        y = self.fcs(y2)
        return {"PM": y}


if __name__ == '__main__':
    net = DS_Aux_Branch()
    summary(net, input_size=(2, 256, 256))

    x1 = torch.rand(1, 2, 256, 256)
    flops, params = profile(net, inputs=(x1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')