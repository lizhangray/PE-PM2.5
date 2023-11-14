from models.backbones import *
from models.PM_Auxiliary_Net import *


class PM_Single_Net(nn.Module):
    def __init__(self, Body=None, pretrained=True):
        """天空 + 地面块共同提取特征，直接输出"""
        super(PM_Single_Net, self).__init__()
        self._get_Body(Body, pretrained)

    def forward(self, data):
        x = data["RGB"]
        y = self.Body(x)
        # y = torch.squeeze(y, dim=[-1, -2])  # TODO: for swint
        y = self.fc(y)
        return {"PM": y}

    def _get_Body(self, name='vgg16', pretrained=True):
        if 'vgg16' == name:
            self.Body = VGG16_15(pretrained)  # Body output: 4096 feature vector
            self.fc = nn.Linear(4096, 1)      # output normalized PM2.5 value)
        elif 'resnet18' == name:
            self.Body = Resnet18_17(pretrained)  # Body output: 512 feature vector
            self.fc = nn.Linear(512, 1)
        elif 'mobilev2' == name:
            self.Body = MobileNetV2(pretrained)
            self.fc = nn.Linear(1280, 1)
        elif 'swint' == name:
            self.Body = SwinT_f(pretrained, del_type='fc')
            self.fc = nn.Linear(768, 1)


if __name__ == '__main__':
    net = PM_Single_Net(Body='mobilev2')
    print(net)

    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 2, 256, 256)
    data = {'RGB': x1, 'DS': x2}

    flops, params = profile(net, inputs=(data,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')