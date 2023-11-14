from models.PM_Auxiliary_Net import *
from models.SE_Blocks import SE_block1D


class Fusion_Block(nn.Module):
    def __init__(self, in_channels=1024):
        super(Fusion_Block, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, 512, 1, padding=0, bias=True)  # same as fully connected layer
        self.se = SE_block1D(512, reduction=8)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(512, 1),
        )

    def forward(self, x1, x2):
        cat = torch.cat([x1, x2], dim=1)
        y = self.fc1(cat)
        y = self.se(y)
        y = torch.flatten(y, 1)
        y = self.out(y)
        return y


class PM_Pie_Net(nn.Module):
    """Prior Information Enhanced (Pie)"""
    def __init__(self, Body=None, DS_type='DC', pretrained=True):
        super(PM_Pie_Net, self).__init__()
        self._get_Body(Body, pretrained)
        if DS_type == 'DC' or DS_type == 'SM':
            self.Pie = DS_Aux_Branch(in_channel=1)
        else:
            self.Pie = DS_Aux_Branch(in_channel=2)

    def forward(self, data):
        x, depth = data["RGB"], data["DS"]
        y1 = self.Body(x)
        y2 = self.Pie(depth)
        y  = self.Fusion(y1, y2)
        return {"PM": y, "Y1": y1, "Y2": y2}

    def _get_Body(self, name='resnet18', pretrained=True):
        if 'resnet18' == name:
            self.Body = Resnet18_17(pretrained, del_type='avgpool')
            self.Fusion = Fusion_Block(in_channels=512 + 512)  # 512 * 2
        elif 'mobilev2' == name:
            self.Body = MobileNetV2(pretrained, del_type='avgpool')
            self.Fusion = Fusion_Block(in_channels=512 + 1280)
        elif 'swint' == name:
            self.Body = SwinT_f(pretrained, del_type='fc')
            self.Fusion = Fusion_Block(in_channels=512 + 768)


if __name__ == '__main__':
    net = PM_Pie_Net(Body='resnet18', DS_type="DS")
    print(net)

    x1 = torch.rand(1, 3, 256, 256)
    x2 = torch.rand(1, 2, 256, 256)
    data = {'RGB': x1, 'DS': x2}

    flops, params = profile(net, inputs=(data,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

