from models.PM_Dual_Net import PM_Pie_Net
from models.PM_Single_Net import PM_Single_Net
from models.PM_Auxiliary_Net import PM_Single_DS_Net


def load_model(opt):
    name = opt.net
    if name == 'PM_Single_Net':
        return PM_Single_Net(Body=opt.backbone, pretrained=opt.pretrained)
    elif name == 'PM_Pie_Net':
        return PM_Pie_Net(Body=opt.backbone, DS_type=opt.DS_type, pretrained=opt.pretrained)
    elif name == 'PM_Single_DS_Net':
        return PM_Single_DS_Net(DS_type=opt.DS_type)