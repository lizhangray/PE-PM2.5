import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs 
import pandas as pd
import numpy as np

from models.PM_Dual_Net import PM_Pie_Net
from common.dcp import DarkChannel, SaturationMap


def process_input(im):
    """将 PIL 图片处理为 Tensor 类型以适用于模型"""
    im = tfs.Resize((256, 256))(im)  # magic number: 256
    t_im = tfs.ToTensor()(im)[None, ::]

    dc = DarkChannel(im)
    sm = SaturationMap(im)
    t_ds = torch.cat([tfs.ToTensor()(dc), tfs.ToTensor()(sm)], dim=0)  # 拼接需要转换类型
    t_ds = t_ds[None, ::]

    return {"RGB": t_im, "DS": t_ds}


# 数据：测试图片 ------------------------------------------
Daytime_PM_MAX = 262.0  # Only for Beijing Dataset
Daytime_PM_MIN = 1.0

# TODO：1. test_imgs dir
test_imgs = r'./imgs'
img_dir = test_imgs + '/'
output_dir = img_dir
print("pred_dir:", output_dir)

# 模型 ---------------------------------------------------
# TODO：2. exp_dir
exp_dir = r"Checkpoints/"
model_dir = exp_dir + r"PM_DBResNet18.pk"  # ['PM_DBResNet18', 'PM_Pie_Net_swint']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
print("Load model: ", model_dir)

# TODO：3. backbone ['resnet18', 'mobilev2', 'swint']
net = PM_Pie_Net(Body='resnet18', DS_type="DS", pretrained=False)
net = nn.DataParallel(net)  # TODO: swint doesn't need this line but resnet18 needs !
net.load_state_dict(ckp['model'])
net.eval()

# 开始测试 ------------------------------------------------
all_pred = np.empty((0, 1), float)
img_names = []

for im in os.listdir(img_dir):
    if not im.endswith(('jpg', 'jpeg', 'png')):
        continue
    print(f'\r {im}', end='\n', flush=True)
    img_names += [im]

    haze = Image.open(img_dir + im)  # [H,W,C], 而 haze.size: W,H。顺序不一样
    haze1 = process_input(haze)

    with torch.no_grad():
        r = net(haze1)
        pred = r['PM']

    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    pred_nor = pred * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN
    print("Estimated PM2.5: ", pred_nor.flatten())
    all_pred = np.vstack((all_pred, pred_nor))

df = pd.DataFrame({'IMG': img_names, 'Preds': all_pred.flatten()})
df.to_excel(os.path.join(output_dir, 'results.xlsx'), index=False)
