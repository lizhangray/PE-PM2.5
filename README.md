# PE-PM2.5

## Image-based PM2.5 Estimation From Imbalanced Data Distribution Using Prior-Enhanced Neural Networks

## Dependencies
Python 3.8
PyTorch 1.12.1
opencv-python 4.7.0.72
matplotlib, scipy, opencv-python, pandas, scikit-image, scikit-learn, thop, torchsummary

## Run: Test

Put your test images in the folder 'imgs/', and then run the following scripts.

```shell
# change parameters in the test.py as you like
python test.py
```

## Run: Train

1. Train single-branch nets   
> python main.py --run_type=train --data_root=D:\workplace\project\PM2.5数据集\Heshan_imgset --net=PM_Single_Net --backbone=resnet18 --loss=L2 --epochs=150 --bs=16 --patch_size=256 --pretrained


2. Train dual-branch nets  
> python main.py  --run_type=train --data_root=D:\workplace\project\PM2.5数据集\Heshan_imgset --net=PM_Pie_Net --backbone=resnet18 --DS_type=DIS --loss=L2 --epochs=150 --bs=16 --patch_size=256 --pretrained


3. Evaluation & Test  
> python main.py --run_type=eval --data_root=D:\workplace\project\PM2.5数据集\Heshan_imgset --net=PM_Single_Net --backbone=resnet18 --model_dir=exp_0510-1629/trained_models/PM_Single_Net_resnet18.pk --bs=16 --patch_size=256


4. Data imbalance learning  
> python main.py --run_type=train --data_root=D:\workplace\project\PM2.5数据集\Heshan_imgset --net=PM_Pie_Net --backbone=resnet18 --DS_type=DIS --loss=L2 --balance --balance_type=LDS_bin --bin_width=10 --lds_clip=90 --epochs=150 --bs=16 --patch_size=256 --pretrained

## Checkpoints & Datasets

[Google](https://drive.google.com/drive/folders/1oE67ZCw2hnKZP_HewZEqq82bwNBchz3D?usp=drive_link)
Or
[Baidu](https://pan.baidu.com/s/1O_nMib7ljTl928aFQaee6A?pwd=k9vn)

## Useful links

[Image-based-PM2.5-Estimation](https://github.com/qing-xue/Image-based-PM2.5-Estimation)

## Citing 

The code is free for academic/research purpose. Please kindly cite our work in your publications if it helps your research.  

```BibTeX
@article{fang2024image,
  title={Image-Based PM2.5 Estimation From Imbalanced Data Distribution Using Prior-Enhanced Neural Networks},
  author={Fang, Xueqing and Li, Zhan and Yuan, Bin and Chen, Yihang},
  journal={IEEE Sensors Journal},
  volume={24},
  number={4},
  pages={4677--4693},
  year={2024},
  publisher={IEEE},
  link={10.1109/JSEN.2023.3343080}
}
```

## Example

<div align=center>
<center class="half">
    <img src="./imgs/P18_20.png" width="400" height="300"/>
    <img src="./imgs/p3_73.jpg" width="400" height="300"/>
</center>

|  Model   | img_1 (left)  |  img_2 (right)
|  ----  | ----  |  ----
| Ground Truth  | 20.00 |  73.00
| PE-ResNet18  | 23.16 |  64.03

<center class="half">
    <img src="./imgs/p18_80.jpg" width="400" height="300"/>
    <img src="./imgs/P8_152.png" width="400" height="300"/>
</center>

|  Model   | img_1 (left)  |  img_2 (right)
|  ----  | ----  |  ----
| Ground Truth  | 80.0 |  152.0
| PE-ResNet18  | 84.84 |  125.80

<center class="half">
    <img src="./imgs/Beijing_20190530051212642_PM=22.jpg" width="400" height="500"/>
    <img src="./imgs/Beijing_20191209132043910_PM=186.jpg" width="400" height="500"/>
</center>

|  Model   | img_1 (left)  |  img_2 (right)
|  ----  | ----  |  ----
| Ground Truth  | 22.0 |  186.0
| PE-ResNet18  | 25.85 |  184.47

</div>


[![Page Views Count](https://badges.toozhao.com/badges/01F0MPA6GQQXGBJSVKT85C4PKT/green.svg)](https://badges.toozhao.com/stats/01F0MPA6GQQXGBJSVKT85C4PKT "Get your own page views count badge on badges.toozhao.com")
