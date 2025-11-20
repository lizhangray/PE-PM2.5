import os
import sys
sys.path.append('.')
sys.path.append('..')

import random
import numpy as np
import PIL
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

from utils.dcp import DarkChannel, SaturationMap, CAPMap
from DIR.PM_utils import lds_weights, lds_weights_bin, cb_weights
import utils.data_load as data_load

random.seed(1024)


class PM_Direct_Dataset(data.Dataset):
    def __init__(self, Xys=None, isTrain=False, patch_size=256, DS_type='DS',
                 Y_weights=None, dataset_name="", DC_patch=15):
        """不切割天空、地面区域，直接 Resize 原始图像送进网络
          @Xys: tuples of (x, y)，其中 x 为包含完整路径的文件名, y 为归一化过的数据l
          @DS_type: 'DC', 'DS', 抽取先验信息暗通道和饱和度 Map
          @Y_weights: 样本权重
        """
        super(PM_Direct_Dataset, self).__init__()
        self.Xys = Xys
        self.isTrain = isTrain
        self.patch_size = patch_size
        self.DS_type = DS_type
        self.Y_weights = Y_weights
        self.dataset_name = dataset_name
        self.DC_patch = DC_patch  # 审稿人要求补充

    def __getitem__(self, index):
        """接口统一用字典"""
        img_name, PM = self.Xys[index]
        while not os.path.exists(img_name):
            index = random.randint(0, len(self.Xys) - 1)
            img_name, PM = self.Xys[index]

        im = Image.open(img_name).convert('RGB')
        im = self._augData(im)
        t_im = tfs.ToTensor()(im)     # [C, H, W]
        t_ds = self._get_fusions(im)

        t_PM = torch.from_numpy(np.array([PM], dtype=np.float32))
        t_weight = 1.0
        if self.Y_weights is not None:
            t_weight = torch.from_numpy(np.array([self.Y_weights[index]], dtype=np.float32))  # weight -> 1.

        # TODO: 不允许返回空值，会增加不必要的计算量
        return {"RGB": t_im, "DS": t_ds, "PM": t_PM, "Y_weight": t_weight, "IMG": img_name}

    def _augData(self, im):
        """不作 toTensor, 归一化等"""
        patch = self.patch_size

        # 鹤山数据集分辨率太大，天空占比太多，需做额外裁剪处理
        if "Heshan" in self.dataset_name:
            w, h = im.size
            box = (w // 8, h // 4, w - w // 8, h)  # [top, left, down, right]
            im = im.crop(box)

        im = tfs.Resize((patch, patch))(im)        # PIL 先 Resize 减少计算量

        if self.isTrain:
            im = tfs.RandomHorizontalFlip()(im)

        return im

    def _get_fusions(self, im):
        """求暗通道和饱和度 Map"""
        if self.DS_type is None:
            return None

        ds = torch.from_numpy(np.asarray(im))  # 先随机赋值
        if self.DS_type == 'DC':
            dc = DarkChannel(im, win=self.DC_patch)
            ds = tfs.ToTensor()(dc)
        elif self.DS_type == 'SM':
            sm = SaturationMap(im)
            ds = tfs.ToTensor()(sm)
        elif self.DS_type == 'DS':
            dc = DarkChannel(im, win=self.DC_patch)
            # print('DS DC_patch size:', self.DC_patch)  # for test
            sm = SaturationMap(im)
            ds = torch.cat([tfs.ToTensor()(dc), tfs.ToTensor()(sm)], dim=0)  # 拼接需要转换类型
        elif self.DS_type == 'DIS':
            dc = DarkChannel(im, win=self.DC_patch)
            # print('DIS DC_patch size:', self.DC_patch)  # for test
            # dc = Image.new('L', (256, 256))
            # print('zero DC')
            sm = SaturationMap(im)
            ism = PIL.ImageOps.invert(sm)  # 255 - i 取反
            # ism = Image.new('L', (256, 256))
            # print('zero ism')
            ds = torch.cat([tfs.ToTensor()(dc), tfs.ToTensor()(ism)], dim=0)
        else:
            return None

        return ds

    def __len__(self):
        return len(self.Xys)


class DataView:
    def __init__(self, opt):
        self.opt = opt
        self._getData()
        self.train_Xys_Weights = np.ones(len(self.df_train_Xys))  # same weight no balance
        if self.opt.balance:
            self._getWeights()
        self._getDataLoader()

    def _getData(self):
        (self.df_train_Xys, self.df_test_Xys), self.dataset_name = data_load.get_PM_Dataset(self.opt.data_root)
        print("Data Done. ---------------------------------------------------------------")

        # TODO: 求出训练集的最大最小值，测试集也以此为准做归一化；作为全局变量服务！归一化前 Re-weighted
        self.Daytime_PM_MIN, self.Daytime_PM_MAX = self.df_train_Xys['PM2.5'].min(), self.df_train_Xys['PM2.5'].max()

    def _getWeights(self):
        """计算权重要在数值归一化之前"""
        if 'LDS' == self.opt.balance_type:
            self.train_Xys_Weights = lds_weights(self.df_train_Xys['PM2.5'].values, lds_kernel='gaussian', lds_ks=5, lds_sigma=2)
        elif 'LDS_bin' == self.opt.balance_type:
            self.train_Xys_Weights = lds_weights_bin(
                self.df_train_Xys['PM2.5'].values, bin_width=self.opt.bin_width,
                clip=self.opt.lds_clip, lds_kernel='gaussian', lds_ks=self.opt.lds_ks, lds_sigma=2
            )
        elif 'resample' == self.opt.balance_type:
            # TODO: LDS resample ----------------
            # self.train_Xys_Weights = resample_weights(self.df_train_Xys['PM2.5'].values)
            print("resample' == self.opt.balance_type -----------------")
            self.train_Xys_Weights = lds_weights_bin(
                self.df_train_Xys['PM2.5'].values, bin_width=self.opt.bin_width,
                clip=self.opt.lds_clip, lds_kernel='gaussian', lds_ks=self.opt.lds_ks, lds_sigma=2
            )
        elif 'CBL' == self.opt.balance_type:
            self.train_Xys_Weights = cb_weights(self.df_train_Xys['PM2.5'].values)

    def _getDataLoader(self):
        # 归一化再还原时可能不是原来的整数值！
        self.df_train_Xys['PM2.5'] = self.df_train_Xys['PM2.5'].apply(
            lambda x: (x - self.Daytime_PM_MIN) / (self.Daytime_PM_MAX - self.Daytime_PM_MIN)
        )
        self.df_test_Xys['PM2.5'] = self.df_test_Xys['PM2.5'].apply(
            lambda x: (x - self.Daytime_PM_MIN) / (self.Daytime_PM_MAX - self.Daytime_PM_MIN)
        )  # 测试集归一化后可能会超出范围

        self.PM_Train_Dataset = PM_Direct_Dataset(
            Xys=self.df_train_Xys.values, isTrain=True, patch_size=self.opt.patch_size, DS_type=self.opt.DS_type,
            Y_weights=self.train_Xys_Weights, dataset_name=self.dataset_name, DC_patch=self.opt.DC_patch
        )
        self.PM_Test_Dataset = PM_Direct_Dataset(
            Xys=self.df_test_Xys.values , isTrain=False, patch_size=self.opt.patch_size, DS_type=self.opt.DS_type,
            dataset_name=self.dataset_name, DC_patch=self.opt.DC_patch
        )
        if self.opt.balance and self.opt.balance_type == 'resample':
            print('data.sampler.WeightedRandomSampler -----------------')
            self.PM_train_loader = DataLoader(
                dataset=self.PM_Train_Dataset,
                batch_size=self.opt.bs,
                sampler=data.sampler.WeightedRandomSampler(self.train_Xys_Weights, len(self.train_Xys_Weights)),
                num_workers=self.opt.num_workers
            )
        else:
            self.PM_train_loader = DataLoader(
                dataset=self.PM_Train_Dataset,
                batch_size=self.opt.bs,
                shuffle=True,
                num_workers=self.opt.num_workers
            )
        self.PM_test_loader = DataLoader(
            dataset=self.PM_Test_Dataset,
            batch_size=self.opt.bs,  # only 1 ？看 test 代码的计算方式，不在 batch 内计算则此参数无影响！
            shuffle=False,
            num_workers=self.opt.num_workers
        )


if __name__ == "__main__":
    pass