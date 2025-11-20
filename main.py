import time
import math
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch import optim
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import random

import warnings
warnings.filterwarnings('ignore')

from option import opt, model_name, dir_trained_models, dir_numpy_files, dir_logs, make_dirs
from datasets import DataView
from utils.metrics import get_metrics
from models.load import load_model
from DIR.loss import weighted_mse_loss, weighted_focal_mse_loss, BMCLoss, weighted_l1_loss, weighted_huber_loss


# 全局变量 ---------------------------------------------------------------------
start_time = time.time()


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False  # 效率，如果注释掉，数据的载入仍然是相同的，但计算的结果会有很微小的差别
seed_torch()


def print_and_log(str_, console=True):
    if console:
        print(str_)
    with open(os.path.join(dir_logs, 'log.txt'), 'a') as f:
        f.write(str(str_) + '\n')


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    """
      we adopt the co-sine annealing strategy (He et al. 2019) to adjust the learning rate
      from the initial value to 0 by following the cosine function.
    """
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion):
    losses, loss_PMs, loss_Emds = [], [], []
    rmses, maes = [], []
    min_rmse, min_mae = float("inf"), float("inf")
    start_epoch = 0
    Early_Stop_Cnt = 0

    steps = len(loader_train)
    T = opt.epochs * steps

    for epoch in range(start_epoch, opt.epochs):

        net.train()

        for i, datas in enumerate(loader_train):
            if not opt.no_lr_sche:
                lr = lr_schedule_cosdecay(epoch * steps + i + 1, T)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            for k, v in datas.items():
                if v is not None and not isinstance(v, list):
                    datas[k] = datas[k].to(opt.device)

            y = datas["PM"]

            outputs = net(datas)  # 改写 forward 函数根据 key 自取
            out_PM = outputs["PM"]

            if opt.balance and opt.balance_type != 'resample':
                loss_pm = 1.0 * criterion(out_PM, y, weights=datas["Y_weight"])
            else:
                loss_pm = 1.0 * criterion(out_PM, y)  # weights

            loss_PMs.append(loss_pm.item())

            loss_all = loss_pm
            loss_all.backward()
            losses.append(loss_all.item())

            optim.step()
            optim.zero_grad()

            str_train_process = f'\rtrain total loss: {loss_all.item():.5f}' \
                                f'|epoch :{epoch}/{opt.epochs}|step :{i+1}/{steps}|lr :{lr:.7f} ' \
                                f'|time_used :{(time.time() - start_time) / 60 :.1f}'
            print(str_train_process, end='', flush=True)

        # 每 5 个 epoch 评估一次, 258 steps * 5
        if epoch % 5 == 0:
            rmse_val, mae_val = test(net, loader_test)
            print(f'\nepoch :{epoch} |rmse:{rmse_val:.4f} |mae:{mae_val:.4f}')

            rmses.append(rmse_val)
            maes.append(mae_val)
            Early_Stop_Cnt = Early_Stop_Cnt + 1  # 验证一次则加 1

            if rmse_val < min_rmse and mae_val < min_mae:  # not or mae
                Early_Stop_Cnt = 0  # 有最优模型保存则置 0
                min_rmse = min(min_rmse, rmse_val)
                min_mae = min(min_mae, mae_val)
                torch.save({
                    'epoch': epoch,
                    'min_rmse': min_rmse,
                    'rmses': rmses,
                    'min_mae': min_mae,
                    'maes': maes,
                    'losses': losses,
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(dir_trained_models, model_name) + ".pk")
                print_and_log(f'\n model saved at epoch :{epoch}| min_rmse:{min_rmse:.4f}| min_mae:{min_mae:.4f}')

        if Early_Stop_Cnt > 10:
            break  # 连续 5 次验证无更优则提前终止训练

    # 记录全局的而非仅仅是最优的
    np.save(os.path.join(dir_numpy_files, f'{model_name}_losses.npy'), losses)
    np.save(os.path.join(dir_numpy_files, f'{model_name}_loss_PMs.npy'), loss_PMs)
    np.save(os.path.join(dir_numpy_files, f'{model_name}_rmses.npy'), rmses)


def test(net, loader_test, show_info=False, save_preds=False):
    net.eval()
    torch.cuda.empty_cache()

    # rmses 不能像 PSNR 那样内部计算、累加，外部求均值
    all_pred = np.empty((0, 1), float)
    all_target = np.empty((0, 1), float)
    img_names = []

    for i, datas in enumerate(loader_test):
        for k, v in datas.items():
            if v is not None and not isinstance(v, list):
                datas[k] = datas[k].to(opt.device)
        targets = datas["PM"]
        img_names += datas["IMG"]

        with torch.no_grad():
            outputs = net(datas)

        pred = torch.mean(outputs["PM"], dim=-1, keepdim=True)  # 可能有一维或多维
        pred = torch.clamp(pred, min=0.0, max=1.0)
        pred = pred.cpu().numpy()
        targets = targets.cpu().numpy()
        pred_nor = pred * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN
        targets_nor = targets * (Daytime_PM_MAX - Daytime_PM_MIN) + Daytime_PM_MIN

        if show_info:
            print('[batch]', i, '-' * 50)
            print('predicts:', pred_nor.flatten())
            print('targets :', targets_nor.flatten())

        all_pred = np.vstack((all_pred, pred_nor))
        all_target = np.vstack((all_target, targets_nor))

    if save_preds:
        np.save(os.path.join(dir_numpy_files, 'all_pred.npy'), all_pred)
        np.save(os.path.join(dir_numpy_files, 'all_target.npy'), all_target)
        df = pd.DataFrame({
            'IMG': img_names, 'Preds': all_pred.flatten(),
            'Targets': all_target.flatten(),
        })
        df.to_csv(os.path.join(dir_numpy_files, 'results.csv'))

    rmse, mae = get_metrics(all_pred, all_target)
    return rmse, mae


if __name__ == "__main__":
    # 日志、数据存放路径 -----------------------------------------------------------------
    make_dirs()
    print_and_log(str(opt))
    print_and_log('log_dir: ' + dir_logs)
    print_and_log('model_name: ' + model_name)

    # 数据 -----------------------------------------------------------------------------
    dataView = DataView(opt)
    Daytime_PM_MAX, Daytime_PM_MIN = dataView.Daytime_PM_MAX, dataView.Daytime_PM_MIN
    print('Max & min PM2.5 for dataset: {:.2f}, {:.2f}'.format(Daytime_PM_MAX, Daytime_PM_MIN))

    # 网络 -----------------------------------------------------------------------------
    net = load_model(opt)
    net = net.to(opt.device)
    print_and_log(net, console=False)

    if opt.device == 'cuda':
        # net = torch.nn.DataParallel(net)
        print_and_log('Using cuda ...')
        cudnn.benchmark = True  # RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED

    # 损失函数 -------------------------------------------------------------------------------
    criterion_PM = None

    # TODO: 对比实验自行添加 Focal-R, HuberLoss...
    if 'L2' == opt.loss:
        if opt.balance and opt.balance_type != 'resample':
            criterion_PM = weighted_mse_loss  # 函数句柄
        else:
            criterion_PM = nn.MSELoss().to(opt.device)
    elif 'L1' == opt.loss:
        criterion_PM = nn.L1Loss().to(opt.device)

    print_and_log(str(criterion_PM))

    # 冻结 freeze 和 optimizer 绑定 --------------------------------------------------------
    if opt.freeze_fea:  # TODO: Only for DB-Resnet18
        print_and_log('Freeze the feature layer...')

        for name, param in net.named_parameters():
            param.requires_grad = False

        net.Fusion.requires_grad_(True)
        opt.lr = opt.lr * 0.01  # 使用原始 lr 退化很快

    optimizer = optim.Adam(
        params=filter(lambda x: x.requires_grad, net.parameters()),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    optimizer.zero_grad()

    # 训练或验证模型  ------------------------------------------------------------------------------
    if 'eval' == opt.run_type or 'resume' == opt.run_type:
        ckp = torch.load(opt.model_dir, map_location=opt.device)  # 从 resume 处传
        net = nn.DataParallel(net)  # TODO: 有些模型如早期ResNet18需要调整 key，则需要添加这一行
        net.load_state_dict(ckp['model'], strict=True)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'eval' == opt.run_type:
            rmse_val, mae_val = test(net, dataView.PM_test_loader, show_info=True, save_preds=True)
            print('rmse_val:', rmse_val, 'mae_val:', mae_val)

    if 'resume' == opt.run_type or 'train' == opt.run_type:  # train
        train(net, dataView.PM_train_loader, dataView.PM_test_loader, optimizer, criterion_PM)