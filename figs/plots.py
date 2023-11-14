import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import figs.figs_options  # 保持字体一致


def plot_dist(data, bins=None, x_label='PM 2.5', y_label='Frequency', density=True, savename=''):
    """绘制直方图"""
    if bins is None:
        bins = 20  # BLH 20 等分，Heshan 50 等分

    plt.hist(data, bins=bins, density=density, color='tab:orange')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))

    plt.xlabel(x_label, labelpad=8)
    plt.ylabel(y_label, labelpad=8)

    plt.tight_layout()
    if savename != '':
        plt.savefig(savename + '-hist.png', bbox_inches='tight')  # 多图会大小不一
    plt.show()


def plot_one_statistics(
        y_values, x_values=None, xlabel="steps", y_label="loss",
        save_dir=None, x_space=50
    ):
    x_major_locator = MultipleLocator(x_space)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.xlabel(xlabel)
    plt.ylabel(y_label)
    if x_values is None:
        plt.plot(y_values, color='tab:orange')
    else:
        plt.plot(x_values, y_values, color='tab:orange')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, y_label + '.png'))

    plt.tight_layout()
    plt.show()


def plot_multi_statistics(arrs, xdata=None, titles=('',), xlabel='sample', ylabel='PM2.5', save_dir=None):
    """多条折线图"""
    if xdata is None:
        xdata = np.arange(1, len(arrs[0]) + 1, 1)
    linestys = ['--', '-.', 'solid', 'dotted', ':']
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
    fig, ax = plt.subplots()      # 创建图实例
    for i in range(len(arrs)):
        tlt = str(i + 1) if len(titles) < i + 1 else titles[i]
        ax.plot(xdata, arrs[i], label=tlt, linestyle=linestys[i], color=colors[i])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, titles[0] + '-对比.png'), dpi=1000)
    plt.show()


def show_scatter(
        x, y, xlable="Estimated PM2.5 indices", ylabel="Meausured PM2.5 indices", save_dir=None,
        marker='^', color='blue', LCC=42, KRC=40,
    ):
    maxL = max(y)  # max(max(x), max(y))
    delta_x = 70
    plt.xlim(xmin=0, xmax=maxL + delta_x)  # PPPC 152 -> 220 偏差较大
    plt.ylim(ymin=0, ymax=maxL + delta_x)  # +20 -> + 70

    plt.scatter(x, y, marker=marker, c=color)

    # plt.text(5, 106, 'LCC = {}%'.format(LCC))  # for Heshan
    # plt.text(5,  97, 'KRC = {}%'.format(KRC))
    # plt.text(193, 37,   'LCC = {}%'.format(LCC))  # for Beijing
    # plt.text(193, 17.5, 'KRC = {}%'.format(KRC))
    plt.text(10, 196, 'LCC = {}%'.format(LCC))  # for VSS
    plt.text(10,  180, 'KRC = {}%'.format(KRC))

    a = np.arange(0, maxL + delta_x + 1, 1)
    plt.plot(a, a, color='k', linestyle='--', linewidth=0.7)

    plt.xlabel(xlable, labelpad=8)
    plt.ylabel(ylabel, labelpad=8)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'scatter.png'), bbox_inches='tight')

    plt.show()

    # 显示相对误差 ------------------------------------------
    x = np.abs(x - y) / y  # 相对误差
    max_x, max_y = max(x), max(y)
    plt.xlim(xmin=0, xmax=1.0)
    plt.ylim(ymin=0, ymax=max_y + 10)

    plt.scatter(x, y, marker=marker, c=color)
    a = np.arange(0, max_y + 10 + 1, 1)
    plt.plot(np.zeros(len(a)) + 0.2, a, color='r', linestyle='--', linewidth=0.7)

    plt.xlabel("Relative error", labelpad=8)
    plt.ylabel("Measured PM2.5 indices", labelpad=8)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'scatter—_r.png'), bbox_inches='tight')

    plt.show()



def show_double_hist(all_target, all_pred):
    """查看测试集预测值、真实值直方图"""
    x_bins = np.linspace(0, 270, 27)  # MAE 10 以内
    plt.hist(all_target, bins=x_bins, alpha=0.5, label='True')   # bins=20
    plt.hist(all_pred, bins=x_bins, alpha=0.5, label='Predict')  # h_pred 有返回值
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    pass
