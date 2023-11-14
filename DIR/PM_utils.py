import numpy as np
from collections import defaultdict
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

import figs.plots as plots


def help_plot(PM25s, bin_width=10):
    """绘制过程图"""
    PM25_min, PM25_max = min(PM25s), max(PM25s)
    bins = range(int(PM25_min), int(PM25_max) + 1 + bin_width, bin_width)
    plots.plot_dist(PM25s, bins=bins, x_label='PM2.5 浓度', y_label='样本数目', density=False)


def help_plot_curve(smoothed_value, bin_width=10):
    x_values = list(range(1, 262, bin_width))  # max, bin_widht
    # sums = np.sum(smoothed_value)
    # smoothed_value = smoothed_value / sums
    plots.plot_one_statistics(
        smoothed_value, x_values=x_values,
        xlabel='PM2.5 浓度', y_label="拟合的样本数目", save_dir=None
    )


# for i in sorted(self.count_dict):
#     print(i, self.count_dict[i])

class PM_ManyShot2():
    """可能要去除极端值再计算"""
    def __init__(self, train_PMs, preds=None, ytrue=None, few_shot_V=20, medium_shot_V=100):
        """
          @train_PMs：训练集样本，根据次划分类别
          @preds: 测试集预测值
          @ytrue: 测试集真实值
        """
        super(PM_ManyShot2, self).__init__()

        # bins = 1 直方图粒度
        train_PMs = [int(x) for x in train_PMs]  # list(map(int, results))
        count_dict = defaultdict(int)  # sorted(key_value)
        for y in train_PMs:
            count_dict[y] += 1

        self.ytrue = ytrue
        self.pred = preds
        self.count_dict = count_dict
        self.few_shot_V = few_shot_V
        self.medium_shot_V = medium_shot_V

        self._classify()

    def _classify(self):
        py_pairs_few, py_pairs_medium, py_pairs_many = [], [], []
        # 下标顺序要保持不变；分类边界：20,100
        for (p, y) in zip(self.pred, self.ytrue):
            if self.count_dict[y] < self.few_shot_V:
                py_pairs_few.append((p, y))
            elif self.count_dict[y] < self.medium_shot_V:
                py_pairs_medium.append((p, y))
            else:
                py_pairs_many.append((p, y))

        # 动态添加
        self.py_pairs_few = py_pairs_few
        self.py_pairs_medium = py_pairs_medium
        self.py_pairs_many = py_pairs_many

        print("len few: {}, medium: {}, many: {}".format(len(py_pairs_few), len(py_pairs_medium), len(py_pairs_many)))
        assert len(self.ytrue) == len(py_pairs_few) + len(py_pairs_medium) + len(py_pairs_many), "Error!"

    def get_ManyShot(self, type="Few"):
        if type == "Few":
            py_pairs = self.py_pairs_few
        elif type == "Medium":
            py_pairs = self.py_pairs_medium
        elif type == "Many":
            py_pairs = self.py_pairs_many
        else:
            return None
        pp = [x[0] for x in py_pairs]
        tt = [x[1] for x in py_pairs]
        return pp, tt


class PM_ManyShot():
    """可能要去除极端值再计算"""
    def __init__(self, preds=None, ytrue=None, few_shot_V=115):
        """
          @preds: 测试集预测值
          @ytrue: 测试集真实值
        """
        super(PM_ManyShot, self).__init__()

        self.ytrue = ytrue
        self.pred = preds
        self.few_shot_V = few_shot_V

        self._classify()

    def _classify(self):
        py_pairs_few = []
        for (p, t) in zip(self.pred, self.ytrue):
            if t > self.few_shot_V:
                py_pairs_few.append((p, t))

        # 动态添加
        self.py_pairs_few = py_pairs_few
        print("len few: {}".format(len(py_pairs_few)))

    def get_FewShot(self):
        py_pairs = self.py_pairs_few
        pp = [x[0] for x in py_pairs]
        tt = [x[1] for x in py_pairs]
        return pp, tt


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def resample_weights(labels):
    """重采样 + 欠采样
      @https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    """
    labels = labels.astype(int)
    min_target, max_target = min(labels), max(labels)
    value_dict = {x: 0 for x in range(min_target, max_target + 1)}
    for label in labels:
        value_dict[label] += 1

    value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[label] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]  # TODO: 没有归一化
    return weights


def cb_weights(labels):
    """CB loss
      @https://github.com/richardaecn/class-balanced-loss/blob/master/data.ipynb
    """
    labels = labels.astype(int)
    min_target, max_target = min(labels), max(labels)

    hist, bin_edges = np.histogram(labels, bins=1+max_target-min_target)  # 左闭区间，右开区间
    value_dict = [np.clip(v, 5, 1000) for v in hist]

    b = 0.999
    for i, x in enumerate(value_dict):
        value_dict[i] = (1.0 - np.power(b, x)) / (1 - b)

    w_dict = [1.0 / x for x in value_dict]
    scale = len(w_dict) / np.sum(w_dict)
    w_dict_nor = [scale * x for x in w_dict]  # TODO: 加和为 C

    w_per_label = [w_dict_nor[label - min_target] for label in labels]
    # from DIR.test_lds import parse_weights
    # x_data, w = parse_weights(w_per_label)
    # plots.plot_one_statistics(w, x_values=x_data)

    return w_per_label


def lds_weights(labels, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    """原始 LDS 标签分布平滑，先放在 Dataset 外面，用于未归一化前
      @https://github.com/YyzHarry/imbalanced-regression/blob/main/agedb-dir/datasets.py
      @labels: 原始标签值，如 [1~262]
      @return: 每个标签对应的权重，按顺序返回
    """
    labels = labels.astype(int)
    min_target, max_target = min(labels), max(labels)
    value_dict = {x: 0 for x in range(min_target, max_target + 1)}  # 预先定义好插入顺序
    for label in labels:
        value_dict[label] += 1

    value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label2 = [value_dict[label] for label in labels]  # 对应回每个样本

    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')

    # Python3.6 版本以后的 dict 是有序的，但作用只是记住元素插入顺序并按顺序输出
    smoothed_value = convolve1d(
        np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')

    # --- 中间结果 ---
    # smoothed_labels = []
    # for k, v in value_dict.items():
    #     for i in range(v):
    #         smoothed_labels.append(k)
    # --- 中间结果 ---

    num_per_label = [smoothed_value[label - min_target] for label in labels]  # smoothed_value 不是键值对，注意下标

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)  # scaling: 127.13
    weights = [scaling * x for x in weights]  # [0.27 ~ 9.08], sqrt_inv [0.49, 22.9]
    return weights


def lds_weights_bin(labels, bin_width=5, lds_kernel='gaussian', lds_ks=5, lds_sigma=2, clip=500):
    """标签分布平滑，先放在 Dataset 外面，用于未归一化前
      @labels: 原始标签值，如 [1~262]
      @return: 每个标签对应的权重，按顺序返回
    """
    labels = labels.astype(int)
    # help_plot(labels, bin_width=bin_width)

    min_target, max_target = min(labels), max(labels)
    bins = range(int(min_target), int(max_target) + 1 + bin_width, bin_width)  # [1, 6, 11, ..., 256, 261, 266]
    hist, bin_edges = np.histogram(labels, bins)  # 左闭区间，右开区间
    if clip > 0 and clip <= 100:
        # 求分位数
        clip = np.percentile(hist, clip)
        print('------- LDS_bin clip: ', clip)

    value_dict = [np.clip(v, 5, clip) for v in hist]  # TODO: 区别，上限 500

    lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
    lds_kernel_window = lds_kernel_window / sum(lds_kernel_window)  # TODO: 区别，归一化核系数
    # print(f'Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})')

    smoothed_value = convolve1d(
        np.asarray([v for v in value_dict]), weights=lds_kernel_window, mode='reflect')
    # help_plot_curve(smoothed_value, bin_width)

    # 根据样本所在 bin 对应回每个样本
    num_per_label = []
    for label in labels:
        idx = (label - 1) // bin_width
        num_per_label.append(smoothed_value[idx])

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)  # bw=5 scaling: 385 TODO: 还是与原始数目相关？
    weights = [1.0 + scaling * x for x in weights]  # bw=5 [0.22 ~ 28.24] TODO: 区别，加偏置

    return weights