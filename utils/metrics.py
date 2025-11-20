import numpy as np
from scipy import stats
import skimage.measure
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity


def get_Allmetrics(pred, gt):
    metrics = dict()
    metrics['RMSE'] = RMSE(pred, gt)
    metrics['MAE'] = MAE(pred, gt)
    metrics['NMGE'] = NMGE(pred, gt)
    metrics['LCC'] = Corr(pred, gt, type='pearson')
    metrics['SRC'] = Corr(pred, gt, type='spearman')
    metrics['KRC'] = Corr(pred, gt, type='kendall')
    metrics['R2'] = R_square(pred, gt)
    return metrics


def get_metrics(pred, gt):
    """For training & Test"""
    rmse = RMSE(pred, gt)
    mae = MAE(pred, gt)
    return rmse, mae


def RMSE(pred, gt):
    rms = mean_squared_error(gt, pred, squared=False)  # squared=False
    return round(rms, 4)


def MAE(pred, gt):
    mae = mean_absolute_error(gt, pred)  # squared=False
    return round(mae, 4)


def NMGE(pred, gt):
    gross = np.sum(np.abs(pred - gt))
    r = gross / np.sum(np.abs(gt))  # gt > 0
    return round(r, 4)


def Corr(pred, gt, type='spearman'):
    corr = None
    if type == 'spearman':
        corr = stats.spearmanr(pred, gt)
    elif type == 'pearson':
        corr = stats.pearsonr(pred, gt)
    elif type == 'kendall':
        corr = stats.kendalltau(pred, gt)
    return round(corr[0], 4)


def R_square(pred, gt):
    r2 = r2_score(pred, gt)
    return round(r2, 4)


def Distance1(p, y, dis='absolute'):
    if dis == 'absolute':
        return abs(p - y)

    return abs(2 * (p - y) / (abs(p) + abs(y)))  # D1 距离，保留方向 p - y


def Entropy(img):
    entropy = skimage.measure.shannon_entropy(img)
    return entropy


def CosineSimilarity(a1, a2):
    """
      a1 = np.arange(15).reshape(3,5)
      a2 = np.arange(20).reshape(4,5)
    """
    s1 = cosine_similarity(a1, a2)   # a1: shape (n_samples_X, n_features)
    return s1


if __name__ == "__main__":
    pass