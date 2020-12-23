import random
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score, matthews_corrcoef

import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def search_f1_auc(y_true, y_score):
    """
    as the metrics when save model
    """
    P = y_true.sum()
    R = y_score.sum()
    TP = ((y_true + y_score) > 1).sum()

    pre = TP / P
    rec = TP / R

    return 2 * (pre * rec) / (pre + rec)

def search_auc(y_true, y_score):
    """
    fp: rarray, shape = [>2]
        Increasing false positive rates such that element i is the false positive rate
        of predictions with score >= thresholds[i].
    tpr: array, shape = [>2]
        Increasing true positive rates such that element i is the true positive rate
        of predictions with score >= thresholds[i].
    thresholds: array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute fpr and tpr.
        thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return auc(fpr, tpr)

def search_f1(y_true, y_score):
    scores = []
    thresholds = [i / 100 for i in range(100)]
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        score = f1_score(y_true, y_pred)
        scores.append(score)

    threshold = thresholds[np.argmax(scores)]

    return threshold

def show_dataframe(df):
    # show all dataframe data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)

def format_time(elapsed):
    # takes a time in seconds
    # round to the nearest second
    elapsed_round = int(round(elapsed))
    # format hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_round))

def ap(y_true, y_score):
    """
    ap: average precision
    指的是在各个召回率上的正确率的平均值
    """
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_score = torch.tensor(y_score, dtype=torch.float32)

    _, idx = torch.sort(y_score, descending=True)
    y_true = y_true[idx].round()

    total = 0.
    for i in range(len(y_score)):
        j = i + 1
        if y_true[i]:
            total += y_true[:j].sum().item() / j

    true_sum = y_true.sum()
    if true_sum != 0.:
        return total / true_sum.item()

    return 0.

def map(y_true, y_score):
    """
    map: mean of ap
    """
    assert len(y_true) == len(y_score), f'Error with label length {len(y_true)} vs {len(y_score)}'

    res = []
    res.append(ap(y_true, y_score))

    return np.mean(res)

def rr(y_true, y_score):
    """
    rr: reciprocal rank
    把标准答案在被评价系统给出结果的排序取倒数作为准确度
    """
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_score = torch.tensor(y_score, dtype=torch.float32)

    _, idx = torch.sort(y_score, descending=True)
    best = y_true[idx].nonzero().squeeze().min().item()

    return 1.0 / (best + 1)

def mrr(y_true, y_score):
    """
    mrr: mean of rr
    根据rr, 再对所有的问题取平均
    """
    assert len(y_true) == len(y_score), f'Error with label length {len(y_true)} vs {len(y_score)}'

    res = []
    res.append(rr(y_true, y_score))

    return np.mean(res)

def search_mcc(y_true, y_score):
    """
    mcc: Matthews correlation coefficient
    The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications
    Binary and multiclass labels are supported. Only in the binary case does this relate to information about true and false positives and negatives
    指标：它的取值范围为[-1,1]，取值为1时表示对受试对象的完美预测，取值为0时表示预测的结果还不如随机预测的结果，-1是指预测分类和实际分类完全不一致
    """
    mcc = matthews_corrcoef(y_true, y_score)

    return mcc