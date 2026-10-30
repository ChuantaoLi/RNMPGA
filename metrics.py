"""
评估指标实现
"""

import numpy as np


# ===========================================================================
# 一、MCE: 最小分类误差
# ===========================================================================
def mce_metric(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    tprs, tnrs, fprs, fnrs = [], [], [], []
    for c in np.unique(y_true):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        tn = int(np.sum((y_true != c) & (y_pred != c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        denom_pos = tp + fn
        denom_neg = tn + fp
        if denom_pos == 0:
            tprs.append(1.0)
            fnrs.append(0.0)
        else:
            tprs.append(tp / denom_pos)
            fnrs.append(fn / denom_pos)
        if denom_neg == 0:
            tnrs.append(1.0)
            fprs.append(0.0)
        else:
            tnrs.append(tn / denom_neg)
            fprs.append(fp / denom_neg)

    tpr, tnr, fpr, fnr = map(np.mean, (tprs, tnrs, fprs, fnrs))
    return float((fpr + fnr) / (tpr + tnr + fpr + fnr))


# ===========================================================================
# 二、FN: 特征数占比
# ===========================================================================
def feature_ratio(n_selected, n_total):
    if n_total <= 0:
        return 0.0
    return float(n_selected) / float(n_total)


# ===========================================================================
# 三、HV: 超体积
# ===========================================================================
def hypervolume_2d(obj, ref):
    obj = np.asarray(obj, dtype=float)
    ref = np.asarray(ref, dtype=float)
    pts = sorted({tuple(p) for p in obj if np.all(p <= ref)})
    if not pts:
        return 0.0
    env = []
    for p in pts:
        if not env or p[1] < env[-1][1]:
            env.append(p)
    x_prev, hv = ref[0], 0.0
    for x, y in reversed(env):
        hv += max(x_prev - x, 0.0) * (ref[1] - y)
        x_prev = x
    return float(hv)


# ===========================================================================
# 四、IGD: 反向世代距离
# ===========================================================================
def igd(approx, true_pf):
    approx = np.asarray(approx, dtype=float)
    true_pf = np.asarray(true_pf, dtype=float)
    if approx.size == 0 or true_pf.size == 0:
        return float("nan")
    # 每对 (f, g) 的欧氏距离, 再对每个 f 取到真实集合的最近距离并平均
    dists = np.sqrt(((approx[:, None, :] - true_pf[None, :, :]) ** 2).sum(axis=2))
    return float(dists.min(axis=1).mean())


# ===========================================================================
# 五、辅助: 非支配筛选
# ===========================================================================
def non_dominated_indices(obj):
    obj = np.asarray(obj, dtype=float)
    n = obj.shape[0]
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                is_nd[i] = False
                break
    return np.where(is_nd)[0]
