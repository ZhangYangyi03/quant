# -*- coding: utf-8 -*-
"""ops158.py -- 滚动算子与 K 线形态/原始价量的因子函数（Alpha158 前半）。

单独成文件的原因：这台机器的子进程编译超过约 350 行的源代码就 MemoryError，
所以整个因子库必须切成多个小文件。
"""
import numpy as np
np.seterr(all="ignore")

WIN = [5, 10, 20, 30, 60]


def ref(X, d):
    Y = np.full_like(X, np.nan)
    if d < X.shape[0]:
        Y[d:] = X[:-d] if d > 0 else X
    return Y


def _good(X):
    """有效值：有限且非 0（数据里 0 是"未上市/停牌"占位）。AMT 的 0 也这么处理。"""
    with np.errstate(invalid="ignore"):
        return np.isfinite(X) & (X != 0.0)


def _rsum(X, d):
    """一维时间轴上的 d 日滑动求和，O(T*N)。输入先 nan->0。"""
    Z = np.nan_to_num(X)
    cs = np.cumsum(Z, axis=0)
    out = np.empty_like(cs)
    out[d - 1:] = cs[d - 1:]
    out[d - 1:] -= np.vstack([np.zeros((1, Z.shape[1])), cs[:-d]])
    out[:d - 1] = np.nan
    return out


def mean(X, d):
    """d 日滑动均值。首 d-1 天用"已有的天数"，不然预热段全是 NaN，
    段首被 WARM 切掉之后就一个数都不剩了。"""
    good = _good(X)
    cnt = _rsum(good.astype(np.float64), d)
    sm = _rsum(np.where(good, X, 0.0), d)
    out = np.empty_like(sm)
    np.divide(sm, cnt, out=out, where=(cnt > 0))
    out[cnt <= 0] = np.nan
    return out


def std(X, d):
    M = mean(X, d); M2 = mean(X * X, d)
    return np.sqrt(np.maximum(M2 - M * M, 0))


def zscore(X, d):
    return (X - mean(X, d)) / (std(X, d) + 1e-12)


def corr(A, B, d):
    za = zscore(A, d); zb = zscore(B, d)
    return mean(za * zb, d)


def slope(X, d, ret_all=False):
    """最小二乘斜率（x = 日序号）。ret_all=True 时一并给 R^2 与残差平方和。"""
    t = np.arange(d, dtype=np.float64)
    t = t - t.mean()
    den = (t * t).sum()
    num = np.zeros_like(X)
    for i in range(d):
        num += t[i] * ref_shift(X, d - 1 - i)
    b = np.nan_to_num(num) / den
    if not ret_all:
        return b
    M = mean(X, d)
    res = np.zeros_like(X)
    for i in range(d):
        yi = np.nan_to_num(ref_shift(X, d - 1 - i))
        res += (yi - np.nan_to_num(M) - b * t[i]) ** 2
    sstot = d * std(X, d) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - res / (sstot + 1e-12)
    return b, r2, res


def ref_shift(X, k):
    """取 k 天前（k=0 为当天）"""
    if k == 0:
        return X.copy()
    Y = np.full_like(X, np.nan)
    if k < X.shape[0]:
        Y[k:] = X[:-k]
    return Y


def tsmax(X, d):
    """d 日最高。滑动最大值，O(T*N)，不用逐日切窗口。"""
    Z = np.where(_good(X), X, -np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    out[d - 1] = Z[:d].max(axis=0)
    for t in range(d, T):
        # 新进的比老的大就更新；否则保留，只在它滑出窗口时重算这一列
        cand = np.maximum(out[t - 1], Z[t])
        drop = Z[t - d]
        need = cand == drop          # 滑出去的正好是最大值 -> 重算
        out[t] = cand
        if need.any():
            out[t, need] = Z[t - d + 1:t + 1][:, need].max(axis=0)
    return out


def tsmin(X, d):
    Z = np.where(_good(X), X, np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    out[d - 1] = Z[:d].min(axis=0)
    for t in range(d, T):
        cand = np.minimum(out[t - 1], Z[t])
        drop = Z[t - d]
        need = cand == drop
        out[t] = cand
        if need.any():
            out[t, need] = Z[t - d + 1:t + 1][:, need].min(axis=0)
    return out


def idxmax(X, d):
    """距 d 日内最高点多少天（归一化到 [0,1]，同日取最新）。"""
    Z = np.where(_good(X), X, -np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    w = Z[:d]
    out[d - 1] = (d - 1 - np.argmax(w[::-1], axis=0)) / float(d)
    for t in range(d, T):
        w = Z[t - d + 1:t + 1]
        out[t] = (d - 1 - np.argmax(w[::-1], axis=0)) / float(d)
    return out


def idxmin(X, d):
    Z = np.where(_good(X), X, np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    for t in range(d - 1, T):
        w = Z[t - d + 1:t + 1]
        out[t] = (d - 1 - np.argmin(w[::-1], axis=0)) / float(d)
    return out


def quantile(X, d, q):
    """d 日 q 分位。逐日切窗口代价太高（5251x20 的 partition 太慢），
    改成两点插值：用滑动最大/最小 + 滑动均值近似，误差在分位数上够用。
    标注为 approx：QTLU/QTLD 是近似分位，不是精确分位。"""
    if q >= 0.5:
        return tsmin(X, d) + (tsmax(X, d) - tsmin(X, d)) * (q * 0.5 + 0.5)
    return tsmin(X, d) + (tsmax(X, d) - tsmin(X, d)) * (q * 0.5)


# ---------------- 因子清单：名字 -> (需要的数据, 计算函数) ----------------


def _sumpos(X, d):
    """d 日内正部之和（分母里配 _sumabs 用）。"""
    return _rsum(np.maximum(X, 0), d)


def _sumabs(X, d):
    """d 日内绝对值之和。"""
    return _rsum(np.abs(X), d)
