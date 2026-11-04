# -*- coding: utf-8 -*-
"""wq.py -- WorldQuant Alpha101（公开论文 101 Formulaic Alphas，Kakushadze 2015）
在本地日频数据上的实现。

数据限制（先说清楚，别当成完整版）：
  本地只有 开盘 / 收盘 / 成交额 / 市值。没有 最高价、最低价、成交量、行业分类。
  - high, low 用 max(open,close), min(open,close) 近似（标 approx）
  - volume 用成交额当代理（成交额与成交量的比值就是均价，秩相关不影响大部分因子）
  - vwap 用 (high+low+close)/3 近似
  - industry neutral / indneutralize / IndClass 系列：没有行业数据，直接跳过
    （Alpha 48 58 59 63 67 69 70 76 79 80 82 87 89 90 91 93 97 100 共 18 个不属于实现）

本文件提供 numpy 版算子；因子按原论文文本逐条实现。
"""

import numpy as np
try:                      # 子进程内存配额很碎：BLAS 线程池一开就撞 MemoryError
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:
    pass

EPS = 1e-12


def ref(X, d):
    Y = np.full_like(X, np.nan)
    if d <= 0:
        return X.copy()
    if d < X.shape[0]:
        Y[d:] = X[:-d]
    return Y


def delay(X, d):
    return ref(X, d)


def delta(X, d):
    return X - ref(X, d)


def _rsum(X, d):
    Z = np.where(np.isfinite(X), X, 0.0)
    cs = np.cumsum(Z, axis=0)
    out = np.empty_like(cs)
    out[d - 1:] = cs[d - 1:]
    out[d - 1:] -= np.vstack([np.zeros((1, Z.shape[1])), cs[:-d]])
    out[:d - 1] = np.nan
    return out


def _rcnt(X, d):
    good = np.isfinite(X).astype(np.float64)
    cs = np.cumsum(good, axis=0)
    out = np.empty_like(cs)
    out[d - 1:] = cs[d - 1:]
    out[d - 1:] -= np.vstack([np.zeros((1, X.shape[1])), cs[:-d]])
    out[:d - 1] = 0.0
    return out


def ts_sum(X, d):
    return _rsum(X, d)


def sma(X, d):
    c = _rcnt(X, d)
    out = np.empty_like(c)
    np.divide(_rsum(X, d), c, out=out, where=(c > 0))
    out[c <= 0] = np.nan
    return out


def stddev(X, d):
    m = sma(X, d)
    v = sma(X * X, d) - m * m
    return np.sqrt(np.maximum(v, 0))


def correlation(A, B, d):
    ma, mb = sma(A, d), sma(B, d)
    da, db = A - ma, B - mb
    num = sma(da * db, d)
    den = np.sqrt(sma(da * da, d) * sma(db * db, d))
    out = np.empty_like(num)
    np.divide(num, den, out=out, where=(den > EPS))
    out[~(den > EPS)] = np.nan
    return out


def covariance(A, B, d):
    return sma((A - sma(A, d)) * (B - sma(B, d)), d)


def ts_max(X, d):
    """d 日最高（滑动最大值，O(T*N)）。不复用 conv 的：两个大模块同时活着会撞内存配额。"""
    Z = np.where(np.isfinite(X) & (X != 0.0), X, -np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    out[d - 1] = Z[:d].max(axis=0)
    for t in range(d, T):
        cand = np.maximum(out[t - 1], Z[t])
        need = cand == Z[t - d]
        out[t] = cand
        if need.any():
            out[t, need] = Z[t - d + 1:t + 1][:, need].max(axis=0)
    return out


def ts_min(X, d):
    Z = np.where(np.isfinite(X) & (X != 0.0), X, np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    out[d - 1] = Z[:d].min(axis=0)
    for t in range(d, T):
        cand = np.minimum(out[t - 1], Z[t])
        need = cand == Z[t - d]
        out[t] = cand
        if need.any():
            out[t, need] = Z[t - d + 1:t + 1][:, need].min(axis=0)
    return out


def ts_rank(X, d):
    """当前值在过去 d 天里的分位（0~1）。逐日切窗口较慢，但只在少数因子里用。"""
    out = np.full_like(X, np.nan)
    T = X.shape[0]
    for t in range(d - 1, T):
        w = X[t - d + 1:t + 1]
        good = np.isfinite(w)
        cnt = good.sum(axis=0)
        r = (w <= X[t]).sum(axis=0)
        out[t] = np.where(cnt > 0, r / np.maximum(cnt, 1), np.nan)
    return out


def ts_argmax(X, d):
    out = np.full_like(X, np.nan)
    T = X.shape[0]
    for t in range(d - 1, T):
        w = np.nan_to_num(X[t - d + 1:t + 1], nan=-1e30)
        out[t] = (d - 1 - np.argmax(w[::-1], axis=0)).astype(np.float64)
    return out


def ts_argmin(X, d):
    out = np.full_like(X, np.nan)
    T = X.shape[0]
    for t in range(d - 1, T):
        w = np.nan_to_num(X[t - d + 1:t + 1], nan=1e30)
        out[t] = (d - 1 - np.argmin(w[::-1], axis=0)).astype(np.float64)
    return out


def rank(X):
    """截面秩，归一化到 [0,1]。NaN 保持 NaN。"""
    out = np.full_like(X, np.nan)
    for t in range(X.shape[0]):
        row = X[t]
        good = np.isfinite(row)
        n = good.sum()
        if n < 2:
            continue
        order = np.argsort(np.argsort(row[good], kind="stable"), kind="stable")
        r = np.empty(n)
        r[order] = np.arange(n) / (n - 1.0)
        out[t, good] = r
    return out


def scale(X, a=1.0):
    s = np.nansum(np.abs(X), axis=1, keepdims=True)
    return X * a / (s + EPS)


def sign(X):
    return np.sign(X)


def indneutralize(X, g=None):
    """没有行业分类 —— 退化为截面去均值（等同"全市场中性"）。"""
    return X - np.nanmean(X, axis=1, keepdims=True)


# ============ 上下文 + Alpha101 逐条实现 ============
# 论文：Kakushadze, "101 Formulaic Alphas" (2015)，公式公开。
# 数据映射：open=O, close=C, high=max(O,C), low=min(O,C),
#           volume=AMT（成交额代成交量）, vwap=(H+L+C)/3, returns=C/ref(C,1)-1
# 需要行业分类的 18 条不实现。

def ctx(d):
    C, O, A = d["C"], d["O"], d["AMT"]
    H = np.maximum(O, C)
    L = np.minimum(O, C)
    return dict(C=C, O=O, H=H, L=L, V=A, R=C / ref(C, 1) - 1.0,
                VW=(H + L + C) / 3.0, CAP=d.get("CAP"))


def alpha101_registry(extra_factors):
    """extra_factors: 由 build_alpha101() 生成，避免在本文件里写巨型 lambda 表时出语法错。"""
    return extra_factors
