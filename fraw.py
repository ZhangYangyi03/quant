# -*- coding: utf-8 -*-
"""fraw.py -- 直接从 fscan_parts/ 取一行或一段因子，不再依赖 fpanel.npz。

为什么：fpanel.npz 42MB，在这台机器上 np.load 解压就炸；而 fscan_parts 里
每个因子每段一个 .npy（<= 8 个月 x 5251），读进去只要几 MB。
"""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "fscan_parts")


def meta():
    """返回 {"dates": [交易日序号...], "seg": {月序号: 段号}, "off": {月序号: 段内行号}, "n": 月数}"""
    ids = sorted(int(x[5:-4]) for x in os.listdir(PARTS) if x.startswith("meta_"))
    pairs = []
    for sid in ids:
        jj = np.load(os.path.join(PARTS, "meta_%05d.npy" % sid))
        pairs.extend([(int(jj[i]), sid, i) for i in range(len(jj))])
    pairs.sort(key=lambda x: x[0])
    JJ = np.array([p[0] for p in pairs], dtype=np.int64)
    seg = {m: pairs[m][1] for m in range(len(pairs))}
    off = {m: pairs[m][2] for m in range(len(pairs))}
    return {"dates": JJ, "seg": seg, "off": off, "n": len(pairs), "ids": ids}


def meta_unique():
    """按交易日去重的时间轴。

    本机 fscan_parts 有重复落盘的段（26 与 29 是同一批 8 个月，24 与 34 有 5 个月
    重叠），直接 meta() 会得到 183 行里 166 个唯一交易日；相邻重复月会让"当月收盘
    -> 下月收盘"的远期收益恒为 0，IC 与组合超额都被稀释。所有下游一律用这个。
    """
    _M = meta()
    seen, keep = set(), []
    for m in range(_M["n"]):
        j = int(_M["dates"][m])
        if j in seen:
            continue
        seen.add(j)
        keep.append(m)
    out = {"dates": [_M["dates"][m] for m in keep],
           "seg": {}, "off": {}, "n": len(keep), "ids": _M["ids"]}
    for new, old in enumerate(keep):
        out["seg"][new] = _M["seg"][old]
        out["off"][new] = _M["off"][old]
    return out


def row(name, m, _M=None):
    # _M 必须显式传：默认 meta() 是含重复月的原始轴，与 meta_unique() 的下标不通用。
    # 9d437f3 那次事故正是把两条轴混用，IC 被算成两倍。
    if _M is None:
        raise ValueError("fraw.row 需要显式传 _M（meta() 或 meta_unique()）")
    if not (0 <= m < _M["n"]):
        raise IndexError("fraw.row 月序号越界: %d" % m)
    a = np.load(os.path.join(PARTS, "%s_%05d.npy" % (name, _M["seg"][m])))
    return a[_M["off"][m]]


def rows(name, i0, i1, _M=None):
    if _M is None:
        raise ValueError("fraw.rows 需要显式传 _M（meta() 或 meta_unique()）")
    if not (0 <= i0 <= i1 <= _M["n"]):
        raise IndexError("fraw.rows 月区间越界")
    out = None
    for m in range(i0, i1):
        r = row(name, m, _M)[None, :]
        if out is None:
            out = np.zeros((i1 - i0, r.shape[1]), dtype=r.dtype)
        out[m - i0] = r[0]
    return out


def matrix(name, _M=None):
    """整段读进来（8 x 5251 x float32 = 168KB），拼成 (月数, 股票数)。"""
    if _M is None:
        raise ValueError("fraw.matrix 需要显式传 _M")
    return rows(name, 0, _M["n"], _M)
