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


def row(name, m, _M=None):
    _M = meta() if _M is None else _M
    a = np.load(os.path.join(PARTS, "%s_%05d.npy" % (name, _M["seg"][m])))
    return a[_M["off"][m]]


def rows(name, i0, i1, _M=None):
    _M = meta() if _M is None else _M
    out = None
    for m in range(i0, i1):
        r = row(name, m, _M)[None, :]
        if out is None:
            out = np.zeros((i1 - i0, r.shape[1]), dtype=r.dtype)
        out[m - i0] = r[0]
    return out


def matrix(name):
    """整段读进来（8 x 5251 x float32 = 168KB），拼成 (月数, 股票数)。"""
    _M = meta()
    return rows(name, 0, _M["n"], _M)
