# -*- coding: utf-8 -*-
# 形态/价量因子（Alpha158 的第一组）。

def _f_kbar():
    """9 个 K 线形态（Alpha158）。"""
    return [
        ("KMID",  lambda d: (d["C"] - d["O"]) / d["O"]),
        ("KLEN",  lambda d: (d["H"] - d["L"]) / d["O"]),
        ("KMID2", lambda d: (d["C"] - d["O"]) / (d["H"] - d["L"] + 1e-12)),
        ("KUP",   lambda d: (d["H"] - np.maximum(d["O"], d["C"])) / d["O"]),
        ("KUP2",  lambda d: (d["H"] - np.maximum(d["O"], d["C"])) / (d["H"] - d["L"] + 1e-12)),
        ("KLOW",  lambda d: (np.minimum(d["O"], d["C"]) - d["L"]) / d["O"]),
        ("KLOW2", lambda d: (np.minimum(d["O"], d["C"]) - d["L"]) / (d["H"] - d["L"] + 1e-12)),
        ("KSFT",  lambda d: (2 * d["C"] - d["H"] - d["L"]) / d["O"]),
        ("KSFT2", lambda d: (2 * d["C"] - d["H"] - d["L"]) / (d["H"] - d["L"] + 1e-12)),
    ]


def _f_raw():
    """原始价量：开/低/高除收盘，VWAP 归一，成交额自比。"""
    return [("O0", lambda d: d["O"] / d["C"]),
            ("H0", lambda d: d["H"] / d["C"]),
            ("L0", lambda d: d["L"] / d["C"]),
            ("VWAP0", lambda d: d["VWAP"] / d["C"]),
            ("VOLUME0", lambda d: d["AMT"] / (d["AMT"] + 1e-12))]


def build_registry():
    from ops158 import WIN
    import ops158 as _o
    g = globals()
    for _n in dir(_o):
        if not _n.startswith('__'):
            g[_n] = getattr(_o, _n)
    from roll158_a import WIN_OPS_0
    from roll158_b import WIN_OPS_1
    WIN_OPS = WIN_OPS_0 + WIN_OPS_1
    # 每个算子函数在自己的模块里查名字，所以把它们要用的算子也注进那两个模块
    import roll158_a as _a, roll158_b as _b
    for _m in (_a, _b):
        for _n in dir(_o):
            if not _n.startswith('__'):
                setattr(_m, _n, getattr(_o, _n))
    """全部因子：Alpha158 的 159 个，按算子拆成小模块，避开编译大文件时的 MemoryError。"""
    R = _f_kbar() + _f_raw()
    for w in WIN:
        for f in WIN_OPS:
            R.extend(f(w))
    return R
