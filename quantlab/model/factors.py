"""因子预处理与中性化（纯标准库线性代数）。

机构因子流水线的标准顺序：去极值 -> 缺失处理 -> 标准化 -> 行业/市值中性化 -> 再标准化。
原项目只做了「行业内排名合并」，且用 log(价格) 代理市值；这里用真实市值列。
"""
import math


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def winsorize(values: list[float], k: float = 3.0) -> list[float]:
    """MAD 去极值，None 原样保留。"""
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if len(xs) < 3:
        return list(values)
    med = _median(xs)
    mad = _median([abs(v - med) for v in xs]) or 1e-12
    lo, hi = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    return [None if v is None or not math.isfinite(v)
            else min(hi, max(lo, v)) for v in values]


def zscore(values: list[float]) -> list[float]:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if len(xs) < 2:
        return [0.0 if v is not None else None for v in values]
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((v - mu) ** 2 for v in xs) / (len(xs) - 1)) or 1e-12
    return [None if v is None or not math.isfinite(v) else (v - mu) / sd
            for v in values]


def rank(values: list[float]) -> list[float]:
    """平均秩，处理并列；返回 0..1 分位。"""
    idx = [(v, i) for i, v in enumerate(values) if v is not None and math.isfinite(v)]
    out: list[float | None] = [None] * len(values)
    n = len(idx)
    if n == 0:
        return out
    idx.sort()
    j = 0
    while j < n:
        k = j
        while k + 1 < n and idx[k + 1][0] == idx[j][0]:
            k += 1
        avg = (j + k) / 2.0
        for t in range(j, k + 1):
            out[idx[t][1]] = avg / max(1, n - 1)
        j = k + 1
    return out


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """高斯消元（部分主元）。规模是因子个数，通常 < 20。"""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[piv][c]) < 1e-12:
            m[c][c] += 1e-9
            piv = c
        m[c], m[piv] = m[piv], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


def neutralize(y: list[float], exposures: list[list[float]]) -> list[float]:
    """对若干暴露做 OLS 正交化，返回残差。None 位置保持 None。

    exposures 每列是一个暴露向量（如 log 市值、行业哑变量），已含截距列则由调用方传入。
    """
    valid = [i for i, v in enumerate(y) if v is not None and math.isfinite(v)]
    if len(valid) < len(exposures) + 3:
        return list(y)
    p = len(exposures)
    xtx = [[sum(exposures[c][i] * exposures[d][i] for i in valid)
            for d in range(p)] for c in range(p)]
    xty = [sum(exposures[c][i] * y[i] for i in valid) for c in range(p)]
    beta = _solve(xtx, xty)
    out: list[float | None] = list(y)
    for i in valid:
        fitted = sum(beta[c] * exposures[c][i] for c in range(p))
        out[i] = y[i] - fitted
    return out


def spearman_ic(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y)
             if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 5:
        return None
    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else None
