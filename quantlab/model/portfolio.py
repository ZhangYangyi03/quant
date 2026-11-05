"""组合构建：从因子分数到目标权重。

支持等权、逆波动率、带约束最小方差；所有方法都强制权重非负、和为 1、
单票上限，避免「对照组退化成持有 1231 只」这类历史 bug。
"""
import math

from ..data.panel import CrossSection


def _project_capped_simplex(w: list[float], cap: float) -> list[float]:
    """投影到 {w>=0, sum(w)=1, w<=cap}，二分求拉格朗日乘子。"""
    lo, hi = min(w) - cap, max(w)
    for _ in range(80):
        tau = (lo + hi) / 2.0
        s = sum(min(cap, max(0.0, x - tau)) for x in w)
        if s > 1.0:
            lo = tau
        else:
            hi = tau
    tau = (lo + hi) / 2.0
    return [min(cap, max(0.0, x - tau)) for x in w]


def equal_weight(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    return {c: 1.0 / len(codes) for c in codes}


def inverse_vol_weights(codes: list[str], vols: dict[str, float]) -> dict[str, float]:
    inv = {c: 1.0 / max(vols.get(c) or 0.02, 1e-4) for c in codes}
    tot = sum(inv.values())
    return {c: v / tot for c, v in inv.items()} if tot > 0 else equal_weight(codes)


def min_variance_weights(codes: list[str], cov: dict[str, dict[str, float]],
                         cap: float = 0.10, steps: int = 400,
                         lr: float = 0.05) -> dict[str, float]:
    """投影梯度最小化 w'Cw。对角加噪声保证正定，规模小无需外部求解器。"""
    n = len(codes)
    if n == 0:
        return {}
    if n == 1:
        return {codes[0]: 1.0}
    diag = [max(cov.get(a, {}).get(a, 0.0004), 1e-6) for a in codes]
    w = [1.0 / n] * n
    for _ in range(steps):
        grad = [2.0 * sum(cov.get(a, {}).get(b, 0.0 if a != b else 1e-4) * w[j]
                          for j, b in enumerate(codes)) for i, a in enumerate(codes)]
        scale = max(max(abs(g) for g in grad), 1e-9)
        w = _project_capped_simplex([w[i] - lr * grad[i] / scale for i in range(n)], cap)
    s = sum(w)
    return {c: v / s for c, v in zip(codes, w)} if s > 0 else equal_weight(codes)


def _enforce_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """统一后处理：非负、和为 1、单票不超过 cap。

    历史实现里等权/逆波动率完全无视上限，最小方差分支又只做了 min(v, cap)
    而不再归一化（权重和会悄悄小于 1，等于凭空留现金）。两者都在这里收敛。
    """
    codes = list(weights)
    if not codes:
        return {}
    vals = [max(0.0, float(weights[c])) for c in codes]
    total = sum(vals)
    if total <= 0:
        vals = [1.0 / len(codes)] * len(codes)
    else:
        vals = [v / total for v in vals]
    # 持仓数太少时 cap 与 sum=1 不可同时满足，放宽到 1/n 并记录在注释里。
    effective_cap = max(cap, 1.0 / len(codes))
    projected = _project_capped_simplex(vals, effective_cap)
    s = sum(projected)
    if s <= 0:
        return equal_weight(codes)
    return {c: v / s for c, v in zip(codes, projected)}


def build_weights(scores: dict[str, float], cs: CrossSection, top_k: int,
                  method: str = "equal", max_weight: float = 0.10,
                  min_price: float = 2.0) -> dict[str, float]:
    """选股 + 加权。价格过滤、ST 过滤、可交易性过滤都在这里统一执行。"""
    elig = []
    for code, sc in scores.items():
        if sc is None or not math.isfinite(sc):
            continue
        if code not in cs.members:
            continue
        b = cs.bar(code)
        if b is None or b.close < min_price or not b.tradable:
            continue
        if cs.universe.is_st(code, cs.at):
            continue
        elig.append((sc, code))
    elig.sort(key=lambda t: (-t[0], t[1]))
    codes = [c for _, c in elig[:max(1, top_k)]]
    if not codes:
        return {}
    if method == "equal":
        w = equal_weight(codes)
    elif method == "inverse_vol":
        vols = {c: cs.volatility(c) or 0.02 for c in codes}
        w = inverse_vol_weights(codes, vols)
    elif method == "min_variance":
        from .risk import covariance
        cov = covariance(cs, codes)
        w = min_variance_weights(codes, cov, cap=max_weight)
    else:
        raise ValueError(f"未知加权方式: {method}")
    return _enforce_cap(w, max_weight)
