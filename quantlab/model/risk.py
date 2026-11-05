"""风险模型：风格暴露、协方差收缩、组合风险分解。

纯标准库实现，规模为数十只持仓；真实全市场可把 covariance 换成
numpy/BLAS 版本，接口不变。这里保留完整的「因子风险 vs 特异性风险」分解，
因为归因结论比速度更重要。
"""
import math

from ..data.panel import CrossSection


def style_exposures(cs: CrossSection, code: str) -> dict[str, float]:
    """简化 Barra 风格：规模、波动、动量、流动性，全部只用 asof 之前的数据。"""
    b = cs.bar(code)
    if b is None:
        return {}
    h = cs.history.get(code, ())
    closes = [x.close for x in h if x.close > 0]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))] if len(closes) > 2 else []
    mu = sum(rets) / len(rets) if rets else 0.0
    vol = math.sqrt(sum((r - mu) ** 2 for r in rets) / (len(rets) - 1)) if len(rets) > 1 else 0.02
    mom = closes[-1] / closes[0] - 1 if len(closes) > 2 and closes[0] > 0 else 0.0
    return {
        "size": math.log(max(b.mktcap, 1.0)),
        "volatility": vol,
        "momentum": mom,
        "liquidity": math.log1p(max(cs.adv(code), 1.0)),
    }


def return_series(cs: CrossSection, codes: list[str], lookback: int = 20) -> dict[str, list[float]]:
    out = {}
    for c in codes:
        h = cs.history.get(c, ())
        xs = [b.close for b in h[-(lookback + 1):] if b.close > 0]
        out[c] = [xs[i] / xs[i - 1] - 1 for i in range(1, len(xs))] if len(xs) > 2 else []
    return out


def covariance(cs: CrossSection, codes: list[str], shrink: float = 0.20) -> dict[str, dict[str, float]]:
    """样本协方差 + 向对角阵收缩（Ledoit-Wolf 思想的极简版），保证正定。"""
    series = return_series(cs, codes)
    n = min((len(v) for v in series.values() if v), default=0)
    out: dict[str, dict[str, float]] = {}
    for a in codes:
        out[a] = {}
        for b in codes:
            if n < 3 or a not in series or b not in series:
                out[a][b] = 0.0004 if a == b else 0.0
                continue
            xa, xb = series[a][-n:], series[b][-n:]
            ma, mb = sum(xa) / n, sum(xb) / n
            c = sum((xa[i] - ma) * (xb[i] - mb) for i in range(n)) / (n - 1)
            out[a][b] = c * (1 - shrink) if a != b else max(c * (1 - shrink) + shrink * 0.0004, 1e-6)
    return out


def portfolio_risk(weights: dict[str, float], cov: dict[str, dict[str, float]]) -> dict[str, float]:
    codes = list(weights)
    var = sum(weights[a] * weights[b] * cov.get(a, {}).get(b, 0.0)
              for a in codes for b in codes)
    idio = sum(weights[c] ** 2 * cov.get(c, {}).get(c, 0.0) for c in codes)
    return {
        "variance": var,
        "volatility": math.sqrt(max(var, 0.0)),
        "idiosyncratic_var": idio,
        "factor_var": max(0.0, var - idio),
        "diversification_ratio": (idio / var) if var > 0 else 1.0,
    }
