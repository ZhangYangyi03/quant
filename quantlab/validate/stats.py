"""绩效与统计检验：纯标准库实现，不依赖 scipy / statsmodels。

包含机构研究必备的四道关：
  Newey-West HAC t 值 · Deflated Sharpe（多重检验）· PBO（CSCV 过拟合概率）
  · bootstrap 版 White's Reality Check
"""
import itertools
import math
import random

from ..config import ValidationConfig


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """二分求标准正态分位数，避免引入 scipy。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p 必须在 (0,1)")
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def annualized_sharpe(returns: list[float], cfg: ValidationConfig | None = None) -> float:
    ann = cfg.trading_days if cfg else 250
    sd = _std(returns)
    return _mean(returns) / sd * math.sqrt(ann) if sd > 0 else 0.0


def performance(equity_curve: list[tuple[str, float]],
                cfg: ValidationConfig | None = None) -> dict:
    v = cfg or ValidationConfig()
    if len(equity_curve) < 2:
        return {"error": "too_short"}
    dates = [d for d, _ in equity_curve]
    eq = [float(x) for _, x in equity_curve]
    rets = [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]
    n = len(rets)
    mult = eq[-1] / eq[0] if eq[0] > 0 else 0.0
    cagr = (mult ** (v.trading_days / max(n, 1)) - 1) if mult > 0 else -1.0
    vol = _std(rets) * math.sqrt(v.trading_days)
    mu = _mean(rets)
    sd = _std(rets)
    downside = [min(0.0, r - mu) for r in rets]
    dsd = math.sqrt(sum(x * x for x in downside) / max(n - 1, 1))
    peak, mdd = eq[0], 0.0
    for x in eq:
        peak = max(peak, x)
        mdd = min(mdd, x / peak - 1)
    sk = ku = 0.0
    if sd > 0 and n > 3:
        sk = sum(((r - mu) / sd) ** 3 for r in rets) / n
        ku = sum(((r - mu) / sd) ** 4 for r in rets) / n
    return {
        "start": dates[0], "end": dates[-1], "days": len(eq),
        "total_multiple": round(mult, 4), "cagr_pct": round(cagr * 100, 4),
        "ann_vol_pct": round(vol * 100, 4),
        "sharpe": round(mu / sd * math.sqrt(v.trading_days), 4) if sd > 0 else 0.0,
        "sortino": round(mu / dsd * math.sqrt(v.trading_days), 4) if dsd > 0 else 0.0,
        "max_drawdown_pct": round(mdd * 100, 4),
        "calmar": round(cagr / abs(mdd), 4) if mdd < 0 else None,
        "skew": round(sk, 4), "excess_kurtosis": round(ku - 3.0, 4),
    }


def newey_west_t(returns: list[float], lags: int | None = None) -> float:
    """HAC 修正的均值 t 统计量；lags 默认 floor(4*(T/100)^(2/9))。"""
    n = len(returns)
    if n < 5:
        return 0.0
    if lags is None:
        lags = int(4 * (n / 100.0) ** (2.0 / 9.0))
    mu = _mean(returns)
    dev = [r - mu for r in returns]
    gamma0 = sum(x * x for x in dev) / n
    var = gamma0
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        gl = sum(dev[i] * dev[i - L] for i in range(L, n)) / n
        var += 2.0 * w * gl
    se = math.sqrt(max(var, 1e-18) / n)
    return mu / se if se > 0 else 0.0


def deflated_sharpe(sr_daily: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurtosis: float = 3.0,
                    variance_of_sr: float = 0.0) -> float:
    """Bailey & Lopez de Prado (2014) 的 DSR，返回概率值（>0.95 才算显著）。"""
    n_obs = max(n_obs, 2)
    n_trials = max(n_trials, 1)
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    if n_trials <= 1:
        # 只试了一次时不存在「多重检验膨胀」，期望最大 Sharpe 为 0。
        # 直接套 1 - 1/N 会得到 p=0 并让 _norm_ppf 抛 ValueError。
        sr0 = 0.0
    else:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(max(variance_of_sr, 1e-12)) * ((1 - gamma) * z1 + gamma * z2)
    denom = math.sqrt(max(1.0 - skew * sr_daily + (kurtosis - 1) / 4 * sr_daily ** 2, 1e-12))
    z = (sr_daily - sr0) * math.sqrt(n_obs - 1) / denom
    return _norm_cdf(z)


def pbo(returns_by_config: dict[str, list[float]], slices: int = 8) -> float:
    """CSCV 组合对称交叉验证：IS 最优配置在 OOS 排名低于中位数的比例。"""
    keys = [k for k, v in returns_by_config.items() if len(v) >= slices]
    if len(keys) < 2:
        return 0.5
    n = min(len(returns_by_config[k]) for k in keys)
    n -= n % slices
    if n < slices * 2:
        return 0.5
    seg = n // slices
    mat = {k: returns_by_config[k][:n] for k in keys}
    ranks_below, total = 0, 0
    for is_idx in itertools.combinations(range(slices), slices // 2):
        oos_idx = [i for i in range(slices) if i not in is_idx]
        def sr(k, idxs):
            vals = [x for i in idxs for x in mat[k][i * seg:(i + 1) * seg]]
            sd = _std(vals)
            return _mean(vals) / sd if sd > 0 else 0.0
        best = max(keys, key=lambda k: sr(k, is_idx))
        oos_sr = {k: sr(k, oos_idx) for k in keys}
        rank = sorted(keys, key=lambda k: -oos_sr[k]).index(best) + 1
        if rank > len(keys) / 2:
            ranks_below += 1
        total += 1
    return ranks_below / total if total else 0.5


def bootstrap_max_sharpe_p(returns_by_config: dict[str, list[float]],
                           draws: int = 500, seed: int = 7) -> float:
    """White's Reality Check 的 bootstrap 实现。

    零假设：所有配置的真实超额收益为 0（各自去均值后重抽样）。
    统计每次重抽样的最大 Sharpe，给出观测最大 Sharpe 的 p 值。
    """
    keys = list(returns_by_config)
    if not keys:
        return 1.0
    n = min(len(returns_by_config[k]) for k in keys)
    if n < 10:
        return 1.0
    centred = {k: [x - _mean(returns_by_config[k][:n]) for x in returns_by_config[k][:n]]
               for k in keys}
    obs = max(abs(annualized_sharpe(returns_by_config[k][:n])) for k in keys)
    rng = random.Random(seed)
    exceed = 0
    for _ in range(draws):
        best = 0.0
        for k in keys:
            pool = centred[k]
            sample = [pool[rng.randrange(n)] for _ in range(n)]
            best = max(best, abs(annualized_sharpe(sample)))
        if best >= obs:
            exceed += 1
    return (exceed + 1) / (draws + 1)
