"""绩效与统计：numpy + 标准库，不依赖 scipy/statsmodels。

包含：CAGR、年化波动、夏普（含无风险利率）、Sortino、Calmar、最大回撤与修复期、
      偏度峰度、Newey-West t 统计量、Deflated Sharpe、自相关、月度胜率、相对基准的 alpha/beta。
"""
import math
import numpy as np

# 为什么不用 scipy/statsmodels：在本机 mmap 住 3000+ 只股票后，
# 再 import scipy 会因进程地址空间不足直接 OSError(WinError 8)。
# 这里只用到三个统计量，全部用 numpy/标准库重写，数值定义一致。
class _sps:
    @staticmethod
    def norm_sf(z):
        return 0.5 * math.erfc(z / math.sqrt(2))

    @staticmethod
    def skew(x):
        x = np.asarray(x, dtype=float); n = len(x)
        m = x.mean(); s = x.std(ddof=0)
        return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else 0.0

    @staticmethod
    def kurtosis(x, fisher=True):
        x = np.asarray(x, dtype=float)
        m = x.mean(); s = x.std(ddof=0)
        if s <= 0: return 0.0
        k = float(((x - m) ** 4).mean() / s ** 4) - 3.0
        return k if fisher else k + 3.0
sps = _sps

def _norm_ppf(p):
    """标准正态分位函数（Acklam 有理逼近，最大相对误差 1.15e-9）。
    只替代 scipy.stats.norm.ppf，定义与 scipy 相同。"""
    if p <= 0.0: return float("-inf")
    if p >= 1.0: return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        qv = math.sqrt(-2 * math.log(p))
        return (((((c[0]*qv+c[1])*qv+c[2])*qv+c[3])*qv+c[4])*qv+c[5]) / ((((d[0]*qv+d[1])*qv+d[2])*qv+d[3])*qv+1)
    if p > phigh:
        qv = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*qv+c[1])*qv+c[2])*qv+c[3])*qv+c[4])*qv+c[5]) / ((((d[0]*qv+d[1])*qv+d[2])*qv+d[3])*qv+1)
    qv = p - 0.5; r = qv*qv
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*qv / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _ols_hac(y, X, maxlags):
    """普通最小二乘 + Newey-West(HAC) 标准误，返回 t 值数组。
    只替代 statsmodels.OLS(y, X).fit(cov_type="HAC")，公式即 Newey-West 1987。"""
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    n, k = X.shape
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None]) / n
    for L in range(1, maxlags + 1):
        w = 1.0 - L / (maxlags + 1.0)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None]) / n
        S = S + w * (G + G.T)
    cov = XtXi @ S @ XtXi / n
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    return beta, beta / se

def _ljung_box_p(r, m):
    """Ljung-Box 白噪声检验的 p 值（Q ~ chi2(m)）。chi2 尾概率用正则化不完全伽马递推，标准库实现。"""
    r = np.asarray(r, dtype=float); n = len(r)
    if n < m + 2: return None
    r = r - r.mean()
    denom = float((r ** 2).sum())
    if denom <= 0: return None
    Q = 0.0
    for k in range(1, m + 1):
        rho = float((r[k:] * r[:-k]).sum()) / denom
        Q += rho * rho / (n - k)
    Q *= n * (n + 2)
    return round(_chi2_sf(Q, m), 4)


def _chi2_sf(x, k):
    """卡方分布尾概率 P(X > x)，即正则化上不完全伽马 Q(k/2, x/2)。"""
    if x <= 0: return 1.0
    a, z = k / 2.0, x / 2.0
    # 级数（z < a+1 收敛快）
    if z < a + 1:
        term = 1.0 / a; s = term; n = 0
        while True:
            n += 1; term *= z / (a + n); s += term
            if abs(term) < abs(s) * 1e-15 or n > 10000: break
        p = s * math.exp(-z + a * math.log(z) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    # 连分式（Lentz），算 Q(a,z) 直接
    tiny = 1e-300
    b = z + 1 - a; c = 1 / tiny; d = 1 / b; h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b; d = tiny if abs(d) < tiny else d
        c = b + an / c; c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c; h *= de
        if abs(de - 1) < 1e-15: break
    Qv = math.exp(-z + a * math.log(z) - math.lgamma(a)) * h
    return max(0.0, min(1.0, Qv))


TRADING_DAYS = 250

def to_returns(equity):
    """equity: [(date, value)] -> (dates, np.array of daily returns)"""
    vals = np.array([v for _, v in equity], dtype=float)
    rets = np.diff(vals)/vals[:-1]
    return [d for d, _ in equity][1:], rets

def perf(equity, rf=0.0, periods=TRADING_DAYS):
    dates, r = to_returns(equity)
    n = len(r)
    if n < 2: return {"error": "too few observations"}
    total = equity[-1][1]/equity[0][1] - 1
    cagr = (1+total)**(periods/n) - 1
    vol = r.std(ddof=1)*math.sqrt(periods)
    sharpe = (r.mean()*periods - rf)/vol if vol > 0 else float("nan")
    downside = r[r < 0]
    sortino = (r.mean()*periods - rf)/(downside.std(ddof=1)*math.sqrt(periods)) if len(downside) > 1 else float("nan")
    vals = np.array([v for _, v in equity])
    peak = np.maximum.accumulate(vals)
    dd = vals/peak - 1
    mdd = dd.min()
    # 修复期：从谷底回到前高所需交易日
    trough = int(dd.argmin())
    recover = None
    for i in range(trough, len(vals)):
        if vals[i] >= peak[trough]:
            recover = i - trough; break
    calmar = cagr/abs(mdd) if mdd < 0 else float("nan")
    # Newey-West t 统计量（滞后取 n**(1/4)）
    lag = max(1, int(round(n**0.25)))
    X = np.ones((n, 1))
    _b, _t = _ols_hac(r, X, lag)
    t_nw = float(_t[0])
    # Deflated Sharpe（Bailey & Lopez de Prado）
    T = n
    skew = float(sps.skew(r)); kurt = float(sps.kurtosis(r, fisher=False))
    sr = r.mean()/r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    se_sr = math.sqrt((1 - skew*sr + (kurt-1)/4*sr*sr)/(T-1))
    psr = float((1 - sps.norm_sf((sr - 0.0)/se_sr))) if se_sr > 0 else float("nan")
    return {
        "days": n,
        "start": equity[0][0], "end": equity[-1][0],
        "total_return_pct": round(total*100, 2),
        "cagr_pct": round(cagr*100, 2),
        "ann_vol_pct": round(vol*100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd*100, 2),
        "recovery_days": recover,
        "calmar": round(calmar, 3),
        "skew": round(skew, 3), "excess_kurtosis": round(kurt-3, 3),
        "t_stat_newey_west": round(t_nw, 3),
        "probabilistic_sharpe": round(psr, 4),
        "hit_rate_pct": round(float((r > 0).mean())*100, 2),
        "best_day_pct": round(float(r.max())*100, 2),
        "worst_day_pct": round(float(r.min())*100, 2),
    }

def alpha_beta(equity, bench_equity, rf=0.0):
    """对基准做 OLS：alpha（年化）、beta、R²、t 值。"""
    d1, r1 = to_returns(equity); d2, r2 = to_returns(bench_equity)
    common = sorted(set(d1) & set(d2))
    if len(common) < 30: return {"error": "insufficient overlap"}
    i1 = {d: i for i, d in enumerate(d1)}; i2 = {d: i for i, d in enumerate(d2)}
    y = np.array([r1[i1[d]] for d in common]); x = np.array([r2[i2[d]] for d in common])
    X = np.column_stack([np.ones_like(x), x])
    _b, _t = _ols_hac(y, X, max(1, int(round(len(common)**0.25))))
    resid = y - X @ _b
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p_alpha = math.erfc(abs(float(_t[0])) / math.sqrt(2))
    return {
        "obs": len(common),
        "alpha_annual_pct": round(float(_b[0])*TRADING_DAYS*100, 2),
        "alpha_t": round(float(_t[0]), 3),
        "alpha_p": round(p_alpha, 4),
        "beta": round(float(_b[1]), 3),
        "beta_t": round(float(_t[1]), 3),
        "r_squared": round(r2, 4),
    }

def _rank(v):
    """平均秩（并列取平均），与 scipy.stats.rankdata 同义。"""
    a = np.asarray(v, dtype=float)
    order = a.argsort()
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    # 并列项取平均秩
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for idx in np.where(cnt > 1)[0]:
        mask = inv == idx
        ranks[mask] = ranks[mask].mean()
    return ranks


def ic_signal(scores_by_date, fwd_returns_by_date):
    """日度 IC（Spearman 秩相关）与 ICIR。"""
    ics = []
    for d in sorted(set(scores_by_date) & set(fwd_returns_by_date)):
        sc = scores_by_date[d]; r = fwd_returns_by_date[d]
        common = [c for c in sc if c in r]
        if len(common) < 10: continue
        a = [sc[c] for c in common]; b = [r[c] for c in common]
        ra = _rank(a); rb = _rank(b)
        rho = float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")
        if rho == rho: ics.append(float(rho))
    if len(ics) < 10: return {"error": "too few IC observations"}
    arr = np.array(ics)
    icir = arr.mean()/arr.std(ddof=1)*math.sqrt(TRADING_DAYS) if arr.std(ddof=1) > 0 else float("nan")
    t = arr.mean()/(arr.std(ddof=1)/math.sqrt(len(arr)))
    return {"n": len(arr), "mean_ic": round(float(arr.mean()), 4), "ic_std": round(float(arr.std(ddof=1)), 4),
            "icir_annual": round(float(icir), 3), "ic_t": round(float(t), 3),
            "ic_positive_pct": round(float((arr > 0).mean())*100, 1)}


def monthly_table(equity):
    """按自然月聚合，返回 [(YYYY-MM, 收益率%)]，并给出月度胜率。"""
    import collections
    months = collections.OrderedDict()
    for d, v in equity:
        months.setdefault(d[:7], []).append(v)
    out = []
    prev_end = equity[0][1]
    for m, vals in months.items():
        r = vals[-1]/prev_end - 1
        out.append((m, round(r*100, 2)))
        prev_end = vals[-1]
    wins = sum(1 for _, r in out if r > 0)
    return out, round(wins/max(len(out), 1)*100, 1)

def rolling_sharpe(equity, window=250):
    """滚动 1 年夏普，用来看策略是不是只在某一段行情里有效。"""
    dates, r = to_returns(equity)
    vals = []
    for i in range(window, len(r)+1):
        seg = r[i-window:i]
        sd = seg.std(ddof=1)
        vals.append((dates[i-1], round(float(seg.mean()/sd*TRADING_DAYS**0.5), 2) if sd > 0 else None))
    return vals

def stationarity_hint(equity):
    """收益的一阶自相关与 Ljung-Box：检验"策略收益是否可复现"的最小证据。"""
    dates, r = to_returns(equity)
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 2 else None
    return {"autocorr_lag1": round(ac1, 4) if ac1 is not None else None,
            "ljung_box_p_5": _ljung_box_p(r, 5),
            "ljung_box_p_10": _ljung_box_p(r, 10)}

def walk_forward(split_ratio=0.6):
    """占位说明 —— 真正的 walk-forward 在 run_backtest 里按时间切分执行。"""
    return {"note": "见 run_backtest.walk_forward_split"}


def deflated_sharpe(sr_observed, n_trials, T, skew=0.0, kurt=3.0, sr_variance_across_trials=None):
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)。

    关键点：在试了 n_trials 个配置之后，"最好那个" 的夏普天然偏高。
    这里用多次试验的最大值期望 E[max SR] 去扣这个选择偏差，返回扣完之后仍然显著的概率。
    """
    if T < 3 or n_trials < 1:
        return {"error": "insufficient data"}
    euler = 0.5772156649
    var_sr = sr_variance_across_trials if sr_variance_across_trials else 1.0/T
    # 期望最大值（Gumbel 近似）
    e_max = (1-euler)*_norm_ppf(1-1.0/n_trials) + euler*_norm_ppf(1-1.0/(n_trials*2.718281828))
    sr0 = e_max*math.sqrt(var_sr)
    denom = math.sqrt(max(1e-12, (1 - skew*sr_observed + (kurt-1)/4.0*sr_observed*sr_observed)/(T-1)))
    z = (sr_observed - sr0)/denom
    return {"trials": n_trials, "sr_observed": round(sr_observed, 4),
            "sr_expected_max_from_selection": round(sr0, 4),
            "deflated_z": round(z, 3), "deflated_prob": round(float(1 - sps.norm_sf(z)), 4),
            "survives_selection_bias": bool(1 - sps.norm_sf(z) > 0.95)}

def pbo(returns_matrix):
    """Probability of Backtest Overfitting（组合对称交叉验证，简化版）。

    returns_matrix: {配置名: [日收益...]}，长度对齐。
    把时间轴切成 S 段，轮流拿一段做"样本内挑最优"，看它在样本外是否还在前半。
    """
    names = list(returns_matrix)
    mat = np.array([returns_matrix[n] for n in names], dtype=float)
    T = mat.shape[1]
    S = 8
    seg = T//S
    if seg < 20: return {"error": "too few observations"}
    out_of_sample_ranks = []
    for s in range(S):
        test = slice(s*seg, (s+1)*seg)
        train = np.r_[0:s*seg, (s+1)*seg:S*seg]
        sr_train = mat[:, train].mean(axis=1)/(mat[:, train].std(axis=1, ddof=1)+1e-12)
        best = int(np.argmax(sr_train))
        sr_test = mat[:, test].mean(axis=1)/(mat[:, test].std(axis=1, ddof=1)+1e-12)
        rank = float((sr_test < sr_test[best]).sum())/len(names)
        out_of_sample_ranks.append(rank)
    med = float(np.median(out_of_sample_ranks))
    return {"segments": S, "median_oos_rank": round(med, 3),
            "pbo": round(float((np.array(out_of_sample_ranks) <= 0.5).mean()), 3),
            "note": "pbo 越高说明样本内最优越难在样本外复现"}


# ---------------------------------------------------------------- 纯标准库实现
# 全市场版必须用这些：沙箱里 numpy 数组一大就 MemoryError（见 METHOD.md 第四节）。

def perf_py(equity, rf=0.0, periods=TRADING_DAYS):
    vals = [v for _, v in equity]
    if len(vals) < 3: return {"error": "too few observations"}
    r = [vals[i]/vals[i-1] - 1 for i in range(1, len(vals)) if vals[i-1] > 0]
    n = len(r)
    if n < 2: return {"error": "too few observations"}
    mean = sum(r)/n
    var = sum((x-mean)**2 for x in r)/(n-1)
    sd = math.sqrt(var)
    total = vals[-1]/vals[0] - 1
    cagr = (1+total)**(periods/n) - 1 if total > -1 else -1.0
    vol = sd*math.sqrt(periods)
    sharpe = (mean*periods - rf)/vol if vol > 0 else float("nan")
    down = [x for x in r if x < 0]
    dsd = math.sqrt(sum(x*x for x in down)/(len(down)-1)) if len(down) > 1 else float("nan")
    sortino = (mean*periods - rf)/(dsd*math.sqrt(periods)) if dsd == dsd and dsd > 0 else float("nan")
    peak = vals[0]; mdd = 0.0; trough_i = 0
    for i, v in enumerate(vals):
        if v > peak: peak = v
        dd = v/peak - 1
        if dd < mdd: mdd = dd; trough_i = i
    recover = None
    pv = max(vals[:trough_i+1])
    for i in range(trough_i, len(vals)):
        if vals[i] >= pv: recover = i - trough_i; break
    m2 = sum((x-mean)**2 for x in r)/n; m3 = sum((x-mean)**3 for x in r)/n; m4 = sum((x-mean)**4 for x in r)/n
    skew = m3/(m2**1.5) if m2 > 0 else 0.0
    kurt = m4/(m2*m2) if m2 > 0 else 3.0
    L = max(1, int(round(n**0.25)))
    lrv = sum((x-mean)**2 for x in r)/n
    for l in range(1, L+1):
        gl = sum((r[i]-mean)*(r[i-l]-mean) for i in range(l, n))/n
        lrv += 2*(1-l/(L+1))*gl
    se = math.sqrt(max(lrv, 0)/n)
    t_nw = mean/se if se > 0 else float("nan")
    sr_d = mean/sd if sd > 0 else 0.0
    se_sr = math.sqrt(max(1e-12, (1 - skew*sr_d + (kurt-1)/4.0*sr_d*sr_d)/(n-1)))
    psr = 0.5*(1+math.erf((sr_d/se_sr)/math.sqrt(2))) if se_sr > 0 else float("nan")
    return {
        "days": n, "start": equity[0][0], "end": equity[-1][0],
        "total_return_pct": round(total*100, 2), "cagr_pct": round(cagr*100, 2),
        "ann_vol_pct": round(vol*100, 2), "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3) if sortino == sortino else None,
        "max_drawdown_pct": round(mdd*100, 2), "recovery_days": recover,
        "calmar": round(cagr/abs(mdd), 3) if mdd < 0 else None,
        "skew": round(skew, 3), "excess_kurtosis": round(kurt-3, 3),
        "t_stat_newey_west": round(t_nw, 3), "probabilistic_sharpe": round(psr, 4),
        "hit_rate_pct": round(100.0*sum(1 for x in r if x > 0)/n, 2),
        "best_day_pct": round(max(r)*100, 2), "worst_day_pct": round(min(r)*100, 2),
        "final_value": round(vals[-1], 2),
    }


def alpha_beta_py(equity, bench_equity, periods=TRADING_DAYS):
    ev = dict(equity); bv = dict(bench_equity)
    common = sorted(set(ev) & set(bv))
    if len(common) < 60: return {"error": "insufficient overlap"}
    y = []; x = []
    for i in range(1, len(common)):
        a0, a1 = common[i-1], common[i]
        if bv[a0] > 0 and ev[a0] > 0:
            y.append(ev[a1]/ev[a0]-1); x.append(bv[a1]/bv[a0]-1)
    n = len(y)
    if n < 30: return {"error": "insufficient overlap"}
    mx = sum(x)/n; my = sum(y)/n
    sxx = sum((v-mx)**2 for v in x); sxy = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    beta = sxy/sxx if sxx > 0 else 0.0
    alpha = my - beta*mx
    resid = [y[i]-alpha-beta*x[i] for i in range(n)]
    s2 = sum(e*e for e in resid)/(n-2)
    se_a = math.sqrt(s2*(1.0/n + mx*mx/sxx)) if sxx > 0 else float("nan")
    t_a = alpha/se_a if se_a and se_a > 0 else float("nan")
    sst = sum((v-my)**2 for v in y)
    r2 = 1 - sum(e*e for e in resid)/sst if sst > 0 else float("nan")
    return {"obs": n, "alpha_annual_pct": round(alpha*periods*100, 2),
            "alpha_t": round(t_a, 3) if t_a == t_a else None,
            "beta": round(beta, 3), "r_squared": round(r2, 4)}


def _ppf(p):
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo+hi)/2
        if 0.5*(1+math.erf(mid/math.sqrt(2))) < p: lo = mid
        else: hi = mid
    return (lo+hi)/2


def deflated_sharpe_py(sr_observed, n_trials, T, skew=0.0, kurt=3.0, sr_variance_across_trials=None):
    """注意：负的夏普这里没有意义。策略跑输基准（SR<0）时 DSR 只会更小，
    负数说明它连"瞎猜"都不如，不要把正数当成"通过"。"""
    if T < 3 or n_trials < 1: return {"error": "insufficient data"}
    if sr_observed <= 0:
        return {"trials": n_trials, "sr_observed": round(sr_observed, 4),
                "deflated_z": None, "deflated_prob": None, "survives_selection_bias": False,
                "note": "夏普为负，无需检验：策略跑输无风险资产"}
    euler = 0.5772156649
    var_sr = sr_variance_across_trials if sr_variance_across_trials else 1.0/T
    e_max = (1-euler)*_ppf(1-1.0/n_trials) + euler*_ppf(1-1.0/(n_trials*2.718281828))
    sr0 = e_max*math.sqrt(var_sr)
    denom = math.sqrt(max(1e-12, (1 - skew*sr_observed + (kurt-1)/4.0*sr_observed*sr_observed)/(T-1)))
    z = (sr_observed - sr0)/denom
    p = 0.5*(1+math.erf(z/math.sqrt(2)))
    return {"trials": n_trials, "sr_observed": round(sr_observed, 4),
            "sr_expected_max_from_selection": round(sr0, 4),
            "deflated_z": round(z, 3), "deflated_prob": round(p, 4),
            "survives_selection_bias": bool(p > 0.95)}


def pbo_py(returns_matrix, S=8):
    names = list(returns_matrix)
    T = min(len(v) for v in returns_matrix.values())
    seg = T//S
    if seg < 20: return {"error": "too few observations"}
    mat = [returns_matrix[n][:T] for n in names]

    def sr(series, sl):
        m_ = sum(series[k] for k in sl)/len(sl)
        v = sum((series[k]-m_)**2 for k in sl)/(len(sl)-1)
        return m_/math.sqrt(v) if v > 0 else 0.0

    ranks = []
    for s in range(S):
        test = list(range(s*seg, (s+1)*seg))
        train = list(range(0, s*seg)) + list(range((s+1)*seg, S*seg))
        srs = [sr(x, train) for x in mat]
        best = max(range(len(names)), key=lambda i: srs[i])
        tsr = [sr(x, test) for x in mat]
        ranks.append(sum(1 for v in tsr if v < tsr[best])/len(names))
    ranks.sort()
    return {"segments": S, "median_oos_rank": round(ranks[len(ranks)//2], 3),
            "pbo": round(sum(1 for x in ranks if x <= 0.5)/len(ranks), 3),
            "note": "pbo 越高说明样本内最优越难在样本外复现（纯标准库版）"}


def yearly_table(equity):
    out = {}
    for d, v in equity:
        y = d[:4]
        if y not in out: out[y] = [v, v]
        out[y][1] = v
    res = {}
    for y in sorted(out):
        a, b = out[y]
        res[y] = round((b/a - 1)*100, 2)
    return res
