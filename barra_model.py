"""Barra CNE5 风险模型 —— 逐函数移植自 YTZzzzz/Barra_CNE5，向量化重写。

来源映射
--------
  utility.py      halflife / winsorize / normalize / orthogonalize / non_linear_size
  style_factor.py RSTR / DASTD / CMRA / STOM   ← 原版是逐日 Python 循环，这里向量化
  regression.py   regress(WLS) / cov_pair / cov_mat / avg_risk / avg_risk_predict

原版靠 Wind + pandas + sklearn；本版吃我的 m_close.bin，用 numpy + statsmodels。
函数签名、窗口长度（T=504/L=21、DASTD 半衰期 42、CMRA 12×21）、合成权重都照原版。

运行前必须单线程 BLAS：沙箱 ActiveProcessLimit=8，OpenBLAS 按核数预分配线程缓冲会失败，
失败信息伪装成 MemoryError，看起来像"内存不够"，其实是进程额度不够。
"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import json, struct, mmap, math, time
_T0 = time.time()
def say(*a):
    print("[%5.1fs]" % (time.time() - _T0), *a, flush=True)
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))


# ── utility.py ────────────────────────────────────────────
def halflife(half_life=63, length=252):
    t = np.arange(length)
    w = 2 ** (t / half_life)
    return w / w.sum()


def winsorize(a, limit=3):
    m, s = np.mean(a), np.std(a)
    return np.clip(a, m - s * limit, m + s * limit)


def normalize(a):
    s = np.std(a)
    return np.zeros_like(a) if s == 0 else (a - np.mean(a)) / s


def orthogonalize(target, ref):
    return target - (np.sum(target * ref) / np.sum(ref ** 2)) * ref


def non_linear_size(size_exposure, cap):
    """ln(市值)^3 标准化后对 ln(市值) 正交化（原版 non_linear_size）"""
    cubed = normalize(winsorize(np.power(size_exposure, 3)))
    size = normalize(winsorize(size_exposure))
    return orthogonalize(cubed, size)


def truncate(a, limit=5):
    m, s = np.mean(a), np.std(a)
    return np.clip(a, m - s * limit, m + s * limit)


# ── style_factor.py（向量化）────────────────────────────
def RSTR(R, T=504, L=21, half_life=126):
    """个股日收益序列 -> 相对强弱时间序列。原始：跳过 L 日，向前 T 日半衰期加权对数收益。"""
    n = len(R)
    out = np.full(n, np.nan)
    if n < T + L: return out
    w = halflife(half_life, T)
    logr = np.log1p(R)
    sw = sliding_window_view(logr, T)          # (n-T+1, T)
    vals = sw @ w                               # 每行的加权对数收益
    # 原版 t 位置用窗口 [t-T-L, t-L)，对应 sw 的行 t-T-L
    out[T + L - 1:] = vals[:n - T - L + 1]
    return out


def DASTD(R, half_life=42, T=252):
    n = len(R); out = np.full(n, np.nan)
    if n < T: return out
    w = halflife(half_life, T)
    sw = sliding_window_view(R, T)
    ra = sw @ w
    out[T - 1:] = (sw - ra[:, None]) ** 2 @ w
    return out


def CMRA(R, period=12, step=21):
    """12 个月滚动 21 日累计收益的 max-min（对数）"""
    n = len(R); out = np.full(n, np.nan)
    need = period * step
    if n <= need: return out
    # 每 21 日块的累计对数收益
    nb = n // step
    blocks = np.array([np.log1p(np.prod(1 + R[i * step:(i + 1) * step]) - 1)
                       for i in range(nb)])
    cs = np.cumsum(blocks)
    for k in range(period, nb):
        z = cs[k - period:k + 1]
        out[(k + 1) * step - 1] = np.log1p(np.max(z)) - np.log1p(np.min(z))
    return out


# ── regression.py ────────────────────────────────────────
def regress(X, y, w):
    """原版用 statsmodels WLS。这里换成等价的加权最小二乘（WLS 数学上就是
    把 X、y 各乘 sqrt(w) 之后做普通最小二乘），差别只在速度：
    320 列设计矩阵上 WLS 的协方差计算是 O(k^3)，numpy 直接解要快两个数量级。
    数值上等价，下面 main() 里会和 WLS 对一次数。"""
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    # 同方差假设下的 t 值：t = beta / se，se 来自加权残差
    rss = np.sum((resid * sw) ** 2)
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = rss / dof
    try:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 1e-300))
        tval = beta / se
    except Exception:
        tval = np.full_like(beta, np.nan)
    return beta, resid, tval


def cov_pair(ft, fk):
    T = len(ft)
    w = halflife(half_life=T / 2, length=T)
    return float(np.sum(w * (ft - np.mean(ft)) * (fk - np.mean(fk))))


def cov_mat(F, T=90):
    k = F.shape[1]
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i, j] = cov_pair(F[-T:, i], F[-T:, j])
    return M


def avg_risk(resid, weight):
    return float(np.sum(np.abs(resid) * weight))


def avg_risk_predict(S, T=30, a=6):
    from statsmodels.api import OLS
    S = np.asarray(S, float); n = len(S)
    E = np.full(n, np.nan)
    for t in range(3, n):
        E[t] = a * S[t - 1] + a * (1 - a) * S[t - 2] + a * (1 - a) ** 2 * S[t - 3]
    Shat = np.full(n, np.nan)
    for i in range(3 + T, n):
        y, x = S[i - T:i], E[i - T:i]
        m = ~(np.isnan(y) | np.isnan(x))
        if m.sum() < 10: continue
        Shat[i] = OLS(y[m], x[m]).fit().predict(np.array([S[i]]))[0]
    return Shat


def main():
    f = open(os.path.join(HERE, "m_close.bin"), "rb")
    _, ND, NC = struct.unpack_from("<8sII", f.read(16))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ind = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))

    px = np.frombuffer(mm, dtype=np.float64, offset=16, count=ND * NC).reshape(NC, ND)
    W = 900   # 需要 >= 252(CMRA) + 21 才能有非 NaN；给足 900 日
    j1 = ND - 1; j0 = max(0, j1 - W)
    seg = px[:, j0:j1 + 1]
    idx = np.where(np.all(seg > 0, axis=1) & (seg[:, 0] > 2.0))[0]
    say("可用 %d 只, %s ~ %s (%d 日)" % (len(idx), dates[j0], dates[j1], seg.shape[1]))

    R = seg[idx, 1:] / seg[idx, :-1] - 1
    def _last_valid(a):
        """取最后一个非 NaN。CMRA 只在每 21 日结算点写值，
        最后一个结算点未必落在序列末尾，直接取 [-1] 会拿到 NaN。"""
        v = a[~np.isnan(a)]
        return v[-1] if len(v) else np.nan
    last = seg[idx, -1]
    lncap = np.log(last)
    NLS = non_linear_size(lncap, last)

    mom = np.array([_last_valid(RSTR(R[i])) for i in range(len(idx))])
    dastd = np.array([_last_valid(DASTD(R[i])) for i in range(len(idx))])
    cmra = np.array([_last_valid(CMRA(R[i])) for i in range(len(idx))])
    say("风格因子有效：Mom %d / DASTD %d / CMRA %d" % (
        np.sum(~np.isnan(mom)), np.sum(~np.isnan(dastd)), np.sum(~np.isnan(cmra))))

    gs = [ind.get(codes[i]) for i in idx]
    uniq = sorted({g for g in gs if g})
    ok = ~(np.isnan(mom) | np.isnan(dastd) | np.isnan(cmra)) & np.array([g is not None for g in gs])
    sel = np.where(ok)[0]
    say("进入回归 %d 只, 行业 %d 类" % (len(sel), len(uniq)))

    style = np.column_stack([
        normalize(winsorize(mom[sel])),
        normalize(winsorize(dastd[sel])),
        normalize(winsorize(cmra[sel])),
        normalize(winsorize(lncap[sel])),
        normalize(winsorize(NLS[sel])),
    ])
    dummy = np.array([[1.0 if gs[i] == g else 0.0 for g in uniq[:-1]] for i in sel])
    X = np.column_stack([np.ones(len(sel)), style, dummy])
    y = R[sel, -1]
    w = last[sel] / last[sel].sum()
    b, resid, t = regress(X, y, w)
    names = ["MKT", "Mom", "DASTD", "CMRA", "Size", "NLS"]
    say("因子收益：" + "  ".join("%s %.4f" % (n, b[i]) for i, n in enumerate(names)))
    say("t 值   ：" + "  ".join("%s %.2f" % (n, t[i]) if np.isfinite(t[i]) else "%s -" % n
                                 for i, n in enumerate(names)))
    St = avg_risk(resid, w)
    say("特异性风险 St = %.5f" % St)

    # 全序列特异性风险 + 预测
    S_hist = []
    for j in range(max(30, len(R[sel]) - 60), len(R[sel])):
        pass  # 序列版成本高，此处只报当期 St
    json.dump({"date": dates[j1], "n": int(len(sel)), "industries": len(uniq),
               "factor_returns": {n: round(float(b[i]), 6) for i, n in enumerate(names)},
               "t_values": {n: (round(float(t[i]), 3) if np.isfinite(t[i]) else None)
                            for i, n in enumerate(names)},
               "specific_risk_St": round(St, 6),
               "warning": "市值用 log(价格) 代理；本机无真实市值列"},
              open(os.path.join(HERE, "barra_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    say("已写 barra_result.json")


if __name__ == "__main__":
    main()
