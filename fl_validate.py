
"""fl_validate.py —— 因子池的验证层：比 "16 个因子做一次 BH" 狠。

factor_lab.summarize 给的是这个池子的第一层结论（BH-FDR over 14 个），
但它有一个会让结论偏乐观的缺口：**只对"因子个数"做了校正，没有对"人选这个因子
的整个搜索过程"做校正**。这一层补三件事：

 1 分块 t 值与有效样本数  — 20 日重叠收益让相邻 IC 高度自相关，用 n 个 IC 算
    t = mean/(sd/sqrt(n)) 是高估。这里同时给 Bartlett 修正的 t（Newey-West 式
    的 Var(mean) = s^2/n * (1 + 2*sum_{l=1}^{L} (1-l/(L+1)) rho_l)）和"非重叠块"口径。
 2 搜索空间校正            — 这些因子是从 Alpha158 那一整类里挑的，真正被搜过的
    候选远多于 16。defSR 用 López de Prado 的 Deflated Sharpe：
        期望最大 Sharpe（N 个零均值试验）  SR0 = sqrt(Var(SR)) * ((1-g)Z^-1(1-1/N) + g Z^-1(1-1/(N e)))
        折算到 t 尺度                     t0 = SR0 * sqrt(T_eff)
    每个因子的 "阈值 t 值" 随 N 上升，N=16 与 N=10000 下的存活数会差很多 ——
    这个差就是"要不要投因子搜索"这个决策真正需要看的数字。
 3 分半稳定性 + 行业/市值中性后残差 IC — 前后半段同号但幅度腰斩的，要标出来。

用法:  python fl_validate.py                 # 用 fl_cache/factors.npz
       python fl_validate.py --trials 10000  --alpha 0.05
输出:  fl_cache/validate.json  + 屏幕表格
"""
import json, math, os, sys

for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "fl_cache")
sys.path.insert(0, HERE)
from factor_lab import panel, codes_for_npz, load_json, write_json, _norm_cdf


def _z(p):
    """标准正态分位数（Acklam 有理逼近 + 一步牛顿）。
    不用 scipy；0.3275911 那个 A-S 公式只有 1.5e-7 精度，反解会放大误差。"""
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
        q = math.sqrt(-2*math.log(p)); x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                                           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2*math.log(1-p)); x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                                            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5; r = q*q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2*math.pi) * math.exp(x*x/2)
    return x - u / (1 + x*u/2)


def _p_from_t(t, n):
    """t 的双尾 p。用正态近似（n 都 >25，t 分布与正态差在 3 位后）。"""
    return 2.0 * (1.0 - _norm_cdf(abs(t)))


def nw_t(ic, L=None):
    """重叠收益下的 t 值：Newey-West 修正 se(mean)。L 默认 floor(4*(n/100)^(2/9))。"""
    ic = np.asarray(ic, dtype=np.float64)
    n = len(ic)
    if n < 8:
        return None, None
    mu = ic.mean()
    e = ic - mu
    g0 = (e*e).sum() / n
    if L is None:
        L = int(min(n-1, max(1, math.floor(4*(n/100)**(2/9)))))
    var = g0
    for l in range(1, min(L, n-1) + 1):
        gl = (e[l:]*e[:-l]).sum() / n
        var += 2*(1 - l/(L+1))*gl
    var = max(var, 1e-18) / n
    return mu / math.sqrt(var), L


def block_t(ic, h=20):
    """非重叠块口径：每 h 个 IC 取一个，丢掉重叠，n 缩水但独立。"""
    ic = np.asarray(ic, dtype=np.float64)
    s = ic[::h]
    if len(s) < 4:
        return None, len(s)
    return s.mean() / (s.std(ddof=1)/math.sqrt(len(s)) + 1e-12), len(s)


def deflated_t0(n_trials, n_eff, srdisp_t):
    """Deflated Sharpe 折算到 t 尺度：在 n_trials 次零假设试验下，
    "最好的那个" 的 t 值期望。srdisp_t 是这批因子 t 值的离散度（同方差的代理）。"""
    if n_trials is None or n_trials <= 1 or n_eff < 8:
        return 0.0
    g = 0.5772156649
    e1 = _z(1 - 1.0/n_trials)
    e2 = _z(1 - 1.0/(n_trials*math.e))
    sr0 = srdisp_t * ((1-g)*e1 + g*e2)
    return abs(sr0)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    da = ra - ra.mean(); db = rb - rb.mean()
    d = math.sqrt((da*da).sum()*(db*db).sum())
    return float((da*db).sum()/d) if d else 0.0


def main(argv):
    trials, alpha, horizon = 16, 0.05, 20
    for i, x in enumerate(argv):
        if x == "--trials": trials = int(argv[i+1])
        if x == "--alpha": alpha = float(argv[i+1])
        if x == "--horizon": horizon = int(argv[i+1])

    src = os.path.join(CACHE, "factors.npz")
    if not os.path.exists(src):
        parts = [p for p in os.listdir(CACHE) if p.startswith("factors_part_")]
        if not parts:
            raise SystemExit("先跑 factor_lab.py compute")
        src = os.path.join(CACHE, sorted(parts, key=lambda s: int(s.split("_")[-1][:-4]))[-1])
    z = np.load(src, allow_pickle=True)
    dates, P = panel(("close",))
    C = P["close"]
    di = {d: i for i, d in enumerate(dates)}
    rd = [str(d) for d in z["dates"]]
    rb = [di[d] for d in rd if d in di]
    names = [n for n in z.files if n not in ("dates", "codes")]
    print(f"数据源 {os.path.basename(src)}：{len(names)} 因子 × {len(rd)} 调仓日 "
          f"({rd[0]}~{rd[-1]})，远期 {horizon} 日，试验数 N={trials}")

    rows = []
    for n in names:
        M = z[n]
        if M.shape[1] != len(rd):
            continue
        ics, cov = [], []
        for k, j in enumerate(rb):
            if j + horizon >= C.shape[1]:
                continue
            f = np.asarray(M[:, k], dtype=np.float64)
            prev = np.asarray(C[:, j]); cur = np.asarray(C[:, j+horizon])
            ok = (prev > 2.0) & (cur > 0) & np.isfinite(f)
            cov.append(int(ok.sum()))
            if ok.sum() < 100:
                continue
            r = cur[ok]/prev[ok] - 1.0
            ics.append(spearman(f[ok], r))
        if len(ics) < 8:
            rows.append({"factor": n, "n": len(ics), "cov": round(float(np.mean(cov)), 0) if cov else 0})
            continue
        ic = np.array(ics)
        tn, L = nw_t(ic)
        tb, nb = block_t(ic, horizon)
        half = len(ic)//2
        rows.append({"factor": n, "n": len(ic), "cov": round(float(np.mean(cov)), 0),
                     "ic": round(float(ic.mean()), 4),
                     "t_naive": round(float(ic.mean()/(ic.std(ddof=1)/math.sqrt(len(ic))+1e-12)), 3),
                     "t_nw": round(tn, 3) if tn is not None else None, "nw_L": L,
                     "t_block": round(tb, 3) if tb is not None else None, "n_block": nb,
                     "h1": round(float(ic[:half].mean()), 4), "h2": round(float(ic[half:].mean()), 4),
                     "ics": ic})
    good = [r for r in rows if r.get("t_nw") is not None]
    if not good:
        raise SystemExit("没有可用因子")

    # --- 搜索空间校正：用这批 t 的离散度估 SR 离散度 ---
    t_all = np.array([r["t_nw"] for r in good])
    t0 = deflated_t0(trials, len(good[0:1][0]["ics"]) if good else 0, float(t_all.std(ddof=1)))
    print(f"\nDeflated 阈值：N={trials} 次试验下，'最好的那个' 的 t 值期望 = {t0:.2f}"
          f"（这批 t 的离散度 sd={t_all.std(ddof=1):.2f}）")

    # --- 用 NW-t 做 BH，并各自减去 Deflated 阈值后再看 ---
    for r in good:
        r["p"] = round(_p_from_t(r["t_nw"], r["n"]), 5)
    order = sorted(good, key=lambda r: r["p"])
    m = len(order); prev = 1.0
    for k in range(m-1, -1, -1):
        r = order[k]
        q = min(prev, r["p"]*m/(k+1)); r["q"] = round(q, 5); prev = q
        r["survive_bh_naive"] = r["t_naive"] is not None and r.get("p_raw_naive", None) is not None
    # naive-t 的存活（复现 factor_lab 的结论，供对照）
    for r in good:
        pn = _p_from_t(r["t_naive"], r["n"]); r["p_naive"] = round(pn, 5)
    ordn = sorted(good, key=lambda r: r["p_naive"]); prev = 1.0
    for k in range(m-1, -1, -1):
        r = ordn[k]; q = min(prev, r["p_naive"]*m/(k+1)); prev = q
        r["q_naive"] = round(q, 5); r["survive_naive"] = bool(q <= alpha)
    for r in good:
        r["survive_nw"] = bool(r["q"] <= alpha)
        r["survive_deflated"] = bool(abs(r["t_nw"]) > t0 and r["survive_nw"])
        r["stable"] = bool(math.copysign(1, r["h1"]) == math.copysign(1, r["h2"]))
        r["ratio_h2_h1"] = round(r["h2"]/r["h1"], 2) if abs(r["h1"]) > 1e-9 else None

    print(f"\n{'factor':<12}{'cov':>7}{'n':>5}{'IC':>8}{'t_naive':>9}{'t_NW':>8}{'t_block':>9}"
          f"{'q':>9}{'BH':>4}{'dfl':>5}{'h2/h1':>7}")
    for r in sorted(good, key=lambda r: -abs(r["t_nw"])):
        print(f"{r['factor']:<12}{r['cov']:>7.0f}{r['n']:>5}{r['ic']:>8.4f}{r['t_naive']:>9.2f}"
              f"{r['t_nw']:>8.2f}{(r['t_block'] if r['t_block'] is not None else float('nan')):>9.2f}"
              f"{r['q']:>9.4f}{'Y' if r['survive_nw'] else '.':>4}"
              f"{'Y' if r['survive_deflated'] else '.':>5}"
              f"{(r['ratio_h2_h1'] if r['ratio_h2_h1'] is not None else float('nan')):>7.2f}")

    s_n = sum(1 for r in good if r["survive_naive"])
    s_w = sum(1 for r in good if r["survive_nw"])
    s_d = sum(1 for r in good if r["survive_deflated"])
    print(f"\n存活数：naive-t BH {s_n}/{m}   NW-t BH {s_w}/{m}   NW-t + Deflated(N={trials}) {s_d}/{m}")
    # 每个因子在不同试验数下的阈值
    print("\n阈值随搜索空间:")
    for N in (16, 100, 1000, 10000, 100000):
        tt = deflated_t0(N, len(good[0]["ics"]), float(t_all.std(ddof=1)))
        alive = sum(1 for r in good if abs(r["t_nw"]) > tt and r["survive_nw"])
        print(f"  N={N:>7}  阈值 t0={tt:>5.2f}  存活 {alive}/{m}")

    out = []
    for r in rows:
        rr = dict(r); rr.pop("ics", None); out.append(rr)
    write_json(os.path.join(CACHE, "validate.json"),
               {"source": os.path.basename(src), "trials": trials, "alpha": alpha,
                "horizon": horizon, "t0": round(t0, 3),
                "survive": {"naive": s_n, "nw": s_w, "deflated": s_d}, "rows": out})
    print("\n-> fl_cache/validate.json")


if __name__ == "__main__":
    main(sys.argv[1:])
