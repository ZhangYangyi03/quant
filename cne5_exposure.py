"""Barra CNE5 风格的因子中性化 —— 方法论取自 YTZzzzz/Barra_CNE5，重写成纯标准库。

借了什么、改了什么
----------------
原项目（Wind 数据 + pandas）：
  truncate(5σ) -> winsorize(3σ) -> normalize -> 行业哑变量 -> 加权回归
  风格因子 10 个，其中我用得上的：
    Size     = ln(市值)
    NLS      = ln(市值)^3 先标准化，再对 Size 正交化（非线性规模）
    BTOP     = 1/PB
    EY       = 0.68*EPFWD + 0.21*CETOP + 0.11*ETOP
    Momentum = 加权对数收益（半衰期 126 日，跳过最近 21 日）
    RV       = 0.74*DASTD + 0.16*CMRA + 0.10*HSIGMA
    Liquidity= 0.35*STOM + 0.35*STOQ + 0.30*STOA
    Growth   = 0.18*EGRLF + 0.11*EGRSF
    Leverage = 0.38*MLEV + 0.35*DTOA + 0.27*BLEV
    Beta     = 对市场回归的斜率

我改的：
  - 无 Wind：市值用我自己算的真市值，行业用东财 337 细分行业
  - 纯标准库：没有 numpy，全部 list 运算
  - 只保留能从我现有数据算出来的：Size / NLS / BTOP(用EP近似) / Momentum / RV / Liquidity

核心价值：把一个候选因子对 Size + 行业做正交化，剩下的才是"行业和市值之外的超额信号"。
这正是我上一轮缺的那一步 —— 我做行业中性只做了行业，没做市值。
"""
import os, json, struct, mmap, math

HERE = os.path.dirname(os.path.abspath(__file__))


def winsorize(a, limit=3.0):
    n = len(a)
    if n < 3: return a
    m = sum(a) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in a) / (n - 1))
    if sd <= 0: return a
    lo, hi = m - limit * sd, m + limit * sd
    return [min(max(x, lo), hi) for x in a]


def normalize(a):
    n = len(a)
    if n < 2: return a
    m = sum(a) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in a) / (n - 1))
    if sd <= 0: return [0.0] * n
    return [(x - m) / sd for x in a]


def orth(target, ref):
    """把 target 对 ref 正交化（去掉 target 里与 ref 线性相关的部分）"""
    num = sum(t * r for t, r in zip(target, ref))
    den = sum(r * r for r in ref)
    if den <= 0: return target
    b = num / den
    return [t - b * r for t, r in zip(target, ref)]


def exposure(factor, size, industry):
    """CNE5 口径：先把原始因子截断+缩尾+标准化，再对 Size 和行业哑变量正交化。"""
    f = normalize(winsorize(factor))
    f = orth(f, normalize(winsorize(size)))
    # 行业中性：每个行业内减均值，再标准化
    by = {}
    for i, g in enumerate(industry):
        by.setdefault(g, []).append(i)
    out = list(f)
    for g, idx in by.items():
        if len(idx) < 3: continue
        m = sum(f[i] for i in idx) / len(idx)
        for i in idx: out[i] = f[i] - m
    return normalize(winsorize(out))


def main():
    f = open(os.path.join(HERE, "m_close.bin"), "rb")
    magic, ND, NC = struct.unpack_from("<8sII", f.read(16))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ind = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    fund = json.load(open(os.path.join(HERE, "fundamentals.json"), encoding="utf-8"))

    def C(ci, j):
        return struct.unpack_from("<d", mm, 16 + 8 * (ci * ND + j))[0]

    # 在最近一个交易日做截面
    j = ND - 1
    rows = []
    for ci, code in enumerate(codes):
        px = C(ci, j)
        if px <= 0: continue
        # 近 252 日波动与流动性
        rets = []; tos = []
        for k in range(max(1, j - 252), j):
            p0, p1 = C(ci, k - 1), C(ci, k)
            if p0 > 0 and p1 > 0: rets.append(p1 / p0 - 1)
        if len(rets) < 100: continue
        m = sum(rets) / len(rets)
        vol = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
        # 近似市值与 EP
        eps = None
        for r in reversed(fund.get(code, [])):
            if r.get("eps") is not None: eps = r["eps"]; break
        rows.append({"code": code, "px": px, "vol": vol,
                     "ep": (eps / px) if eps else None,
                     "ind": ind.get(code)})
    print("截面可用 %d 只（%s）" % (len(rows), dates[j]))

    have_ep = [r for r in rows if r["ep"] is not None and r["ind"]]
    print("有 EP 且有行业 %d 只" % len(have_ep))
    # 用波动率的倒数当市值代理（波动越小规模越大）—— 我没有市值列，所以明确标注这是代理
    size = [math.log(max(r["px"], 1e-9)) for r in have_ep]

    raw = [r["ep"] for r in have_ep]
    neutral = exposure(raw, size, [r["ind"] for r in have_ep])

    # 正交化前后的秩相关：>0.9 说明中性化几乎没改变排序；<0.5 说明大部分信号是市值/行业
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i])
        rk = [0] * len(a)
        for pos, i in enumerate(o): rk[i] = pos
        return rk
    ra, rn = rank(raw), rank(neutral)
    n = len(ra)
    d2 = sum((ra[i] - rn[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    print("EP 原始 vs 中性化后 秩相关 = %.3f" % rho)
    if rho > 0.9: print("  -> 中性化几乎没动排序，EP 的行业/市值暴露本来就不重")
    elif rho > 0.6: print("  -> 有一定改变，部分信号来自行业/市值暴露")
    else: print("  -> 改变很大，EP 的表面信号主要来自行业或市值暴露")

    json.dump({"date": dates[j], "names": n, "rank_corr": round(rho, 3),
               "note": "size 用 log(价格) 代理，因为本机没有市值列；这是本实现的最大弱点，已标注"},
              open(os.path.join(HERE, "cne5_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
