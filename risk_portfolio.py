# -*- coding: utf-8 -*-
"""risk_portfolio.py -- 给 size+amt20 加风险约束，并检验新因子正交化后还剩多少信息。

M154 的下一步。此前 mlfit2.py 的结论是 size+amt20 在测试段 1.27%/月、t=4.66、
换手 13.3%，但那个组合是"top 分位等权、无任何约束"。这一层回答三个问题：

  1. 组合约束（单票上限 / 行业集中度 / 换手带）会不会把超额吃掉？
  2. 把打分对 ln(市值) 和行业做中性化之后，还剩不剩 alpha？
     这是关键：size 因子本身就是 ln(市值)，如果收益全来自市值暴露，
     市值中性化会直接把它打回零。
  3. 17 因子里那些"新因子"是不是只是 size/amt20 的变体（正交化检验）。

口径与 mlfit2.py 完全一致：同一份 fscan_parts 面板、同一个方向表 ic_fast.json、
train < 2019-01-01 / test >= 2019-01-01、基准 = 当期可交易池等权收益、FEE=0.001。
新增的只是持仓层的权重与约束，不改动任何打分公式，因此结果可比。

用法：
    python risk_portfolio.py --mode portfolio    # 风险约束组合
    python risk_portfolio.py --mode orth         # 新因子正交化检验
"""
from __future__ import annotations
import os, sys, json, math, argparse, datetime
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraw
from compact import Panel
from fastpanel import FACTORS

FEE = 0.001
SPLIT = "2019-01-01"
K = 50                 # 约束组合的持仓只数
Q = 5                  # 复刻 mlfit2 的 top 分位用
PAIR = ("size", "amt20")


def zscore_row(v):
    ok = np.isfinite(v) & (np.abs(v) < 1e12)
    if ok.sum() < 200:
        return np.full(v.shape, np.nan, dtype=np.float64)
    med = np.median(v[ok]); sd = v[ok].std()
    return np.where(ok, (v - med) / (sd if sd > 0 else np.nan), np.nan)


def load_common():
    M = fraw.meta_unique()
    JJ = M["dates"]
    NS = len(JJ) - 1
    dates = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))["dates"]
    icj = json.load(open(os.path.join(HERE, "ic_fast.json"), encoding="utf-8"))
    dirn = {k: (1.0 if icj[k]["ic_mean"] >= 0 else -1.0) for k in icj}
    tr = [m for m in range(NS) if dates[int(JJ[m])] < SPLIT]
    te = [m for m in range(NS) if dates[int(JJ[m])] >= SPLIT]
    CLr = fraw.rows("close", 0, NS + 1, M).astype(np.float64)
    ALr = fraw.rows("alive", 0, NS, M)
    RET = np.empty((NS, CLr.shape[1]), dtype=np.float64)
    for m in range(NS):
        cur, nxt = CLr[m], CLr[m + 1]
        RET[m] = np.where((cur > 0) & (nxt > 0), nxt / np.where(cur > 0, cur, np.nan) - 1.0, np.nan)
    P = Panel()
    MKT = np.empty((NS, CLr.shape[1]), dtype=np.float64)
    for m in range(NS):
        MKT[m] = P.col("mktcap", int(JJ[m]))
    ind_map = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    codes = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))["codes"]
    names = sorted(set(ind_map.values()))
    id_of = {n: i for i, n in enumerate(names)}
    IND = np.array([id_of.get(ind_map.get(c), -1) for c in codes], dtype=np.int32)
    return dict(M=M, JJ=JJ, NS=NS, dates=dates, dirn=dirn, tr=tr, te=te,
                RET=RET, AL=ALr, MKT=MKT, IND=IND, n_ind=len(names), icj=icj)


def factor_rows(C, keys):
    """逐因子读成 (NS, NC) float64。17 因子全装约 77MB，这台机器分批更稳。"""
    out = {}
    NS = C["NS"]
    for k in keys:
        A = fraw.rows(k, 0, NS, C["M"]).astype(np.float64) * C["dirn"][k]
        for m in range(NS):
            A[m] = zscore_row(A[m])
        out[k] = A
    return out


def neutralize(score, mkt_z, ind, mode, valid):
    """对打分做横截面中性化，只在有效股票上算。"""
    s = score.copy()
    v = valid & np.isfinite(s)
    if v.sum() < 200:
        return s
    if mode in ("size", "both"):
        x = np.where(v & np.isfinite(mkt_z), mkt_z, 0.0)
        # 简单 OLS：s = a + b*x + e
        n = float(v.sum()); mx = x[v].mean(); my = s[v].mean()
        dx = x[v] - mx; den = float(dx @ dx)
        b = float(dx @ (s[v] - my)) / den if den > 0 else 0.0
        a = my - b * mx
        s = np.where(v, s - (a + b * x), s)
    if mode in ("ind", "both"):
        out = s.copy()
        for g in np.unique(ind[v]):
            idx = np.flatnonzero(v & (ind == g))
            if len(idx) >= 3:
                out[idx] = s[idx] - s[idx].mean()
        s = out
    return zscore_row(s)


def cap_weights(w, cap):
    """water-filling：把超过单票上限的权重按比例补给未触顶的票。"""
    if cap <= 0 or cap >= 1:
        return w
    w = w.copy()
    for _ in range(60):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        tot = float(w[under].sum())
        if tot <= 0:
            w[under] += excess / max(1, under.sum())
        else:
            w[under] += excess * w[under] / tot
    return w / w.sum()


def ind_weight_cap(w, ind, cap):
    """行业总权重上限；超出的钱按比例补给未触顶行业。"""
    if cap <= 0 or cap >= 1:
        return w
    w = w.copy()
    for _ in range(30):
        totals = {}
        for g in np.unique(ind):
            totals[g] = float(w[ind == g].sum())
        over = [g for g, t in totals.items() if t > cap + 1e-12]
        if not over:
            break
        freed = 0.0
        for g in over:
            idx = np.flatnonzero(ind == g)
            freed += float(w[idx].sum()) - cap
            w[idx] *= cap / float(w[idx].sum())
        under = np.flatnonzero(~np.isin(ind, over))
        if under.size == 0 or float(w[under].sum()) <= 0:
            break
        w[under] += freed * w[under] / float(w[under].sum())
    return w / w.sum()


def pick_top(score, valid, k, ind=None, per_ind_max=0):
    idx = np.flatnonzero(valid & np.isfinite(score))
    if len(idx) < k:
        return np.array([], dtype=int)
    order = idx[np.argsort(-score[idx], kind="stable")]
    if per_ind_max and ind is not None:
        used = {}
        keep = []
        for c in order:
            g = int(ind[c])
            if used.get(g, 0) >= per_ind_max:
                continue
            used[g] = used.get(g, 0) + 1
            keep.append(c)
            if len(keep) >= k:
                break
        return np.array(keep, dtype=int)
    return order[:k].astype(int)


def score_weights(order, score):
    """按打分名次线性递减加权：第 1 名权 k，第 k 名权 1。"""
    n = len(order)
    if n == 0:
        return np.zeros(0)
    raw = np.arange(n, 0, -1, dtype=np.float64)
    return raw / raw.sum()


def evaluate(rec, C, label, extra=None):
    tr, te = set(C["tr"]), set(C["te"])
    def st(vals):
        v = np.asarray(vals, dtype=np.float64)
        if len(v) < 2:
            return (0.0, 0.0, 0.0)
        return (float(v.mean() * 100),
                float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))) if v.std() > 0 else 0.0,
                float((v > 0).mean()))
    a = st([x["ex"] for x in rec if x["m"] in tr])
    b = st([x["ex"] for x in rec if x["m"] in te])
    allv = np.array([x["ex"] for x in rec], dtype=np.float64)
    d = {"label": label, "months": len(rec),
         "train_ex": round(a[0], 3), "train_t": round(a[1], 2),
         "test_ex": round(b[0], 3), "test_t": round(b[1], 2),
         "test_win": round(b[2], 3),
         "all_ex": round(float(allv.mean() * 100), 3),
         "all_t": round(st(allv)[1], 2),
         "turn_w": round(float(np.mean([x["turn_w"] for x in rec])), 4),
         "turn_list": round(float(np.mean([x["turn_list"] for x in rec])), 4),
         "max_w": round(float(np.max([x["max_w"] for x in rec])), 4),
         "max_ind_w": round(float(np.mean([x["max_ind_w"] for x in rec])), 4),
         "n_ind": round(float(np.mean([x["n_ind"] for x in rec])), 1),
         "size_expo": round(float(np.mean([x["size_expo"] for x in rec])), 4)}
    if extra:
        d.update(extra)
    return d


def run_portfolio(C, spec):
    """跑一套配置，返回逐月记录。"""
    NS, RET, AL, MKT, IND = C["NS"], C["RET"], C["AL"], C["MKT"], C["IND"]
    Z = C["Z"]
    neu = spec.get("neutral", "raw")
    rec = []
    prev_w = {}
    for m in range(NS):
        r = RET[m]
        valid = np.array(AL[m], dtype=bool) & np.isfinite(r)
        if valid.sum() < 500:
            continue
        mk = float(np.mean(r[valid]))
        mz = np.log(np.where(MKT[m] > 0, MKT[m], np.nan))
        mz = zscore_row(mz)
        s = np.zeros(len(r)); acc = np.zeros(len(r))
        for k in spec["factors"]:
            v = Z[k][m]
            ok = np.isfinite(v)
            s[ok] += spec["weights"].get(k, 1.0) * v[ok]
            acc[ok] += abs(spec["weights"].get(k, 1.0))
        with np.errstate(invalid="ignore"):
            s = np.where(acc > 0, s / np.where(acc > 0, acc, 1), np.nan)
        s = neutralize(s, mz, IND, neu, valid)

        if spec.get("quintile"):
            idx = np.flatnonzero(valid & np.isfinite(s))
            order = idx[np.argsort(s[idx], kind="stable")]
            b = np.linspace(0, len(order), Q + 1).astype(int)
            top = order[b[-2]:b[-1]]
            w = {int(c): 1.0 / max(1, len(top)) for c in top}
            new_top = set(int(c) for c in top)
            to_list = 1.0 if not prev_w else len(new_top - set(prev_w)) / max(1, len(new_top))
        else:
            per_ind = spec.get("per_ind_max", 0)
            order = pick_top(s, valid, spec.get("k", K), IND, per_ind)
            if len(order) == 0:
                continue
            if spec.get("weight") == "score":
                wv = score_weights(order, s)
            else:
                wv = np.ones(len(order)) / len(order)
            wv = cap_weights(wv, spec.get("cap", 0))
            wv = ind_weight_cap(wv, IND[order], spec.get("ind_cap", 0))
            # 换手带：上一期持仓只要在 top (1+buffer)k 里就保留，减少无谓买卖
            if spec.get("buffer", 0) > 0 and prev_w:
                band = int(spec["k"] * (1 + spec["buffer"]))
                zone = set(int(c) for c in pick_top(s, valid, band, IND, per_ind))
                keep = [c for c in prev_w if c in zone][:spec["k"]]
                if keep:
                    keep_set = set(keep)
                    rest = [int(c) for c in order if int(c) not in keep_set][:max(0, spec["k"] - len(keep))]
                    order = np.array(keep + rest, dtype=int)
                    if spec.get("weight") == "score":
                        wv = score_weights(order, s)
                    else:
                        wv = np.ones(len(order)) / len(order)
                    wv = cap_weights(wv, spec.get("cap", 0))
                    wv = ind_weight_cap(wv, IND[order], spec.get("ind_cap", 0))
            w = {int(c): float(wv[i]) for i, c in enumerate(order)}
            new_top = set(int(c) for c in order)
            to_list = 1.0 if not prev_w else len(new_top - set(prev_w)) / max(1, len(new_top))

        keys = set(w) | set(prev_w)
        turn_w = 0.5 * sum(abs(w.get(c, 0.0) - prev_w.get(c, 0.0)) for c in keys)
        gross = sum(w[c] * r[c] for c in w)
        net = gross - turn_w * FEE
        ex = net - mk
        iw = {}
        for c, ww in w.items():
            g = int(IND[c]); iw[g] = iw.get(g, 0.0) + ww
        se = 0.0
        for c, ww in w.items():
            if np.isfinite(mz[c]):
                se += ww * mz[c]
        rec.append({"m": m, "ex": float(ex), "turn_w": float(turn_w),
                    "turn_list": float(to_list), "max_w": float(max(w.values())),
                    "max_ind_w": float(max(iw.values())), "n_ind": len(iw),
                    "size_expo": float(se)})
        prev_w = w
    return rec


def mode_portfolio(C):
    C["Z"] = factor_rows(C, list(PAIR))
    base = dict(factors=list(PAIR), weights={k: 1.0 for k in PAIR}, k=K)
    specs = [
        ("A_quintile_mlfit2", dict(base, quintile=True), "复刻 mlfit2：top 分位等权（锚点）"),
        ("B_top50_equal",    dict(base, weight="equal"), "top50 等权，无约束"),
        ("C_score_cap4",     dict(base, weight="score", cap=0.04), "打分加权 + 单票上限 4%"),
        ("D_plus_ind",       dict(base, weight="score", cap=0.04, per_ind_max=8, ind_cap=0.15), "再加每行业≤8只 / 行业权重≤15%"),
        ("E_plus_band",      dict(base, weight="score", cap=0.04, per_ind_max=8, ind_cap=0.15, buffer=0.5), "再加换手带 buffer=0.5"),
        ("F_size_neutral",   dict(base, weight="score", cap=0.04, per_ind_max=8, ind_cap=0.15, neutral="size"), "打分对 ln(市值) 取残差"),
        ("G_ind_neutral",    dict(base, weight="score", cap=0.04, per_ind_max=8, ind_cap=0.15, neutral="ind"), "打分行业内去均值"),
        ("H_both_neutral",   dict(base, weight="score", cap=0.04, per_ind_max=8, ind_cap=0.15, neutral="both"), "市值 + 行业双中性"),
    ]
    print("训练 %d 月 / 测试 %d 月，持仓 %d 只，FEE=%.4f" % (len(C["tr"]), len(C["te"]), K, FEE), flush=True)
    print("%-20s %8s %7s %8s %7s %7s %8s %8s %8s" % (
        "配置", "训练%/t", "", "测试%/t", "", "胜率", "权重换手", "最大票权", "市值暴露"))
    rows = []
    for tag, spec, desc in specs:
        t0 = datetime.datetime.now()
        rec = run_portfolio(C, spec)
        if not rec:
            print(tag, "NO REC"); continue
        d = evaluate(rec, C, tag, {"desc": desc, "spec": {k: v for k, v in spec.items() if k not in ("factors", "weights")}})
        rows.append(d)
        print("%-20s %7.2f %5.2f %8.2f %6.2f %7.3f %8.3f %8.3f %8.3f  [%s]" % (
            tag, d["train_ex"], d["train_t"], d["test_ex"], d["test_t"], d["test_win"],
            d["turn_w"], d["max_w"], d["size_expo"],
            (datetime.datetime.now() - t0).total_seconds()), flush=True)
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "basis": "fscan_parts 183 月面板，size+amt20 等权 z，train<2019 / test>=2019，基准=当期池等权，FEE=0.001",
           "rows": rows}
    json.dump(out, open(os.path.join(HERE, "risk_portfolio.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved risk_portfolio.json", flush=True)


def mode_orth(C):
    """新因子对 size+amt20 正交化后，残差 IC 还剩多少。"""
    keys = [k for k in FACTORS if k not in PAIR]
    C["Z"] = factor_rows(C, list(PAIR) + keys)
    NS, RET, AL = C["NS"], C["RET"], C["AL"]
    def spearman(a, b):
        if len(a) < 100: return np.nan
        ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
        return float((ra * rb).sum() / den) if den > 0 else np.nan
    res = {}
    for k in keys:
        raw, resid, corr = [], [], []
        for m in range(NS):
            y = RET[m]
            v = np.array(AL[m], dtype=bool) & np.isfinite(y)
            x1, x2, f = C["Z"][PAIR[0]][m], C["Z"][PAIR[1]][m], C["Z"][k][m]
            v &= np.isfinite(x1) & np.isfinite(x2) & np.isfinite(f)
            idx = np.flatnonzero(v)
            if len(idx) < 300: continue
            X = np.column_stack([np.ones(len(idx)), x1[idx], x2[idx]])
            try:
                beta = np.linalg.lstsq(X, f[idx], rcond=None)[0]
            except Exception:
                continue
            e = f[idx] - X @ beta
            raw.append(spearman(f[idx], y[idx]))
            resid.append(spearman(e, y[idx]))
            ss_tot = float(((f[idx] - f[idx].mean()) ** 2).sum())
            corr.append(float(1.0 - (e @ e) / ss_tot) if ss_tot > 0 else 0.0)
        def st(v):
            a = np.array([x for x in v if np.isfinite(x)])
            if len(a) < 12: return (0.0, 0.0, 0)
            return (round(float(a.mean()), 4), round(float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))), 2), len(a))
        r0, r1 = st(raw), st(resid)
        res[k] = {"raw_ic": r0[0], "raw_t": r0[1], "n": r0[2],
                  "resid_ic": r1[0], "resid_t": r1[1],
                  "r2_on_pair": round(float(np.mean(corr)), 3) if corr else None,
                  "kept": round(r1[0] / r0[0], 3) if r0[0] else None}
        print("%-14s 原始IC %8.4f t%6.2f | 残差IC %8.4f t%6.2f | 被两因子解释 %5.1f%%" % (
            k, r0[0], r0[1], r1[0], r1[1], 100 * (np.mean(corr) if corr else 0)), flush=True)
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "basis": "逐月对 [1, size, amt20] 做 OLS，取残差算 Spearman IC；r2_on_pair 是被两因子解释的方差比例",
           "factors": res}
    json.dump(out, open(os.path.join(HERE, "factor_orth.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved factor_orth.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("portfolio", "orth"), default="portfolio")
    a = ap.parse_args()
    C = load_common()
    print("[load] %d 个唯一月（fraw 原始 %d 行，重复落盘 %d 行）、%d 个可回测区间 %s ~ %s，股票 %d，行业 %d" % (
        len(C["JJ"]), fraw.meta()["n"], fraw.meta()["n"] - len(C["JJ"]), C["NS"],
        C["dates"][int(C["JJ"][0])], C["dates"][int(C["JJ"][-1])],
        C["RET"].shape[1], C["n_ind"]), flush=True)
    (mode_portfolio if a.mode == "portfolio" else mode_orth)(C)


if __name__ == "__main__":
    main()
