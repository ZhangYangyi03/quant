# -*- coding: utf-8 -*-
"""alt_orth.py —— 用 risk_portfolio.mode_orth 的完全相同口径，检验 6 个非价量特征
对 [1, size, amt20] 正交后还剩多少 IC。

复用 risk_portfolio 的 C 构造（同一份 fscan_parts 面板、同一收益/掩码/方向表），
只把 C["Z"] 里的因子行替换成 alt_panel.json 的月度截面。
"""
import os, sys, json, math, datetime
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import risk_portfolio as rp

C = rp.load_common()
NS, RET, AL, dates, JJ = C["NS"], C["RET"], C["AL"], C["dates"], C["JJ"]
codes = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))["codes"]
ci = {c: i for i, c in enumerate(codes)}
Zpair = rp.factor_rows(C, list(rp.PAIR))
alt = json.load(open(os.path.join(HERE, "alt_data", "alt_panel.json"), encoding="utf-8"))
FEATS = alt["features"]
M = C["M"]

def zs(v):
    x = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 2: return x
    m, s = x[ok].mean(), x[ok].std()
    return np.where(ok, (x - m) / (s if s > 0 else 1.0), np.nan)

# 为每个月准备 (NC,) 的因子行；无数据 = NaN
rows = {f: np.full((NS, len(codes)), np.nan) for f in FEATS}
for m in range(NS):
    month = dates[int(JJ[m])][:7]
    cm = alt["months"].get(month)
    if not cm: continue
    for c, v in cm.items():
        k = ci.get(c)
        if k is None: continue
        for f in FEATS:
            x = v.get(f)
            if x is not None and np.isfinite(x): rows[f][m, k] = float(x)
    for f in FEATS: rows[f][m] = zs(rows[f][m])

def spearman(a, b):
    if len(a) < 100: return np.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan

def st(v):
    a = np.array([x for x in v if np.isfinite(x)])
    if len(a) < 12: return (0.0, 0.0, 0)
    return (round(float(a.mean()), 4),
            round(float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a)))), 2), len(a))

res = {}
for f in FEATS:
    raw, resid, corr, cover = [], [], [], []
    for m in range(NS):
        y = RET[m]
        v = np.array(AL[m], dtype=bool) & np.isfinite(y)
        x1, x2, ft = Zpair[rp.PAIR[0]][m], Zpair[rp.PAIR[1]][m], rows[f][m]
        v &= np.isfinite(x1) & np.isfinite(x2) & np.isfinite(ft)
        idx = np.flatnonzero(v)
        if len(idx) < 300: continue
        cover.append(len(idx))
        X = np.column_stack([np.ones(len(idx)), x1[idx], x2[idx]])
        beta = np.linalg.lstsq(X, ft[idx], rcond=None)[0]
        e = ft[idx] - X @ beta
        raw.append(spearman(ft[idx], y[idx]))
        resid.append(spearman(e, y[idx]))
        ss = float(((ft[idx] - ft[idx].mean()) ** 2).sum())
        corr.append(float(1.0 - (e @ e) / ss) if ss > 0 else 0.0)
    r0, r1 = st(raw), st(resid)
    res[f] = {"raw_ic": r0[0], "raw_t": r0[1], "n": r0[2],
              "resid_ic": r1[0], "resid_t": r1[1],
              "r2_on_pair": round(float(np.mean(corr)), 3) if corr else None,
              "kept": round(r1[0] / r0[0], 3) if r0[0] else None,
              "median_coverage": int(np.median(cover)) if cover else 0}
    print("%-18s 原始IC %7.4f t%6.2f | 残差IC %7.4f t%6.2f | 被解释 %5.1f%% | 覆盖 %d" % (
        f, r0[0], r0[1], r1[0], r1[1], 100 * (np.mean(corr) if corr else 0),
        res[f]["median_coverage"]), flush=True)

out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "basis": "与 factor_orth.json 完全同口径：逐月对 [1, size, amt20] 做 OLS，取残差算 Spearman IC",
       "note": "非价量数据来自东财免费接口；lhb/margin 滞后 1 月，holdernum/orghold 按真实公告日可见",
       "factors": res}
json.dump(out, open(os.path.join(HERE, "alt_orth.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("saved alt_orth.json")
