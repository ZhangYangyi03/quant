# -*- coding: utf-8 -*-
"""etf_holdout.py —— 严格时间外样本检验。

设计期 2016-08-08 ~ 2023-12-31，检验期 2024-01-01 ~ 2026-11-02。
检验期不参与任何参数选择（lowvol 60 日 / top2 / 20 日调仓 / 5bp 全部来自设计期的
既有设定，且这些设定在 2024 之前就已写进 run_backtest.py）。
输出两段各自的净值、夏普、三层审计、跨年度判定。
"""
import os, sys, json
import numpy as np
Q = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, Q)
from audit_etf import load_panel, signal_lowvol
from audit_backtest import audit, multi_year_verdict

DATES, CODES, P = load_panel()
T, N = P["close"].shape
S = signal_lowvol(P, 60)
C, O, A = P["close"], P["open_"], P["amount"]
ROO = np.full((T-1, N), np.nan)
ok = np.isfinite(O[:-1]) & np.isfinite(O[1:]) & (O[:-1] > 0)
ROO[ok] = O[1:][ok] / O[:-1][ok] - 1
GAP = np.full((T, N), np.nan)
ok = np.isfinite(C[:-1]) & np.isfinite(O[1:]) & (C[:-1] > 0)
GAP[1:][ok] = O[1:][ok] / C[:-1][ok] - 1
GAP[(~np.isfinite(A)) | (A <= 0)] = np.nan
s = np.where(np.isfinite(S), S, -1e9)


def nav(lo, hi, rebal=20, cost_bp=5.0):
    eq = [1.0]; cur = None; prev = None
    for k in range(max(1, lo), min(T-2, hi)):
        if cur is None or (k-1) % rebal == 0:
            sc = s[k-1].copy()
            bad = ~np.isfinite(GAP[k+1]) | (GAP[k+1] > 0.095)
            sc[bad] = -1e9
            cur = np.argsort(sc)[-2:]
        r = np.nanmean(ROO[k, cur]); r = 0 if not np.isfinite(r) else r
        if prev is not None and not np.array_equal(cur, prev):
            r -= cost_bp/1e4 * len(set(cur) ^ set(prev))/2
        prev = cur
        eq.append(eq[-1]*(1+r))
    x = np.diff(eq)/np.array(eq[:-1])
    return float(eq[-1]), float(x.mean()/x.std(ddof=1)*np.sqrt(244)), len(x)


def audit_seg(lo, hi):
    sl = slice(lo, hi)
    sub = [(str(d), c) for d in DATES[lo:hi] for c in CODES]
    return audit(S[sl], C[sl], O[sl], A[sl], DATES[lo:hi], CODES,
                 topk=2, cost_bp=5.0, signal_index=sub)


i_split = next(i for i, d in enumerate(DATES) if d >= "2024-01-01")
print("[holdout] design", DATES[0], "~", DATES[i_split-1], "days", i_split)
print("[holdout] test  ", DATES[i_split], "~", DATES[-1], "days", T-i_split, flush=True)

for label, lo, hi in (("DESIGN", 1, i_split), ("HOLDOUT", i_split, T-1)):
    n, sh, nd = nav(lo, hi)
    a = audit_seg(lo, hi)
    print("[holdout] %-8s nav %.4f  sharpe %.3f  days %d  audit=%s fails=%s exec=%.4f lag=%s" %
          (label, n, sh, nd, a["verdict"], a["fails"], a["executable_alpha_nav"],
           a["L2_alignment"]["nav_by_lag"]), flush=True)

# holdout 逐年
reps = []
for Y in (2024, 2025, 2026):
    idx = [i for i, d in enumerate(DATES) if d[:4] == str(Y)]
    if len(idx) < 100: continue
    a, b = idx[0], idx[-1]+1
    r = audit_seg(a, b)
    n, sh, nd = nav(a, b)
    reps.append((Y, r))
    print("[holdout] year %d nav %.4f sharpe %.3f audit=%s exec=%.4f" %
          (Y, n, sh, r["verdict"], r["executable_alpha_nav"]), flush=True)
v = multi_year_verdict(reps)
print("[holdout] verdict", json.dumps(v, ensure_ascii=False)[:700], flush=True)
json.dump(v, open(os.path.join(Q, "etf_holdout.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
