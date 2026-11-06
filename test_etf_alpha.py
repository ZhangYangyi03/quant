# -*- coding: utf-8 -*-
"""test_etf_alpha.py —— ETF 低波轮动结论的回归测试。

保证 ETF_ALPHA.md 里的数字不是一次性跑出来的：
  A 全样本三层审计通过、无前视（lag1 >= lag0）
  B 跨年度 10/10 正、机器判定 pass
  C 成本 50bp 下净值仍 > 2.0
  D 单日截 ±1% 后净值仍 > 2.0（不靠极端日）
  E 留一法至少 6/8 通过
  F 严格时间外样本 3/3 通过且夏普不低于设计期
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_etf import load_panel, signal_lowvol
from audit_backtest import audit, multi_year_verdict

Q = os.path.dirname(os.path.abspath(__file__))
DATES, CODES, P = load_panel()
S = signal_lowvol(P, 60)
T, N = P["close"].shape


def _nav(rebal=20, cost_bp=5.0, winsor=None):
    C, O, A = P["close"], P["open_"], P["amount"]
    ROO = np.full((T-1, N), np.nan)
    ok = np.isfinite(O[:-1]) & np.isfinite(O[1:]) & (O[:-1] > 0)
    ROO[ok] = O[1:][ok] / O[:-1][ok] - 1
    GAP = np.full((T, N), np.nan)
    ok = np.isfinite(C[:-1]) & np.isfinite(O[1:]) & (C[:-1] > 0)
    GAP[1:][ok] = O[1:][ok] / C[:-1][ok] - 1
    GAP[(~np.isfinite(A)) | (A <= 0)] = np.nan
    s = np.where(np.isfinite(S), S, -1e9)
    eq = [1.0]; cur = None; prev = None
    for k in range(1, T-1):
        if cur is None or (k-1) % rebal == 0:
            sc = s[k-1].copy()
            bad = ~np.isfinite(GAP[k+1]) | (GAP[k+1] > 0.095)
            sc[bad] = -1e9
            cur = np.argsort(sc)[-2:]
        r = np.nanmean(ROO[k, cur])
        r = 0 if not np.isfinite(r) else r
        if winsor is not None:
            r = float(np.clip(r, -winsor, winsor))
        if prev is not None and not np.array_equal(cur, prev):
            r -= cost_bp/1e4 * len(set(cur) ^ set(prev))/2
        prev = cur
        eq.append(eq[-1]*(1+r))
    return float(eq[-1])


def test_A_full_audit():
    r = audit(S, P["close"], P["open_"], P["amount"], DATES, CODES,
              topk=2, cost_bp=5.0,
              signal_index=[(str(d), c) for d in DATES for c in CODES])
    assert r["verdict"].startswith("三层审计通过"), r["verdict"]
    lag = r["L2_alignment"]["nav_by_lag"]
    assert lag[1] >= lag[0], "lag1 < lag0 说明有前视: %s" % lag
    assert not r["L2_alignment"]["ic_collapse"]
    assert r["L2_alignment"]["rankic_by_lag"][1] is not None, "小池子 RankIC 应可用"
    print("[A] full audit ->", r["verdict"], "lag", lag, "exec", r["executable_alpha_nav"])


def test_B_yearly():
    v = json.load(open(os.path.join(Q, "etf_verdict.json"), encoding="utf-8"))
    assert v["positive_years"] == v["total_years"] == 10, v
    assert v["all_three_layers_pass"] and v["pass"]
    assert v["compound_excluding_best"] > 2.0
    print("[B] yearly ->", v["verdict"], "ex-best", v["compound_excluding_best"])


def test_C_cost():
    nav = _nav(rebal=20, cost_bp=50.0)
    assert nav > 2.0, nav
    print("[C] cost 50bp ->", round(nav, 4))


def test_D_tail():
    nav = _nav(rebal=20, cost_bp=5.0, winsor=0.01)
    assert nav > 2.0, nav
    print("[D] winsor 1% ->", round(nav, 4))


def test_F_holdout():
    v = json.load(open(os.path.join(Q, "etf_holdout.json"), encoding="utf-8"))
    assert v["positive_years"] == v["total_years"] == 3, v
    assert v["all_three_layers_pass"] and v["pass"]
    assert v["compound_excluding_best"] > 1.10
    h = json.load(open(os.path.join(Q, "etf_alpha.json"), encoding="utf-8"))["time_holdout"]
    assert h["test_sharpe"] > h["design_sharpe"], "样本外夏普应不低于设计期"
    print("[F] holdout ->", v["verdict"], "test_sharpe", h["test_sharpe"],
          ">", "design", h["design_sharpe"])

def test_E_leave_one_out():
    passed = 0
    for j in range(N):
        keep = [x for x in range(N) if x != j]
        sub = [CODES[x] for x in keep]
        r = audit(signal_lowvol({k: P[k][:, keep] for k in ("close", "open_", "amount")}, 60),
                  P["close"][:, keep], P["open_"][:, keep], P["amount"][:, keep],
                  DATES, sub, topk=2, cost_bp=5.0)
        passed += int(not r["fails"])
    assert passed >= 6, passed
    print("[E] leave-one-out ->", passed, "/", N)


if __name__ == "__main__":
    for fn in (test_A_full_audit, test_B_yearly, test_C_cost, test_D_tail,
               test_E_leave_one_out, test_F_holdout):
        fn()
    print("ALL ETF TESTS PASSED")
