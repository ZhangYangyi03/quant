# -*- coding: utf-8 -*-
"""etf_capacity.py —— 容量约束下 lowvol60 还成不成立。

用 audit(capital=..., max_participation=0.10) 把「本金等权下单、
单只金额超过当日成交额 10% 即不可成交」加进 L3，重跑全样本与逐年。
"""
import os, sys, json
import numpy as np
Q = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, Q)
from audit_etf import load_panel, signal_lowvol
from audit_backtest import audit, multi_year_verdict

DATES, CODES, P = load_panel()
S = signal_lowvol(P, 60)
T, N = P["close"].shape
IDX = [(str(d), c) for d in DATES for c in CODES]

def run(capital):
    r = audit(S, P["close"], P["open_"], P["amount"], DATES, CODES,
              topk=2, cost_bp=5.0, signal_index=IDX,
              capital=capital, max_participation=0.10)
    c = r["L3_untradable"]["capacity"]
    return r["executable_alpha_nav"], c["o2o_nav_with_capacity"], c["capacity_drop"], c["cap_blocked_cells"]

print("[capacity] baseline (no cap) 2.6699 是 exec nav（年化复合 10 年）", flush=True)
out = {}
for cap in (1e6, 1e7, 1e8, 1e9):
    a, b, d, cells = run(cap)
    print("[capacity] capital=%10.0f  exec=%.4f  with_cap=%.4f  drop=%.4f  blocked_cells=%d" %
          (cap, a, b, d, cells), flush=True)
    out[str(int(cap))] = {"exec": a, "with_cap": b, "drop": d, "blocked_cells": cells}

# 逐年（1000 万本金）
reps = []
for Y in range(2017, 2027):
    idx = [i for i, d in enumerate(DATES) if d[:4] == str(Y)]
    if len(idx) < 100: continue
    a, b = idx[0], idx[-1]+1
    sub = [(str(d), c) for d in DATES[a:b] for c in CODES]
    r = audit(S[a:b], P["close"][a:b], P["open_"][a:b], P["amount"][a:b],
              DATES[a:b], CODES, topk=2, cost_bp=5.0, signal_index=sub,
              capital=1e7, max_participation=0.10)
    # exec 用容量后的净值更诚实
    rr = dict(r)
    rr2 = {"executable_alpha_nav": r["L3_untradable"]["capacity"]["o2o_nav_with_capacity"],
           "fails": r["fails"]}
    reps.append((Y, rr2))
    print("[capacity] year %d exec_cap=%.4f fails=%s" %
          (Y, rr2["executable_alpha_nav"], rr2["fails"]), flush=True)
v = multi_year_verdict(reps)
print("[capacity] yearly verdict", json.dumps(v, ensure_ascii=False)[:600], flush=True)
out["yearly_1e7"] = v
json.dump(out, open(os.path.join(Q, "etf_capacity.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
