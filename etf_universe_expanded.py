# -*- coding: utf-8 -*-
"""etf_universe_expanded.py —— 扩池检验：29 只 ETF、1000 天、独立数据源。

数据源：新浪日线（不复权，2022-09-16 ~ 2026-11-05），与 data/*.csv（东财前复权）
口径不同，因此不混用；本脚本只用 data_extra/ 内部自洽的 29 只。
amount 为 close*volume 近似（新浪不提供成交额），已在文档标注。

检验：
  A 全池 29 只跑三层审计
  B 按资产类别分组，看结论是否只在「恰好有国债+黄金」的池子里成立
  C 严格时间 holdout：2022-09~2024-12 设计 / 2025-01~2026-11 检验
"""
import os, sys, json, csv
import numpy as np
Q = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, Q)
from audit_backtest import audit, multi_year_verdict

D = os.path.join(Q, "data_extra")
GROUPS = {
    "债券/货币":   ["511380.SH","511220.SH","511260.SH","511090.SH","511990.SH","159649.SZ","159972.SZ"],
    "海外权益":   ["513500.SH","159941.SZ","159920.SZ","510900.SH","513050.SH"],
    "商品":       ["159985.SZ","159980.SZ","159981.SZ","159934.SZ"],
    "REITs":      ["508000.SH","180101.SZ","508056.SH"],
    "A股权益":    ["512890.SH","515180.SH","512100.SH","588000.SH","512690.SH","515790.SH","512480.SH","159915.SZ","510300.SH"],
    "跨类别不含国债黄金": ["511380.SH","513500.SH","159985.SZ","508000.SH","512890.SH","588000.SH","510300.SH","159920.SZ"],
}
ALL = sorted({c for g in GROUPS.values() for c in g} | {"511010.SH"})


def load(codes):
    series = {}
    for c in codes:
        p = os.path.join(D, c + ".csv")
        if not os.path.exists(p): return None, None, None
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        series[c] = {r["date"]: tuple(float(r[k]) for k in ("open","close","high","low","volume","amount")) for r in rows}
    dates = sorted(set.intersection(*[set(v) for v in series.values()]))
    if len(dates) < 100: return dates, codes, None
    n, t = len(codes), len(dates)
    P = {k: np.full((t, n), np.nan) for k in ("close","open_","amount")}
    for j, c in enumerate(codes):
        for i, d in enumerate(dates):
            o, cl, h, l, v, a = series[c][d]
            P["open_"][i,j] = o; P["close"][i,j] = cl; P["amount"][i,j] = a
    return dates, codes, P


def lowvol(P, win=60):
    C = P["close"]; T, N = C.shape
    r = np.full_like(C, np.nan); ok = np.isfinite(C[:-1]) & (C[:-1] > 0)
    r[1:][ok] = C[1:][ok] / C[:-1][ok] - 1
    S = np.full((T, N), np.nan)
    for t in range(win, T):
        x = r[t-win:t]; m = np.isfinite(x).sum(axis=0) >= 40
        v = np.nanstd(x, axis=0)
        S[t, m & np.isfinite(v)] = -v[m & np.isfinite(v)]
    return S


def audit_on(dates, codes, P, topk=2, cost=5.0):
    S = lowvol(P)
    sub = [(str(d), c) for d in dates for c in codes]
    return audit(S, P["close"], P["open_"], P["amount"], dates, codes,
                 topk=min(topk, len(codes)), cost_bp=cost, signal_index=sub)


def main():
    print("[expanded] universe", len(ALL), "codes", flush=True)
    out = {}
    for label, codes in [("全池29只", ALL)] + list(GROUPS.items()):
        dates, cc, P = load(codes)
        if P is None:
            print("[expanded] %-20s skip (no common dates)" % label, flush=True); continue
        r = audit_on(dates, cc, P)
        print("[expanded] %-20s n=%2d days=%4d %s fails=%s exec=%.4f lag=%s" %
              (label, len(cc), len(dates), r["verdict"], r["fails"],
               r["executable_alpha_nav"], r["L2_alignment"]["nav_by_lag"]), flush=True)
        out[label] = {"n": len(cc), "days": len(dates), "verdict": r["verdict"],
                      "fails": r["fails"], "exec": r["executable_alpha_nav"],
                      "lag": r["L2_alignment"]["nav_by_lag"]}

    # 时间 holdout（全池）
    dates, cc, P = load(ALL)
    i = next(i for i, d in enumerate(dates) if d >= "2025-01-01")
    print("[expanded] holdout split", dates[i], "design", i, "test", len(dates)-i, flush=True)
    reps = []
    for label, lo, hi in (("design", 1, i), ("test", i, len(P["close"])-1)):
        sub = [(str(d), c) for d in dates[lo:hi] for c in cc]
        r = audit(lowvol(P)[lo:hi], P["close"][lo:hi], P["open_"][lo:hi],
                  P["amount"][lo:hi], dates[lo:hi], cc, topk=2, cost_bp=5.0,
                  signal_index=sub)
        print("[expanded] %-8s %s exec=%.4f lag=%s" %
              (label, r["verdict"], r["executable_alpha_nav"],
               r["L2_alignment"]["nav_by_lag"]), flush=True)
        out["holdout_"+label] = {"verdict": r["verdict"], "exec": r["executable_alpha_nav"]}
    for Y in (2025, 2026):
        idx = [k for k, d in enumerate(dates) if d[:4] == str(Y)]
        if len(idx) < 100: continue
        a, b = idx[0], idx[-1]+1
        sub = [(str(d), c) for d in dates[a:b] for c in cc]
        r = audit(lowvol(P)[a:b], P["close"][a:b], P["open_"][a:b],
                  P["amount"][a:b], dates[a:b], cc, topk=2, cost_bp=5.0,
                  signal_index=sub)
        reps.append((Y, r))
        print("[expanded] year %d %s exec=%.4f" % (Y, r["verdict"], r["executable_alpha_nav"]), flush=True)
    v = multi_year_verdict(reps)
    print("[expanded] holdout verdict", json.dumps(v, ensure_ascii=False)[:500], flush=True)
    out["holdout_verdict"] = v
    json.dump(out, open(os.path.join(Q, "etf_expanded.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
