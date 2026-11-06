# -*- coding: utf-8 -*-
"""audit_etf.py —— 把三层审计接到 ETF 轮动上。

为什么值得单独做：个股那条 alpha 被「涨停/停牌不可买」判死（EXEC_CONSTRAINTS.md），
而 ETF 无个股一字板、免印花税、T+0，理论上能绕开 L3。
但 results_etf.json 里 lowvol60/k2 夏普 1.52 / MDD -6.3% 是没过审的数字，
必须先查它是不是前视或索引错配造的。

数据：data/*.csv（8 只 ETF，2007-01-18 ~ 2026-11-02，4808 交易日）。
成本：双边 5bp（ETF 免印花税），调仓周期 20 日（与 run_backtest.py 一致）。
"""
import os, sys, csv, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_backtest import audit, multi_year_verdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REBAL = 20


def load_panel():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".csv"))
    codes = [f[:-4] for f in files]
    per = {}
    for c, fn in zip(codes, files):
        rows = list(csv.DictReader(open(os.path.join(DATA, fn), encoding="utf-8")))
        per[c] = {r["date"]: r for r in rows}
    dates = sorted(set.intersection(*[set(v.keys()) for v in per.values()]))
    T, N = len(dates), len(codes)
    P = {k: np.full((T, N), np.nan) for k in ("close", "open_", "amount", "high", "low")}
    fld = {"close": "close", "open_": "open", "amount": "amount",
           "high": "high", "low": "low"}
    for j, c in enumerate(codes):
        for i, d in enumerate(dates):
            r = per[c].get(d)
            if not r: continue
            for k, src in fld.items():
                try: P[k][i, j] = float(r[src])
                except (ValueError, KeyError): pass
    return dates, codes, P


def signal_lowvol(P, lookback=60, topk=2):
    """低波 60 日：过去 lookback 日收益标准差最小者为优（越大越好的分数）。
    只用 t 日及以前的数据，不含未来。"""
    C = P["close"]
    T, N = C.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.diff(C, axis=0) / C[:-1]
    R = np.where(np.isfinite(R) & (np.abs(R) < 0.5), R, np.nan)
    S = np.full((T, N), np.nan)
    for t in range(lookback, T):
        w = R[t-lookback:t]
        for j in range(N):
            v = w[:, j]
            if np.isfinite(v).sum() >= lookback * 0.8:
                S[t, j] = -np.nanstd(v)     # 波动越小分数越高
    return S


def signal_momentum(P, lookback=20, topk=2):
    C = P["close"]; T, N = C.shape
    S = np.full((T, N), np.nan)
    for t in range(lookback, T):
        ok = np.isfinite(C[t]) & np.isfinite(C[t-lookback]) & (C[t-lookback] > 0)
        S[t, ok] = C[t, ok] / C[t-lookback, ok] - 1
    return S


def run():
    dates, codes, P = load_panel()
    T, N = P["close"].shape
    print("[etf] dates", T, dates[0], dates[-1], "codes", N, codes, flush=True)
    out = {}
    for name, S in (("lowvol60", signal_lowvol(P, 60)),
                    ("momentum20", signal_momentum(P, 20)),
                    ("momentum60", signal_momentum(P, 60))):
        # L3 对 ETF 放宽：ETF 没有个股一字板，但停牌/零成交仍不可买
        # 用 high==low（一字）且 |ret|>9% 作为封板代理，另加零成交
        rep = audit(S, P["close"], P["open_"], P["amount"], dates, codes,
                    topk=2, cost_bp=5.0)
        out[name] = rep
        print("[etf] %-12s verdict=%s fails=%s lag_nav=%s exec=%s rankic=%s" %
              (name, rep["verdict"], rep["fails"],
               rep["L2_alignment"]["nav_by_lag"],
               rep["executable_alpha_nav"],
               rep["L2_alignment"]["rankic_by_lag"]), flush=True)
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "remedies"}
               for k, v in out.items()},
              open(os.path.join(HERE, "audit_etf.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    return out

if __name__ == "__main__":
    run()
