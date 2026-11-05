# -*- coding: utf-8 -*-
"""reconcile.py —— 两套实现的对账层。

1) 价格一致性：qlib 读回的 close/open/amount/mktcap 与 m_compact.bin 源值逐位比对
2) 曲线口径：把 qlib 日频净值转成月度收益，与本项目 strategy_layer 的缓存曲线对齐，
   报同期相关、同号率、以及各自 CAGR/夏普
3) 质量掩码：qlib 侧 corrupt/halt 日的 close 必须是 NaN
输出 reconcile_report.json
"""
import os, sys, json, math, time
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np, pandas as pd
import qlib
from qlib.config import C
from qlib.data import D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from compact import Panel
from perf import perf_py

qlib.init(provider_uri=os.path.join(HERE, "qlib_data"), region="cn",
          expression_cache=None, dataset_cache=None)
C["kernels"] = 1
C["joblib_backend"] = "threading"

meta = json.load(open(os.path.join(ROOT, "px_index_meta.json"), encoding="utf-8"))
dates, codes = meta["dates"], meta["codes"]
ci = {c: i for i, c in enumerate(codes)}
P = Panel()

report = {"price_check": {}, "curve": {}, "quality": {}}

# --- 1) 价格一致性：随机 20 只 × 5 个日期 ---
import random
random.seed(20261105)
picks = random.sample([c for c in codes if c != "__bench__"], 20)
dts = ["2012-09-03", "2015-06-15", "2018-05-15", "2020-03-10", "2024-11-05"]
maxerr, n = 0.0, 0
for c in picks:
    for d in dts:
        f = D.features([c], ["$close", "$open", "$amount", "$mktcap"],
                       start_time=d, end_time=d, freq="day")
        if f.empty:
            continue
        j = dates.index(d); k = ci[c]
        src = {"$close": float(P.col("close", j)[k]),
               "$open": float(P.col("open", j)[k]),
               "$amount": float(P.col("amt", j)[k]),
               "$mktcap": float(P.col("mktcap", j)[k])}
        for col, v in src.items():
            q = float(f[col].iloc[0])
            if np.isfinite(v) and np.isfinite(q):
                maxerr = max(maxerr, abs(v - q)); n += 1
report["price_check"] = {"n_pairs": n, "max_abs_err": maxerr,
                         "ok": bool(maxerr == 0.0 and n > 0)}

# --- 2) 曲线对账 ---
q = json.load(open(os.path.join(HERE, "qlib_curve.json"), encoding="utf-8"))
qc = [(d, v) for d, v in q["curve"]]
def to_monthly(curve):
    last = {}
    for d, v in curve:
        last[d[:7]] = float(v)
    ks = sorted(last); v = np.array([last[k] for k in ks])
    return ks[1:], v[1:] / v[:-1] - 1.0
qk, qr = to_monthly(qc)

# 本项目主策略缓存
cache_dir = os.path.join(ROOT, "_config_cache")
own = {}
for name in ("size_amt20_top50", "ep_top50", "random_k_equal"):
    f = os.path.join(cache_dir, name + ".json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f, encoding="utf-8"))
    ok, orr = to_monthly([tuple(x) for x in d["curve"]])
    mm = dict(zip(ok, orr))
    common = [k for k in qk if k in mm]
    a = np.array([dict(zip(qk, qr))[k] for k in common])
    b = np.array([mm[k] for k in common])
    corr = float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 else None
    same = float(np.mean((a > 0) == (b > 0))) if len(common) else None
    own[name] = {"n_common_months": len(common), "corr": round(corr, 4) if corr else None,
                 "same_sign": round(same, 4) if same else None,
                 "own_cagr_pct": d["stats"].get("cagr_pct"),
                 "own_sharpe": d["stats"].get("sharpe")}
report["curve"] = {
    "qlib_days": len(qc), "qlib_final_nav": round(qc[-1][1], 4),
    "qlib_test": q["test"], "qlib_pool": len(q["pool"]),
    "qlib_monthly": len(qr),
    "vs_ours": own,
    "note": "qlib 侧是 300 只池、日频 TopkDropout、LGB 信号；本项目是 5251 只池、月度调仓、手写因子。策略不同，只验数据层与量级，不要求收益一致。",
}

# --- 3) 质量掩码：corrupt 日 close 应为 NaN ---
corrupt = np.memmap(os.path.join(ROOT, "corrupt_mask.bin"), dtype=np.uint8, mode="r",
                    shape=(len(dates), len(codes)))
bad_ok = 0; bad_tot = 0
for c in picks[:5]:
    k = ci[c]
    js = np.flatnonzero(corrupt[:, k])[:3]
    for j in js:
        d = dates[j]
        f = D.features([c], ["$close"], start_time=d, end_time=d, freq="day")
        bad_tot += 1
        if f.empty or not np.isfinite(float(f["$close"].iloc[0])):
            bad_ok += 1
report["quality"] = {"corrupt_days_checked": bad_tot,
                     "close_is_nan": bad_ok,
                     "ok": bool(bad_ok == bad_tot and bad_tot > 0)}

json.dump(report, open(os.path.join(HERE, "reconcile_report.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
