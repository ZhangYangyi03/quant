# -*- coding: utf-8 -*-
"""校准 eta 下的换手前沿：找 alpha_t > 2 且成本可控的组合。"""
import os, sys, json, datetime
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import tca

def path(name):
    return os.path.join(ROOT, name)

P = tca.Panel()
bench = json.load(open(path("bench_all_a_clean.json"), encoding="utf-8"))
bk, br = tca.monthly(bench)
bm = dict(zip(bk, br))
cal = json.load(open(path("execution_cal.json"), encoding="utf-8"))
eta = cal["impact_conversion"]["at_p_2pct_median"]
rows = []
for buf, hold, tag in [(0, 1, "baseline"), (0.25, 1, "band25"), (0.5, 1, "band50"),
                       (1.0, 1, "band100"), (0.5, 2, "band50_h2"),
                       (0.5, 3, "band50_h3"), (1.0, 3, "band100_h3"),
                       (1.0, 6, "band100_h6"), (1.5, 3, "band150_h3")]:
    r = P.run(fee_on=True, impact=eta, sym_impact=True, limit_check=True,
              exec_open=True, buffer=buf, hold=hold, label=tag)
    a = tca.alpha_of(r["curve"], bm)
    row = {"tag": tag, "buffer": buf, "hold_months": hold, "eta": round(eta, 5),
           "cagr": r["cagr"], "alpha_annual": a.get("alpha_annual"),
           "alpha_t": a.get("alpha_t"), "beta": a.get("beta"),
           "turnover": r["turnover_avg"], "cost_bps": r["cost_bps_avg"],
           "fees_by": r["fees_by"]}
    rows.append(row)
    print("%-13s CAGR %6.2f%%  alpha %6.2f%% t=%5.2f  换手 %.3f  bps %6.2f" % (
        tag, r["cagr"], a.get("alpha_annual", 0), a.get("alpha_t", 0),
        r["turnover_avg"], r["cost_bps_avg"]), flush=True)
json.dump({"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "eta": eta,
           "basis": "t+1开盘 + 双侧冲击 + 涨跌停过滤 + 5分钟线校准eta",
           "rows": rows},
          open(path("tca_calibrated_frontier.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("已写 tca_calibrated_frontier.json")
