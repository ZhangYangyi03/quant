# -*- coding: utf-8 -*-
"""用 5 分钟线校准出的 impact 代回 TCA，判断结论是否翻转。"""
import os,sys,json
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(k,"1")
sys.path.insert(0, r"D:\Users\china\Desktop\项目_开发\quant")
import tca
P = tca.Panel()
bench = json.load(open(r"D:\Users\china\Desktop\项目_开发\quant\bench_all_a_clean.json",encoding="utf-8"))
bk,br = tca.monthly(bench)
bm = dict(zip(bk,br))
cal = json.load(open(r"D:\Users\china\Desktop\项目_开发\quant\execution_cal.json",encoding="utf-8"))
etas = {
 "校准中位": cal["impact_conversion"]["at_p_2pct_median"],
 "校准p25": cal["impact_conversion"]["at_p_2pct_p25"],
 "校准p75": cal["impact_conversion"]["at_p_2pct_p75"],
 "保守0.02": 0.02,
 "原设0.10": 0.10,
}
out=[]
for tag,eta in etas.items():
    r = P.run(fee_on=True, impact=eta, sym_impact=True, limit_check=True,
              exec_open=True, label=tag)
    a = tca.alpha_of(r["curve"],bm)
    row={"tag":tag,"eta":round(eta,5),"cagr":r["cagr"],
         "alpha_annual":a.get("alpha_annual"),"alpha_t":a.get("alpha_t"),
         "cost_bps":r["cost_bps_avg"],"fees_by":r["fees_by"]}
    out.append(row)
    print("%-10s eta %.5f  CAGR %6.2f%%  alpha %6.2f%% t=%5.2f  bps %6.2f" % (
        tag,eta,r["cagr"],a.get("alpha_annual",0),a.get("alpha_t",0),r["cost_bps_avg"]),flush=True)
json.dump({"generated":__import__("datetime").datetime.now().isoformat(timespec="seconds"),
           "basis":"S5_exec_open 口径（t+1开盘成交+双侧冲击+涨跌停过滤+全分项费用）",
           "rows":out},
          open(r"D:\Users\china\Desktop\项目_开发\quant\tca_recalibrated.json","w",encoding="utf-8"),
          ensure_ascii=False,indent=1)
print("已写 tca_recalibrated.json")
