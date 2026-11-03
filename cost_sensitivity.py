"""成本敏感性：同一策略在三档费率下的表现。"""
import sys
sys.path.insert(0, r"D:\Users\china\Desktop\项目_开发\quant")
import run_backtest as rb
from engine import Cost, run_backtest as _run
import perf

bars = rb.load_bars()
SCEN = {
 "优惠费率": Cost(commission_rate=0.0001, min_commission=0.0, transfer_fee=0.0, stamp_duty=0.0, etf_stamp_duty=0.0),
 "券商默认": Cost(commission_rate=0.00025, min_commission=5.0, transfer_fee=0.00001, stamp_duty=0.0, etf_stamp_duty=0.0),
 "高费率":   Cost(commission_rate=0.0005, min_commission=5.0, transfer_fee=0.00001, stamp_duty=0.0, etf_stamp_duty=0.0),
}
for name in ("momentum20", "lowvol60", "mom_vol20", "resid20_60"):
    for k in (2, 3):
        sig = rb.make_signal(bars, name, k, 60)
        parts = []
        for scen, cost in SCEN.items():
            port = _run(bars, sig, start_cash=100000.0, start_date="2016-01-01", cost=cost)
            filled = [t for t in port.trades if t.get("status") == "filled"]
            p = perf.perf(port.equity[60:])
            parts.append("%s CAGR %.2f%% SR %.2f 费 %d" % (scen, p["cagr_pct"], p["sharpe"], sum(t["fee"] for t in filled)))
        print("%s/k%d | " % (name, k) + " | ".join(parts), flush=True)
