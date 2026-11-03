"""交易诊断：换手、盈亏分布、费用占比。"""
import sys, collections
sys.path.insert(0, r"D:\Users\china\Desktop\项目_开发\quant")
import run_backtest as rb
from engine import run_backtest as _run

bars = rb.load_bars()
for name, k in (("momentum20", 3), ("lowvol60", 2)):
    sig = rb.make_signal(bars, name, k, 60)
    port = _run(bars, sig, start_cash=100000.0, start_date="2016-01-01")
    filled = [t for t in port.trades if t.get("status") == "filled"]
    pnls = []; open_pos = {}
    for t in filled:
        c = t["code"]
        if t["side"] == "buy":
            open_pos.setdefault(c, []).append(t)
        elif open_pos.get(c):
            lot = open_pos[c].pop(0)
            pnls.append((t["amount"] - lot["amount"]) - t["fee"] - lot["fee"])
    wins = [x for x in pnls if x > 0]; losses = [x for x in pnls if x <= 0]
    total_fee = sum(t["fee"] for t in filled)
    print("%s/k%d 成交 %d 笔 平仓 %d 回合" % (name, k, len(filled), len(pnls)), flush=True)
    if pnls:
        aw = sum(wins)/len(wins) if wins else 0.0
        al = sum(losses)/len(losses) if losses else 0.0
        print("   胜率 %.1f%%  平均盈利 %.1f  平均亏损 %.1f  盈亏比 %.2f" % (
            100.0*len(wins)/len(pnls), aw, al, (aw/abs(al) if al else float("inf"))), flush=True)
    print("   毛利 %.0f 元  费用 %.0f 元  净 %.0f 元" % (sum(pnls)+total_fee, total_fee, sum(pnls)), flush=True)
