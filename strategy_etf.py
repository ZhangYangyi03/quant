"""ETF 动量轮动：每 20 个交易日按 20 日涨幅重排，持有最强且涨幅>0 的两只；全为负则持国债ETF。
交易成本按双边 5bp（ETF 免印花税）。"""
import sqlite3, statistics
UNIVERSE = ["510300.SH","510500.SH","159915.SZ","512880.SH","518880.SH"]
DEFENSIVE = "511010.SH"
LOOKBACK = 20
TOP_K = 2
COST_BPS = 5
REBAL = 20

def _series(con, codes):
    rows = con.execute("select trade_date,ts_code,close from bars where ts_code in (%s) and close>0 order by trade_date"
                       % ",".join("?"*len(codes)), codes).fetchall()
    dates = sorted({r[0] for r in rows}); idx = {}
    for d,t,c in rows: idx.setdefault(d,{})[t]=c
    return [d for d in dates if all(c in idx[d] for c in codes)], idx

def live_signal(con):
    """输出当下应当持有的标的，以及排序明细。"""
    dates, idx = _series(con, UNIVERSE + [DEFENSIVE])
    d = dates[-1]
    sc = sorted(((t, idx[d][t]/idx[dates[-1-LOOKBACK]][t]-1) for t in UNIVERSE), key=lambda x:-x[1])
    hold = [t for t,v in sc[:TOP_K] if v > 0] or [DEFENSIVE]
    return {"as_of": d, "hold": hold, "rank": [(t, round(v*100,2)) for t,v in sc],
            "reason": "全部动量为负，转防御" if hold==[DEFENSIVE] else "持有动量最强且为正的标的"}

def backtest(con):
    dates, idx = _series(con, UNIVERSE + [DEFENSIVE])
    daily=[]; prev=set(); log=[]
    for i in range(LOOKBACK, len(dates)-1):
        d, nd = dates[i], dates[i+1]
        to=0.0
        if (i-LOOKBACK) % REBAL == 0:
            sc = sorted(((t, idx[d][t]/idx[dates[i-LOOKBACK]][t]-1) for t in UNIVERSE), key=lambda x:-x[1])
            held = {t for t,v in sc[:TOP_K] if v>0} or {DEFENSIVE}
            to = len(held ^ prev)/max(len(held),1)
            prev = held
            log.append({"date": d, "hold": sorted(held)})
        if not prev: continue
        r = sum(idx[nd][t]/idx[d][t]-1 for t in prev)/len(prev)
        daily.append(r - to*COST_BPS/10000.0)
    n=len(daily); sd=statistics.pstdev(daily) or 1e-12
    e=1.0; peak=1.0; mdd=0.0
    for r in daily:
        e*=(1+r); peak=max(peak,e); mdd=min(mdd,e/peak-1)
    return {"start":dates[LOOKBACK],"end":dates[-1],"days":n,
            "total_return_pct":round((e-1)*100,2),"annual_return_pct":round((e**(250/n)-1)*100,2),
            "annual_sharpe":round(statistics.mean(daily)/sd*250**0.5,2),
            "max_drawdown_pct":round(mdd*100,2),"rebalances":len(log),"last":log[-3:]}
