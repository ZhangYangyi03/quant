"""全市场回测驱动。1897 只创业板/科创板，2009-2026，含退市与停牌。

设计约束：进程内存 ~490MB 上限，所以走 mmap 列式存储 + 行游标，
不把任何一整个矩阵装进内存。
"""
import sys, os, json, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import market_engine as ME
from market_engine import PX_OPEN, PX_CLOSE, PX_VOL
import market_signals as MS
import perf
from array import array

STORE = os.path.join(HERE, "store")
OUT = os.path.join(HERE, "universe_results.json")
LOGP = os.path.join(HERE, "universe_backtest.log")


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def run(bars, kind, n, k, rebal=20, start="2010-01-01", cash=1_000_000.0, verbose=False):
    """t 日收盘算信号 -> t+1 日开盘成交。买卖都用开盘价（与 ETF 版 engine.py 同一规则）。"""
    acct = ME.Acct(cash)
    dates = [d for d in bars.dates if d >= start]
    last_price = {}
    pending = None
    nrebal = 0
    dayv = array("d")
    for i, d in enumerate(dates):
        if pending is not None:
            target = pending
            # 先用成交前的价格估一次总资产，再卖，再按卖出后的总资产等分
            # （2010-12 由此产生过"per=419 元"的塌缩：卖出前算 per，卖出后现金不够，最后一轮塌到 1 元）
            for c in list(acct.pos):
                if c not in target:
                    acct.sell(bars, c, d, acct.pos[c][0])
            equity = acct.cash + acct.mv(last_price)
            per = equity / max(len(target), 1)
            for c in target:
                have = acct.pos.get(c, [0, 0])[0]
                b = bars.bar(c, d)
                if not b: continue
                cur = have * b[PX_OPEN]
                if cur < per * 0.9:
                    acct.buy(bars, c, d, per - cur, b=b)
                elif cur > per * 1.1 and have > 0:
                    q = int((cur - per) / b[PX_OPEN] / 100) * 100
                    if q > 0: acct.sell(bars, c, d, q)
            pending = None
        if (i % rebal) == 0 and i >= n and (i - n) % rebal == 0:
            sc = MS.cross_section(bars, d, kind, n)
            if len(sc) >= k:
                ranked = sorted(sc.items(), key=lambda x: -x[1])[:k]
                pending = [c for c, _ in ranked]
                nrebal += 1
        for c in acct.pos:
            b = bars.bar(c, d)
            if b: last_price[c] = b[PX_CLOSE]
            # 没价（停牌/退市）就保留最后价，不删仓、不归零
        dayv.append(round(acct.cash + acct.mv(last_price), 2))
        if verbose and i % 500 == 0:
            log("     d%d %s eq=%.0f pos=%d blocked=%r" % (i, d, dayv[-1], len(acct.pos), acct.blocked))
    acct.equity = list(zip(dates, list(dayv)))
    return acct, nrebal


def main():
    t0 = time.time()
    bars = ME.StockBars(STORE)
    nc = bars.load_dir()
    bars.finalize()
    log("载入 %d 只，%d 个交易日 %s -> %s (%.0fs)" % (nc, len(bars.dates), bars.dates[0], bars.dates[-1], time.time() - t0))

    # 等权全市场基准：复用 run_bench.py 的结果（按股票顺序扫一遍，O(总行数)）
    curve = [tuple(x) for x in json.load(open(os.path.join(HERE, "bench_equal_weight.json"), encoding="utf-8"))]
    dates = [d for d, _ in curve]
    bench = perf.perf_py(curve)
    results = {"_bench": {"label": "等权全创业板+科创板 %d 只" % len(bars.meta), **bench, "yearly": perf.yearly_table(curve)}}
    log("等权基准: CAGR %.2f%% 夏普 %.2f MDD %.2f%% 年化波动 %.2f%%" % (
        bench["cagr_pct"], bench["sharpe"], bench["max_drawdown_pct"], bench["ann_vol_pct"]))
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    import sys as _s
    which = _s.argv[1] if len(_s.argv) > 1 else "a"
    CFG = {"a": [("lowvol", 60, 50)], "b": [("mom", 60, 50)], "c": [("rev", 20, 50)],
           "d": [("momvol", 60, 50)], "e": [("lowvol", 120, 50)]}[which]
    for kind, nn, k in CFG:
        t1 = time.time()
        acct, nrebal = run(bars, kind, nn, k)
        label = "%s%d/k%d" % (kind, nn, k)
        p = perf.perf_py(acct.equity)
        p.update({"fills": acct.fills, "fees": round(acct.fees), "rebalances": nrebal,
                  "blocked": dict(acct.blocked), "yearly": perf.yearly_table(acct.equity),
                  "ab": perf.alpha_beta_py(acct.equity, curve)})
        results[label] = p
        log("%-16s CAGR %6.2f%% 夏普 %5.2f MDD %7.2f%% alpha %6.2f(t=%.1f) 成交 %d 费用 %d (%.0fs)" % (
            label, p["cagr_pct"], p["sharpe"], p["max_drawdown_pct"],
            p["ab"].get("alpha_annual_pct", 0), p["ab"].get("alpha_t", 0),
            acct.fills, acct.fees, time.time() - t1))
        json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 多重检验校正（Deflated Sharpe）
    srs = [p["sharpe"] for k2, p in results.items() if not k2.startswith("_") and "sharpe" in p]
    if srs:
        mu = sum(srs) / len(srs)
        var_sr = sum((v - mu) ** 2 for v in srs) / max(len(srs) - 1, 1)
        for label, p in results.items():
            if label.startswith("_") or "sharpe" not in p: continue
            p["deflated"] = perf.deflated_sharpe_py(p["sharpe"], len(srs), p["days"] or 1,
                                                    skew=p.get("skew", 0), kurt=(p.get("excess_kurtosis") or 0) + 3,
                                                    sr_variance_across_trials=max(var_sr, 1e-6))
            d = p["deflated"]
            log("   %-16s DeflZ %6.2f  存活 %s" % (label, d.get("deflated_z", 0), d.get("survives_selection_bias")))
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("完成 %.0f 秒" % (time.time() - t0))


if __name__ == "__main__":
    main()
