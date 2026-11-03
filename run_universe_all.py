
"""全A 池回测：不再限于创业板/科创板。

与 run_universe.py 的差别只有三点，都是这一轮新增：
  1. 池子换成码表里全部 A 股（5118），含退市、含沪深主板。
  2. 基准不再用写死的 bench_equal_weight.json，而是当场按同一池子算等权，
     这样"策略跑输基准"这句话比的是同一批股票，不是拿创业板基准套全A。
  3. 支持 --turnover 选项：单次调仓换手上限，超过就把信号最弱的部分留着不动。

用法：
  python run_universe_all.py --bench          # 只算基准
  python run_universe_all.py a --turnover 0.3 # lowvol60/k50，换手上限 30%
"""
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import market_engine as ME
from market_engine import PX_OPEN, PX_CLOSE
import market_signals as MS
from array import array

STORE = os.path.join(HERE, "store")
OUT = os.path.join(HERE, "universe_all_results.json")
BENCH = os.path.join(HERE, "bench_all_a_clean.json")
LOGP = os.path.join(HERE, "universe_all.log")


def _yearly(equity):
    import perf
    return perf.yearly_table(equity)


def _ab(equity, bench):
    import perf
    return perf.alpha_beta_py(equity, bench)


def _perf(equity):
    # 为什么延迟 import：本机 WSL 常驻吃掉 6.3GB 提交内存，
    # numpy 与 3000 只股票的 mmap 不能同时存在于一个进程（实测 MemoryError）。
    # 回测进程保持纯标准库，只有算绩效时把 numpy 拉进来。
    import perf
    return perf.perf_py(equity)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOGP, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def equal_weight_bench(bars, start="2010-01-01"):
    """同一池子的等权日收益：每只票当日相对前一日收益，按当日有价的只数平均。"""
    dset = set(d for d in bars.dates if d >= start)
    s, c = {}, {}
    for code, m in bars.meta.items():
        n = m[1]
        prev = None
        for i in range(n):
            r = bars.row_i(code, i)
            if r is None:
                continue
            d = m[2][i * 10:(i + 1) * 10].decode()
            if prev is not None and d in dset and prev > 0 and r[1] > 0:
                s[d] = s.get(d, 0.0) + r[1] / prev - 1
                c[d] = c.get(d, 0) + 1
            prev = r[1]
    eq, curve = 1.0, []
    for d in sorted(dset):
        if c.get(d):
            eq *= 1 + s[d] / c[d]
        curve.append([d, 1_000_000.0 * eq])
    return curve


def run(bars, kind, n, k, rebal=20, start="2010-01-01", cash=1_000_000.0, turnover=None):
    acct = ME.Acct(cash)
    _cur = MS.Cursor(bars)
    dates = [d for d in bars.dates if d >= start]
    last_price, pending, nrebal, held = {}, None, 0, set()
    dayv = array("d")
    for i, d in enumerate(dates):
        if pending is not None:
            target = pending
            # 换手上限：先算要卖的，若卖出市值占比超过上限，就保留一部分最弱的
            if turnover is not None and acct.pos:
                mv_total = acct.cash + acct.mv(last_price)
                sell_list = sorted((c for c in acct.pos if c not in target),
                                   key=lambda c: -acct.pos[c][0] * last_price.get(c, 0))
                budget = turnover * mv_total
                kept = []
                used = 0.0
                for c in sell_list:
                    v = acct.pos[c][0] * last_price.get(c, 0)
                    if used + v <= budget:
                        used += v
                        continue
                    kept.append(c)
                target = list(target) + kept
            for c in list(acct.pos):
                if c not in target:
                    acct.sell(bars, c, d, acct.pos[c][0])
            equity = acct.cash + acct.mv(last_price)
            per = equity / max(len(target), 1)
            for c in target:
                have = acct.pos.get(c, [0, 0])[0]
                b = bars.bar(c, d)
                if not b:
                    continue
                cur = have * b[PX_OPEN]
                if cur < per * 0.9:
                    acct.buy(bars, c, d, per - cur, b=b)
                elif cur > per * 1.1 and have > 0:
                    q = int((cur - per) / b[PX_OPEN] / 100) * 100
                    if q > 0:
                        acct.sell(bars, c, d, q)
            pending = None
        if i >= n and (i - n) % rebal == 0:
            sc = MS.cross_section_fast(bars, _cur, d, kind, n)
            if len(sc) >= k:
                pending = [c for c, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]
                nrebal += 1
                if nrebal % 20 == 0: log("  调仓 %d 次，当前 %s" % (nrebal, d))
        for c in acct.pos:
            b = bars.bar(c, d)
            if b:
                last_price[c] = b[PX_CLOSE]
        if i % 1000 == 0:
            log("  进度 %d/%d 天 %s 净值 %.0f" % (i, len(dates), d, acct.cash + acct.mv(last_price)))
        dayv.append(round(acct.cash + acct.mv(last_price), 2))
    acct.equity = list(zip(dates, list(dayv)))
    return acct, nrebal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="a")
    ap.add_argument("--turnover", type=float, default=None)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--start", default="2010-01-01")
    a = ap.parse_args()

    t0 = time.time()
    bars = ME.StockBars(STORE)
    nc = bars.load_dir()
    bars.finalize()
    log("载入 %d 只，%d 个交易日 %s -> %s (%.0fs)" % (nc, len(bars.dates), bars.dates[0], bars.dates[-1], time.time() - t0))
    _neg = 0
    for _c, _m in bars.meta.items():
        for _i in range(_m[1]):
            _r = bars.row_at(_c, _i)
            if _r and _r[PX_CLOSE] <= 0: _neg += 1
    log("负价/零价行 %d 行（引擎内已按停牌处理，不计入收益）" % _neg)

    if os.path.exists(BENCH):
        curve = [tuple(x) for x in json.load(open(BENCH, encoding="utf-8"))]
        log("基准从缓存载入 %d 天" % len(curve))
    else:
        t1 = time.time()
        curve = [tuple(x) for x in equal_weight_bench(bars, a.start)]
        json.dump(curve, open(BENCH, "w", encoding="utf-8"), ensure_ascii=False)
        log("等权基准现算 %.0fs" % (time.time() - t1))
    json.dump(curve, open(os.path.join(HERE, "bench_curve_cache.json"), "w", encoding="utf-8"), ensure_ascii=False)
    log("等权基准 %d 只：%d 天曲线已写出，绩效由 perf_run.py 在独立进程里算（本进程不 import numpy）" % (nc, len(curve)))

    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    res["_bench"] = {"label": "等权全A %d 只" % nc, "note": "绩效见 perf_run.py 输出"}
    if a.bench:
        json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return

    CFG = {"a": ("lowvol", 60, 50), "b": ("mom", 60, 50), "c": ("rev", 20, 50),
           "d": ("momvol", 60, 50), "e": ("lowvol", 120, 50), "f": ("lowvol", 250, 50)}
    kind, n, k = CFG[a.which]
    t1 = time.time()
    acct, nrebal = run(bars, kind, n, k, turnover=a.turnover, start=a.start)
    label = "%s%d/k%d" % (kind, n, k) + ("/to%g" % a.turnover if a.turnover else "")
    p = _perf(acct.equity)
    p.update({"fills": acct.fills, "fees": round(acct.fees), "rebalances": nrebal,
              "blocked": dict(acct.blocked), "yearly": _yearly(acct.equity),
              "ab": _ab(acct.equity, curve)})
    res[label] = p
    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("%-20s CAGR %6.2f%% 夏普 %5.2f MDD %7.2f%% alpha %6.2f(t=%.1f) 成交 %d 费用 %d (%.0fs)" % (
        label, p["cagr_pct"], p["sharpe"], p["max_drawdown_pct"],
        p["ab"].get("alpha_annual_pct", 0), p["ab"].get("alpha_t", 0),
        acct.fills, acct.fees, time.time() - t1))


if __name__ == "__main__":
    main()
