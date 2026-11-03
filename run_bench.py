import sys, os, time, json, bisect
sys.path.insert(0, 'D:\\Users\\china\\Desktop\\项目_开发\\quant')
import market_engine as ME
import perf

t0 = time.time()
b = ME.StockBars('D:\\Users\\china\\Desktop\\项目_开发\\quant\\store'); nc = b.load_dir(); b.finalize()
print("载入 %d 只 %.1fs" % (nc, time.time()-t0), flush=True)
dates = [d for d in b.dates if d >= "2010-01-01"]
dset = set(dates)
tsum = {d: 0.0 for d in dates}
tcnt = {d: 0 for d in dates}
t1 = time.time()
for code in b.meta:
    n = b.nrows(code)
    prev = None
    for i in range(n):
        r = b.row_i(code, i)
        d = b.meta[code][2][i*10:(i+1)*10].decode()
        if prev is not None and d in dset and prev[1] > 0 and r[4] > 0:
            tsum[d] += r[1]/prev[1] - 1
            tcnt[d] += 1
        prev = r
print("逐股扫描 %.1fs" % (time.time()-t1), flush=True)
eq = 1.0; curve = []
for d in dates:
    if tcnt[d]: eq *= (1 + tsum[d]/tcnt[d])
    curve.append((d, 1_000_000.0*eq))
p = perf.perf_py(curve)
print(json.dumps({k: p[k] for k in ("cagr_pct","sharpe","max_drawdown_pct","ann_vol_pct","days","start","end")}, ensure_ascii=False), flush=True)
json.dump(curve, open(os.path.join('D:\\Users\\china\\Desktop\\项目_开发\\quant', "bench_equal_weight.json"), "w"), ensure_ascii=False)
print("完成 %.0fs" % (time.time()-t0))
