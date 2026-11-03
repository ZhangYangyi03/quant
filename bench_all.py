import sys, os, json, time, struct
sys.path.insert(0, r"D:\Users\china\Desktop\项目_开发\quant")
t0 = time.time()
import market_engine as ME
b = ME.StockBars(r"D:\Users\china\Desktop\项目_开发\quant\store")
n = b.load_dir(); b.finalize()
print("载入 %d 只 %d 天 %.1fs" % (n, len(b.dates), time.time() - t0), flush=True)
dates = [d for d in b.dates if d >= "2010-01-01"]
di = {d: i for i, d in enumerate(dates)}
rets = [0.0] * len(dates); cnts = [0] * len(dates)
t1 = time.time()
for code, m in b.meta.items():
    nrow = m[1]; db = m[2]
    idxs = []
    for i in range(nrow):
        d = bytes(db[i * 10:(i + 1) * 10]).decode()
        idxs.append(di.get(d, -1))
    prev_close = None
    for i in range(nrow):
        r = b.row_at(code, i)
        j = idxs[i]
        if j >= 0 and prev_close is not None and prev_close > 0 and r[1] > 0:
            rets[j] += r[1] / prev_close - 1; cnts[j] += 1
        if j >= 0: prev_close = r[1]
print("逐股 %.1fs" % (time.time() - t1), flush=True)
eq = 1.0; curve = []
for i, d in enumerate(dates):
    if cnts[i]: eq *= 1 + rets[i] / cnts[i]
    curve.append([d, round(1_000_000.0 * eq, 2)])
json.dump(curve, open(r"D:\Users\china\Desktop\项目_开发\quant\bench_all_a.json", "w", encoding="utf-8"), ensure_ascii=False)
print("写出 %d 天 %.0f 倍 总 %.1fs" % (len(curve), eq, time.time() - t0), flush=True)
