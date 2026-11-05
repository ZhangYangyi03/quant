import os, csv, json, struct, mmap, time
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "stockdata")
meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
dates = meta["dates"]; codes = meta["codes"]
di = {d: i for i, d in enumerate(dates)}
ND = len(dates); NC = len(codes); N = ND * NC
real = os.path.join(HERE, "amt_flat.bin")
if not os.path.exists(real):
    with open(real, "wb") as f:
        f.seek(16 + 8 * N - 1); f.write(b"\0")
fp = open(real, "r+b"); mm = mmap.mmap(fp.fileno(), 0)
t0 = time.time()
for ci, code in enumerate(codes):
    off = ci * ND
    with open(os.path.join(SRC, code + ".csv"), encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for r in rd:
            if len(r) < 7: continue
            try: amt = float(r[6])
            except ValueError: continue
            j = di.get(r[0])
            if j is None or amt <= 0: continue
            struct.pack_into("<d", mm, 16 + 8 * (off + j), amt)
    if (ci + 1) % 800 == 0: print(" ", ci + 1, "%.0fs" % (time.time() - t0), flush=True)
struct.pack_into("<8sII", mm, 0, b"AMTF0001", ND, NC)
mm.flush(); mm.close(); fp.close()
print("成交额矩阵 %.0f MB %.0fs" % (os.path.getsize(real) / 1e6, time.time() - t0))
