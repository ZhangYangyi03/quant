import os, json, struct, mmap, time, sys, bisect
HERE = r"D:\Users\china\Desktop\项目_开发\quant"
meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
dates = meta["dates"]; codes = meta["codes"]; ND = len(dates); NC = len(codes)
sh = json.load(open(os.path.join(HERE, "shares.json"), encoding="utf-8"))
need = 16 + 8 * ND * NC
path = os.path.join(HERE, "m_mktcap.bin")
start = int(sys.argv[1]); end = int(sys.argv[2])
if start == 0 or not os.path.exists(path) or os.path.getsize(path) != need:
    with open(path, "wb") as f:
        f.seek(need - 1); f.write(b"\0")
fh_c = open(os.path.join(HERE, "m_close.bin"), "rb")
mm_c = mmap.mmap(fh_c.fileno(), 0, access=mmap.ACCESS_READ)
fp = open(path, "r+b"); mm = mmap.mmap(fp.fileno(), 0)
if start == 0:
    struct.pack_into("<8sII", mm, 0, b"MKTCP001", ND, NC)
    CH = 16 * 1024 * 1024; o = 16
    while o < need:
        n = min(CH, need - o); mm[o:o + n] = b"\0" * n; o += n
t0 = time.time()
for ci in range(start, min(end, NC)):
    c = codes[ci]
    rows = sh.get(c) or []
    if not rows: continue
    rows = sorted(rows, key=lambda r: r[0])
    ds = [r[0] for r in rows]; ns = [float(r[1]) for r in rows]
    base = ci * ND
    for j in range(ND):
        px = struct.unpack_from("<d", mm_c, 16 + 8 * (base + j))[0]
        if px <= 0: continue
        k = bisect.bisect_right(ds, dates[j]) - 1
        if k < 0: continue
        struct.pack_into("<d", mm, 16 + 8 * (base + j), px * ns[k])
mm.flush(); mm.close(); fp.close()
print("  %d-%d %.0fs" % (start, min(end, NC), time.time() - t0), flush=True)
