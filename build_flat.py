import os, csv, json, struct, mmap, time, sys
HERE = r"D:\Users\china\Desktop\项目_开发\quant"
SRC = os.path.join(HERE, "stockdata")

def codes_all():
    return sorted(f[:-4] for f in os.listdir(SRC) if f.endswith(".csv") and not f.startswith("_"))

def phase_dates():
    ds = set()
    for c in codes_all():
        with open(os.path.join(SRC, c + ".csv"), encoding="utf-8") as f:
            rd = csv.reader(f); next(rd, None)
            for r in rd:
                if r: ds.add(r[0])
    dates = sorted(ds)
    json.dump({"dates": dates, "n": len(codes_all())}, open(os.path.join(HERE, "_dates.json"), "w"))
    return dates

def phase_fill(col, path, start, end, dates, codes):
    ND = len(dates); NC = len(codes); di = {d: i for i, d in enumerate(dates)}
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.seek(16 + 8 * ND * NC - 1); f.write(b"\0")
    if start == 0:
        fp = open(path, "r+b"); mm = mmap.mmap(fp.fileno(), 0)
        magic = b"PXFL0002" if col == 2 else b"AMTF0001"
        head = b"\0" * (16 + 8 * ND * NC)
        struct.pack_into("<8sII", mm, 0, magic, ND, NC)
        # 不用整块清零（会 MemoryError），分片清零
        CH = 16 * 1024 * 1024
        total = 16 + 8 * ND * NC
        o = 16
        while o < total:
            n = min(CH, total - o)
            mm[o:o + n] = b"\0" * n
            o += n
        mm.flush(); mm.close(); fp.close()
    fp = open(path, "r+b"); mm = mmap.mmap(fp.fileno(), 0)
    t0 = time.time()
    for ci in range(start, min(end, NC)):
        c = codes[ci]; off = ci * ND
        with open(os.path.join(SRC, c + ".csv"), encoding="utf-8") as f:
            rd = csv.reader(f); next(rd, None)
            for r in rd:
                if len(r) < 7: continue
                j = di.get(r[0])
                if j is None: continue
                try: v = float(r[col])
                except ValueError: continue
                if v > 0: struct.pack_into("<d", mm, 16 + 8 * (off + j), v)
    mm.flush(); mm.close(); fp.close()
    print("  col%d %d-%d 完成 %.0fs" % (col, start, min(end, NC), time.time() - t0), flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "dates":
        d = phase_dates(); print("交易日 %d 股票 %d" % (len(d), json.load(open(os.path.join(HERE, "_dates.json")))["n"]))
    else:
        col = int(sys.argv[2]); path = os.path.join(HERE, sys.argv[3])
        dates = json.load(open(os.path.join(HERE, "_dates.json")))["dates"]; codes = codes_all()
        s, e = int(sys.argv[4]), int(sys.argv[5])
        phase_fill(col, path, s, e, dates, codes)
