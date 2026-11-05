import os, csv, json, struct, mmap, time, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "stockdata")
COLS = {"open": 1, "close": 2, "low": 4, "high": 3, "amount": 6, "volume": 5}

def codes_all():
    return sorted(f[:-4] for f in os.listdir(SRC) if f.endswith(".csv") and not f.startswith("_"))

def main(name, start, end):
    col = COLS[name]
    dates = json.load(open(os.path.join(HERE, "_dates.json")))["dates"]
    codes = codes_all(); ND = len(dates); NC = len(codes)
    di = {d: i for i, d in enumerate(dates)}
    path = os.path.join(HERE, "m_%s.bin" % name)
    need = 16 + 8 * ND * NC
    if start == 0 or not os.path.exists(path) or os.path.getsize(path) != need:
        with open(path, "wb") as f:
            f.seek(need - 1); f.write(b"\0")
    fp = open(path, "r+b"); mm = mmap.mmap(fp.fileno(), 0)
    if start == 0:
        struct.pack_into("<8sII", mm, 0, ("M" + name[:7]).encode()[:8].ljust(8, b"\0"), ND, NC)
        CH = 16 * 1024 * 1024
        o = 16; total = need
        while o < total:
            n = min(CH, total - o); mm[o:o + n] = b"\0" * n; o += n
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
    print("  %s %d-%d %.0fs" % (name, start, min(end, NC), time.time() - t0), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
