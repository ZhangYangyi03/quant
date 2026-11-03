import sys, os, time, struct, csv
sys.path.insert(0, r"D:\Users\china\Desktop\项目_开发\quant")
import build_store as B

def needs(code):
    bi = os.path.join(B.DST, code + ".bin"); ci = os.path.join(B.SRC, code + ".csv")
    if not os.path.exists(bi): return True
    return os.path.getmtime(ci) > os.path.getmtime(bi) + 1

codes = [f[:-4] for f in os.listdir(B.SRC) if f.endswith(".csv") and not f.startswith("_")]
todo = [c for c in codes if needs(c)]
print("待转", len(todo), "已有", len(codes) - len(todo), flush=True)
t0 = time.time(); done = 0
for code in todo:
    fn = code + ".csv"
    rows = []
    with open(os.path.join(B.SRC, fn), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                d = r["date"]
                if len(d) != 10: continue
                rows.append((d, float(r["open"]), float(r["close"]), float(r["high"]),
                             float(r["low"]), float(r["volume"] or 0), float(r["amount"] or 0)))
            except (ValueError, KeyError, TypeError):
                continue
    if len(rows) < 60: continue
    rows.sort(key=lambda x: x[0])
    with open(os.path.join(B.DST, code + ".bin"), "wb") as f:
        f.write(B.MAGIC)
        for r in rows: f.write(struct.pack("<7d", r[1], r[2], r[3], r[4], r[5], r[6], 0.0))
    with open(os.path.join(B.DST, code + ".idx"), "wb") as f:
        f.write(struct.pack("<I", len(rows)))
        for r in rows: f.write(r[0].encode("ascii"))
    done += 1
print("增量转换", done, "只 %.0fs" % (time.time() - t0))
