"""把 stockdata/*.csv 转成列式二进制（mmap 用），并把全市场挂到 entitygraph 图层上。

输出：
  store/<code>.bin   float64 矩阵，7 列：open close high low volume amount pad
  store/<code>.idx   日期表（前 4 字节小端条数，之后每 10 字节一个 YYYY-MM-DD）
  graph_links.json   A 股代码 -> entitygraph 节点/板块/行业标签
"""
import os, csv, struct, json

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "stockdata")
DST = os.path.join(HERE, "store")
MAGIC = b"MBAR0001"
COLS = 7


def convert():
    os.makedirs(DST, exist_ok=True)
    done = 0; skipped = 0; total_rows = 0
    for fn in os.listdir(SRC):
        if not fn.endswith(".csv") or fn.startswith("_"): continue
        code = fn[:-4]
        rows = []
        with open(os.path.join(SRC, fn), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    d = r["date"]
                    if len(d) != 10: continue
                    rows.append((d, float(r["open"]), float(r["close"]), float(r["high"]),
                                 float(r["low"]), float(r["volume"] or 0), float(r["amount"] or 0)))
                except (ValueError, KeyError, TypeError):
                    continue
        if len(rows) < 60: skipped += 1; continue
        rows.sort(key=lambda x: x[0])
        with open(os.path.join(DST, code + ".bin"), "wb") as f:
            f.write(MAGIC)
            for r in rows:
                f.write(struct.pack("<7d", r[1], r[2], r[3], r[4], r[5], r[6], 0.0))
        with open(os.path.join(DST, code + ".idx"), "wb") as f:
            f.write(struct.pack("<I", len(rows)))
            for r in rows:
                f.write(r[0].encode("ascii"))
        done += 1; total_rows += len(rows)
    return done, skipped, total_rows


if __name__ == "__main__":
    d, s, n = convert()
    print("转换 %d 只，跳过 %d 只，共 %d 行" % (d, s, n))
