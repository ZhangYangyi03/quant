"""自建行业指数：用池内个股等权合成，不依赖外部接口。

为什么自己做：东财板块 K 线接口刚被拒，而且外部的板块指数成分会变，
不适合做严谨检验。用自己 5251 只的日线，按 stock_industry.json 的归属
等权合成，口径完全可控。

输出 ind_index.bin：行业 x 日 的等权收益矩阵（float32），附 ind_names.json
"""
import os, json, struct, mmap
import array

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    fh = open(os.path.join(HERE, "m_close.bin"), "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm, 0)
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    codes = meta["codes"]; dates = meta["dates"]
    ind = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))

    groups = {}
    for ci, c in enumerate(codes):
        g = ind.get(c)
        if g: groups.setdefault(g, []).append(ci)
    names = sorted(groups)
    # 只留有足够成分的行业
    names = [n for n in names if len(groups[n]) >= 5]
    print("行业 %d 个（成分>=5），覆盖 %d 只" % (
        len(names), sum(len(groups[n]) for n in names)), flush=True)

    NI = len(names)
    out = array.array("f", [0.0]) * (NI * ND)
    for k, n in enumerate(names):
        mem = groups[n]
        for j in range(1, ND):
            s = 0.0; cnt = 0
            for ci in mem:
                p0 = struct.unpack_from("<d", mm, 16 + 8 * (ci * ND + j - 1))[0]
                p1 = struct.unpack_from("<d", mm, 16 + 8 * (ci * ND + j))[0]
                if p0 > 0 and p1 > 0:
                    s += p1 / p0 - 1; cnt += 1
            out[k * ND + j] = s / cnt if cnt >= 3 else 0.0
        if k % 40 == 0: print("  %d/%d %s" % (k, NI, n), flush=True)

    with open(os.path.join(HERE, "ind_index.bin"), "wb") as f:
        f.write(struct.pack("<8sIII", b"INDIDX01", NI, ND, 0))
        out.tofile(f)
    json.dump({"names": names, "dates": dates},
              open(os.path.join(HERE, "ind_names.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 ind_index.bin  %.1f MB" % (os.path.getsize(os.path.join(HERE, "ind_index.bin")) / 1e6))


if __name__ == "__main__":
    main()
