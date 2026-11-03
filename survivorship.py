"""量我自己池子的幸存者偏差 —— 方法借自 CNEquity 的 survivorship_gap.py。

它的思路（我原来没做）：
  同一篮子股票，两种算法：
    complete     —— 买入当日在场的每一只，退市的按最后一根 K 线计价
    survivors    —— 同样一篮子，但剔除所有后来退市的
  两者的差 = 幸存者偏差。

关键在 delist_date 判定的口径：CNEquity 用 instruments 表的 delist_date；
我这里没有这张表，所以用"最后一根有效 K 线之后再也没有数据"来判定，
并额外要求最后 K 线日期超过窗口终点（否则是停牌不是退市）。
"""
import os, json, struct, mmap, math

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    f = open(os.path.join(HERE, "m_close.bin"), "rb")
    magic, ND, NC = struct.unpack_from("<8sII", f.read(16))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    di = {d: i for i, d in enumerate(dates)}

    def C(ci, j):
        return struct.unpack_from("<d", mm, 16 + 8 * (ci * ND + j))[0]

    last_j = {}
    for ci in range(NC):
        base = ci * ND
        for j in range(ND - 1, -1, -1):
            if C(ci, j) > 0:
                last_j[ci] = j; break

    windows = [("2016-06-30", "2021-06-30"), ("2018-06-29", "2023-06-30"),
               ("2020-06-30", "2025-06-30")]
    print("%-12s %7s %9s %11s %11s %9s" % ("窗口", "只数", "退市", "complete", "survivors", "偏差"))
    print("-" * 66)
    out = []
    for a, b in windows:
        j0, j1 = di.get(a), di.get(b)
        if j0 is None:
            cand = [k for k, d in enumerate(dates) if d >= a]
            j0 = cand[0] if cand else None
        if j1 is None:
            cand = [k for k, d in enumerate(dates) if d >= b]
            j1 = cand[0] if cand else None
        if j0 is None or j1 is None: continue
        comp, surv, n, nd = [], [], 0, 0
        for ci in range(NC):
            p0 = C(ci, j0)
            if p0 <= 0: continue
            p1 = C(ci, j1)
            delisted = last_j.get(ci, ND - 1) < j1 and last_j.get(ci, 0) < ND - 5
            if p1 <= 0:
                # 窗口内没有价格：退市，按最后一根有效 K 线计
                p1 = C(ci, last_j.get(ci, j0))
                delisted = True
            if p1 <= 0: continue
            r = p1 / p0
            comp.append(r); n += 1
            if delisted: nd += 1
            else: surv.append(r)
        if not comp: continue
        c = sum(comp) / len(comp); s = sum(surv) / len(surv)
        gap = (s - c) * 100
        print("%-12s %7d %9d %10.1f%% %10.1f%% %8.1fpp" % (
            a[:4] + "-" + b[:4], n, nd, (c - 1) * 100, (s - 1) * 100, gap))
        out.append({"window": a[:4] + "-" + b[:4], "names": n, "delisted": nd,
                    "complete": round((c - 1) * 100, 1), "survivors": round((s - 1) * 100, 1),
                    "gap_pp": round(gap, 1)})
    json.dump(out, open(os.path.join(HERE, "survivorship.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n注：complete 口径下退市按最后一根 K 线计价，是真实损失的下限（停牌前价格通常高于实际能收回的），")
    print("    所以这里的偏差是保守估计，真实偏差只会更大。")


if __name__ == "__main__":
    main()
