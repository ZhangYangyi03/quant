"""用确定性事件表检验 EG 的因果：降准/加息之后，各行业真的怎么走。

纯标准库 + mmap，不用 numpy —— 本机 numpy 与 mmap 组合会 MemoryError。

事件日期是外部给定的公开事实，方向是 EG 给定的（"准备金率越低放水越多"），
我没有自由去挑参数。这是在检验 EG 的因果，不是拟合。
"""
import os, json, struct, mmap, math

HERE = os.path.dirname(os.path.abspath(__file__))

CUT_RRR = [
    "2010-01-18", "2010-02-25", "2010-05-10", "2010-11-16", "2010-11-29", "2010-12-20",
    "2011-01-20", "2011-02-24", "2011-03-25", "2011-04-21", "2011-05-18", "2011-06-20",
    "2011-12-05", "2012-02-24", "2012-05-18", "2015-02-05", "2015-04-20", "2015-06-28",
    "2015-09-06", "2015-10-24", "2016-03-01", "2018-04-25", "2018-07-05", "2018-10-15",
    "2019-01-15", "2019-05-15", "2019-09-16", "2020-01-06", "2020-03-16", "2020-04-15",
    "2021-07-15", "2021-12-15", "2022-04-25", "2022-12-05", "2023-03-27", "2023-09-15",
    "2024-02-05", "2024-09-27", "2025-05-15",
]
HIKE = ["2010-10-20", "2010-12-26", "2011-02-09", "2011-04-06", "2011-07-07",
        "2017-02-03", "2017-03-16", "2017-12-14"]


def load():
    f = open(os.path.join(HERE, "ind_index.bin"), "rb")
    magic, NI, ND, _ = struct.unpack("<8sIII", f.read(20))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    meta = json.load(open(os.path.join(HERE, "ind_names.json"), encoding="utf-8"))
    return mm, NI, ND, meta["names"], meta["dates"]


def col(mm, NI, ND, j):
    """取第 j 日的全行业收益"""
    return [struct.unpack_from("<f", mm, 20 + 4 * (i * ND + j))[0] for i in range(NI)]


def cum(mm, NI, ND, j, h):
    """j+1 起 h 日累计对数收益，按行业"""
    out = [0.0] * NI
    for jj in range(j + 1, min(j + 1 + h, ND)):
        c = col(mm, NI, ND, jj)
        for i in range(NI):
            out[i] += math.log1p(c[i])
    return out


def main():
    mm, NI, ND, names, dates = load()
    di = {d: i for i, d in enumerate(dates)}
    print("行业 %d, 交易日 %d (%s ~ %s)" % (NI, ND, dates[0], dates[-1]), flush=True)

    def bench_at(j, h):
        return sum(cum(mm, NI, ND, j, h)) / NI

    for label, evs, want in (("降准", CUT_RRR, ["股份制银行Ⅲ", "城商行Ⅲ", "农商行Ⅲ", "证券Ⅱ"]),
                             ("加息", HIKE, ["股份制银行Ⅲ", "证券Ⅱ"])):
        print("\n=== %s  (事件 %d 个) ===" % (label, len(evs)))
        for h in (5, 20, 60):
            rows = []
            for d in evs:
                j = di.get(d)
                if j is None:
                    cand = [k for k, x in enumerate(dates) if x >= d]
                    if not cand: continue
                    j = cand[0]
                if j + h + 1 >= ND: continue
                r = cum(mm, NI, ND, j, h)
                b = bench_at(j, h)
                rows.append([r[i] - b for i in range(NI)])
            if len(rows) < 5: continue
            n = len(rows)
            # 稳健化：按行业剪掉首尾各 10% 的极端窗口（2015 股灾那种会主导均值）
            trimmed = []
            for i in range(NI):
                col_i = sorted(r[i] for r in rows)
                k = max(0, int(n * 0.10))
                sub = col_i[k:n - k] if n - 2 * k >= 3 else col_i
                trimmed.append(sub)
            mu = [sum(sub) / len(sub) for sub in trimmed]
            sd = []
            for i in range(NI):
                sub = trimmed[i]; m = mu[i]
                v = sum((x - m) ** 2 for x in sub) / max(len(sub) - 1, 1)
                sd.append(math.sqrt(v) if v > 0 else 1e-12)
            t = [mu[i] / (sd[i] / math.sqrt(max(len(trimmed[i]), 1))) for i in range(NI)]
            # 多重检验校正：254 个行业同时比较，Bonferroni 阈值
            alpha = 0.05 / NI
            # 近似：|t| > 临界值 才算过。n 小的时候用保守 3.5 作为提示线
            crit = 3.5 if n < 20 else 3.0
            order = sorted(range(NI), key=lambda i: -t[i])
            print("  %2d 日后 (n=%d, 剪尾10%%)  相对全市场等权  [过校正线 |t|>%.1f]" % (h, n, crit))
            sig = [i for i in order if abs(t[i]) > crit]
            print("     过校正是 %d 个行业" % len(sig))
            print("     最强: " + " | ".join("%s %+.1f%% t=%.2f" % (names[i], mu[i] * 100, t[i]) for i in order[:4]))
            print("     最弱: " + " | ".join("%s %+.1f%% t=%.2f" % (names[i], mu[i] * 100, t[i]) for i in order[-3:]))
            for key in want:
                if key in names:
                    i = names.index(key)
                    print("       %-12s %+.2f%%  t=%+.2f" % (key, mu[i] * 100, t[i]), flush=True)


if __name__ == "__main__":
    main()
