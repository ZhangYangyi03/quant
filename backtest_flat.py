"""直接用扁平矩阵跑截面策略。

与 run_universe_all.py 的区别：不再逐股 mmap + 二分，改读 px_flat.bin
（3160 只 × 8709 天，开盘/收盘两个 d 矩阵，按 code*ND+date 定位）。

这类回测的瓶颈一直是"按 (股票, 日期) 随机访问"，不是策略本身。
矩阵一落到磁盘上，同一个调仓日的截面就是一列连续内存，速度差两个数量级。
"""
import sys, os, json, time, struct, mmap
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "px_flat.bin")
META = os.path.join(HERE, "px_index_meta.json")

MAGIC = b"PXLF0001"


class Flat:
    def __init__(self):
        fp = open(FLAT, "rb")
        self.mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.ND, self.NC = struct.unpack_from("<8sII", self.mm, 0)
        assert magic == MAGIC, "px_flat.bin 不是本程序写的"
        self.B_OPEN = 16
        self.B_CLOSE = 16 + 8 * self.ND * self.NC
        meta = json.load(open(META, encoding="utf-8"))
        self.dates = meta["dates"]; self.codes = meta["codes"]
        self.di = {d: i for i, d in enumerate(self.dates)}
        self.ci = {c: i for i, c in enumerate(self.codes)}

    def o(self, flat_i):
        return struct.unpack_from("<d", self.mm, self.B_OPEN + 8 * flat_i)[0]

    def c(self, flat_i):
        return struct.unpack_from("<d", self.mm, self.B_CLOSE + 8 * flat_i)[0]


def cross_section(F, j, kind, n):
    """第 j 天收盘后的截面打分。只用 <= j 的数据。"""
    out = []
    for ci_, code in enumerate(F.codes):
        base = ci_ * F.ND
        if j - n - 20 < 0: continue
        px = F.c(base + j)
        if px < 2.0: continue          # 停牌(0) 或 低价股
        rets = []
        prev = 0.0
        ok = True
        for k in range(j - n, j + 1):
            p = F.c(base + k)
            if p <= 0:
                ok = False; break
            if prev > 0:
                r = p / prev - 1
                if r > 0.3 or r < -0.3:   # 异常跳价（复权/数据错误），跳过这只
                    ok = False; break
                rets.append(r)
            prev = p
        if not ok or len(rets) < n - 2: continue
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
        vol = var ** 0.5
        cum = 1.0
        for x in rets: cum *= (1 + x)
        cum -= 1.0
        if kind == "lowvol":
            if vol <= 0: continue
            out.append((code, -vol))
        elif kind == "mom":
            out.append((code, cum))
        elif kind == "momvol":
            if vol <= 0: continue
            out.append((code, cum / vol))
        elif kind == "rev":
            out.append((code, -cum))
    return out


def main(which="a", k=50, n=60, rebal=20, cash=1_000_000.0, start="2010-01-01", tag=None):
    F = Flat()
    dates = [i for i, d in enumerate(F.dates) if d >= start]
    kind = {"a": ("lowvol", 60), "b": ("mom", 60), "c": ("rev", 20),
            "d": ("momvol", 60), "e": ("lowvol", 120), "f": ("lowvol", 250)}[which]
    kind, n = kind
    pos = {}          # code -> shares
    cashleft = cash
    curve = []
    lastc = {}
    fees = 0.0
    fills = 0
    t0 = time.time()
    nrebal = 0
    for idx, j in enumerate(dates):
        d = F.dates[j]
        # 先按当日收盘估值
        mv = 0.0
        for code, sh in pos.items():
            p = F.c(F.ci[code] * F.ND + j)
            if p <= 0: p = lastc.get(code, 0.0)
            else: lastc[code] = p
            mv += sh * p
        equity = cashleft + mv
        curve.append([d, round(equity, 2)])

        if idx >= n and (idx - n) % rebal == 0:
            sc = cross_section(F, j, kind, n)
            if len(sc) >= k:
                target = [c for c, _ in sorted(sc, key=lambda x: -x[1])[:k]]
                nrebal += 1
                # 次日开盘撮合：卖出不在目标里的，再等权买入
                jn = j + 1
                if jn < F.ND:
                    for code in list(pos):
                        if code in target: continue
                        sh = pos.pop(code)
                        p = F.o(F.ci[code] * F.ND + jn)
                        if p <= 0: p = lastc.get(code, 0.0)
                        amt = sh * p
                        fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                        cashleft += amt - fee; fees += fee; fills += 1
                    per = (cashleft + sum(sh * lastc.get(c, 0) for c, sh in pos.items())) / max(len(target), 1)
                    for code in target:
                        p = F.o(F.ci[code] * F.ND + jn)
                        if p <= 0: continue
                        have = pos.get(code, 0) * p
                        if have >= per * 0.9: continue
                        budget = per - have
                        q = int(budget / p / 100) * 100
                        if q <= 0: continue
                        amt = q * p
                        fee = max(5.0, amt * 0.00025) + amt * 0.0001
                        if amt + fee > cashleft: continue
                        cashleft -= amt + fee; fees += fee
                        pos[code] = pos.get(code, 0) + q; fills += 1
        if idx % 500 == 0:
            print("  %d/%d %s 净值 %.0f 持仓 %d %.0fs" % (idx, len(dates), d, equity, len(pos), time.time() - t0), flush=True)

    out = {"curve": curve, "fees": round(fees), "fills": fills, "rebalances": nrebal,
           "kind": kind, "n": n, "k": k, "rebal": rebal}
    name = tag or "%s%d_k%d" % (kind, n, k)
    json.dump(out, open(os.path.join(HERE, "flat_%s.json" % name), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：%d 天 净值 %.2f 倍 费用 %.0f 成交 %d 次 %.0fs" % (
        name, len(curve), curve[-1][1] / curve[0][1], fees, fills, time.time() - t0))


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "a"
    main(w)
