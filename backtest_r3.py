"""第三轮：以"可执行等权全A"（8.33%）为基线做增强。

上一轮确认了基线不是沪深300(1.92%)，也不是理论每日等权(14.42%)，
而是月度实际再平衡、扣真实费率的等权全A：8.33%。

这一轮只做三件不做选股的事，每件只改一个条件：
  A  剔除波动最高的 30%（低波倾斜），其余等权
  B  剔除跌破 200 日均线的，其余等权
  C  A 和 B 叠加，再加一个空仓开关：中证500 在 200 日线下时持现金
每个都跟同一区间的基线比。增强的定义是"夏普更高"，不是"收益更高" ——
只在收益上找增强，最后一定会得到一串过拟合的十年期冠军。
"""
import sys, os, json, struct, mmap, time

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "px_flat.bin")
META = os.path.join(HERE, "px_index_meta.json")


class Flat:
    def __init__(self):
        fp = open(FLAT, "rb")
        self.mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.ND, self.NC = struct.unpack_from("<8sII", self.mm, 0)
        self.B_OPEN = 16
        self.B_CLOSE = 16 + 8 * self.ND * self.NC
        meta = json.load(open(META, encoding="utf-8"))
        self.dates = meta["dates"]; self.codes = meta["codes"]


def main(which="A", cash=1_000_000.0, start="2010-06-01", vol_n=120, cut=0.30):
    F = Flat()
    import struct as _s
    def C(i): return _s.unpack_from("<d", F.mm, F.B_CLOSE + 8 * i)[0]
    def O(i): return _s.unpack_from("<d", F.mm, F.B_OPEN + 8 * i)[0]
    ND, NC, dates = F.ND, F.NC, F.dates
    si = next(i for i, d in enumerate(dates) if d >= start)

    # 指数择时用的中证500 收盘（与扁平矩阵日期对齐；缺的沿用前值）
    switch = None
    p = os.path.join(HERE, "idx_中证500.json")
    if os.path.exists(p) and which == "C":
        raw = dict((d, v) for d, v in json.load(open(p, encoding="utf-8")))
        ser = []; last = None
        for d in dates:
            last = raw.get(d, last); ser.append(last)
        ma200 = [None] * ND
        for j in range(200, ND):
            w = ser[j - 199:j + 1]
            if all(x is not None for x in w): ma200[j] = sum(w) / len(w)
        switch = (ser, ma200)

    pos = {}; cashleft = cash; lastc = {}; curve = []; fees = 0.0; fills = 0
    prev_ym = None; t0 = time.time(); rebal = 0
    for j in range(si, ND):
        d = dates[j]
        mv = 0.0
        for ci, sh in pos.items():
            px = C(ci * ND + j)
            if px > 0: lastc[ci] = px
            mv += sh * lastc.get(ci, 0.0)
        eq = cashleft + mv
        curve.append([d, round(eq, 2)])

        if d[:7] != prev_ym:
            prev_ym = d[:7]
            jn = j + 1
            if jn >= ND: break
            if switch is not None:
                s, ma = switch
                on = (ma[j] is not None and s[j] is not None and s[j] > ma[j])
            else:
                on = True
            live = []
            if on:
                # 波动率：近 vol_n 日收盘收益率的标准差；只在有连续行情的票上算
                vols = []
                for ci in range(NC):
                    base = ci * ND
                    if C(base + j) <= 2.0: continue
                    if O(base + jn) <= 0: continue
                    rets = []; prev = 0.0; ok = True
                    for k in range(j - vol_n, j + 1):
                        pv = C(base + k)
                        if pv <= 0: ok = False; break
                        if prev > 0: rets.append(pv / prev - 1)
                        prev = pv
                    if not ok or len(rets) < vol_n - 2: continue
                    m = sum(rets) / len(rets)
                    vol = (sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
                    if which in ("A", "C") and vol > 0:
                        vols.append((ci, vol))
                    elif which == "B":
                        live.append(ci)
                if which in ("A", "C"):
                    # 跌破 200 日线的不进（B 用），A 只看波动
                    vols.sort(key=lambda x: x[1])
                    keepn = int(len(vols) * (1 - cut))
                    live = [ci for ci, _ in vols[:keepn]]
                    if which == "C":
                        ma200 = {}
                        f2 = []
                        for ci in live:
                            base = ci * ND
                            w = [C(base + k) for k in range(j - 199, j + 1)]
                            if any(x <= 0 for x in w): continue
                            if C(base + j) > sum(w) / 200: f2.append(ci)
                        live = f2
            if not live:
                for ci in list(pos):
                    sh = pos.pop(ci); px = O(ci * ND + jn) or lastc.get(ci, 0.0)
                    amt = sh * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                    cashleft += amt - fee; fees += fee; fills += 1
                continue
            per = eq / len(live)
            keep = set(live)
            for ci in list(pos):
                if ci in keep: continue
                sh = pos.pop(ci); px = O(ci * ND + jn)
                if px <= 0: px = lastc.get(ci, 0.0)
                amt = sh * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                cashleft += amt - fee; fees += fee; fills += 1
            for ci in keep:
                px = O(ci * ND + jn)
                if px <= 0: continue
                have = pos.get(ci, 0) * px
                if have >= per * 0.9: continue
                qq = int((per - have) / px / 100) * 100
                if qq <= 0: continue
                amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
                if amt + fee > cashleft: continue
                cashleft -= amt + fee; fees += fee; pos[ci] = pos.get(ci, 0) + qq; fills += 1
            rebal += 1
        if j % 1000 == 0:
            print("  %d/%d %s 净值 %.0f 持仓 %d %.0fs" % (j, ND, d, eq, len(pos), time.time() - t0), flush=True)

    json.dump({"curve": curve, "fees": round(fees), "fills": fills, "which": which,
               "cut": cut, "vol_n": vol_n}, open(os.path.join(HERE, "r3_%s.json" % which), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：净值 %.2f 倍 费用 %.0f 成交 %d 调仓 %d %.0fs" % (
        which, curve[-1][1] / curve[0][1], fees, fills, rebal, time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
