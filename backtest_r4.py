"""用 entity-graph 的交易所名录做行业中性回测。

这一轮换的是数据源，不是参数：行业分类来自 entity-graph 的深交所名录
（证监会门类，A-S 共 18 类），带交易所 URL 出处，不是猜的。

覆盖只有 1180 只（且仅当前在册），所以池子从 3160 缩到覆盖率内的部分。
这会让绝对收益不可比 —— 所以每个策略都跟"同一池子内的等权"比，
不同池子之间不比。这是唯一诚实的读法。

三组对照：
  raw      全池等权
  lowvol   池内波动最低的 30%，等权
  ind_neu  每个行业门类内部各取低波 30%，等权（行业中性）
如果低波优势在行业中性后消失，说明它赚的是行业暴露；
如果还在，说明它是真的选股信号。
"""
import os, json, struct, mmap, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "px_flat.bin")
META = os.path.join(HERE, "px_index_meta.json")
IND = os.path.join(HERE, "eg_cn_industry.json")


def main(which="ind_neu", cash=1_000_000.0, start="2010-06-01", vol_n=120, cut=0.30):
    fp = open(FLAT, "rb")
    mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm, 0)
    B_O = 16; B_C = 16 + 8 * ND * NC
    def C(i): return struct.unpack_from("<d", mm, B_C + 8 * i)[0]
    def O(i): return struct.unpack_from("<d", mm, B_O + 8 * i)[0]

    meta = json.load(open(META, encoding="utf-8"))
    dates = meta["dates"]; codes = meta["codes"]
    indm = json.load(open(IND, encoding="utf-8"))["ind"]
    # 只在有行业分类的股票里操作
    pool = [ci for ci, c in enumerate(codes) if c in indm]
    grp = {ci: indm[codes[ci]] for ci in pool}
    print("池子 %d 只（有交易所行业分类）" % len(pool), flush=True)

    si = next(i for i, d in enumerate(dates) if d >= start)
    pos = {}; cashleft = cash; lastc = {}; curve = []; fees = 0.0; fills = 0
    prev_ym = None; t0 = time.time()
    for j in range(si, ND):
        d = dates[j]
        mv = 0.0
        for ci, sh in pos.items():
            px = C(ci * ND + j)
            if px > 0: lastc[ci] = px
            mv += sh * lastc.get(ci, 0.0)
        equity = cashleft + mv
        curve.append([d, round(equity, 2)])

        if d[:7] != prev_ym:
            prev_ym = d[:7]
            jn = j + 1
            if jn >= ND: break
            cand = []
            for ci in pool:
                base = ci * ND
                if C(base + j) <= 2.0 or O(base + jn) <= 0: continue
                rets = []; prev = 0.0; ok = True
                for k in range(j - vol_n, j + 1):
                    pv = C(base + k)
                    if pv <= 0: ok = False; break
                    if prev > 0: rets.append(pv / prev - 1)
                    prev = pv
                if not ok or len(rets) < vol_n - 2: continue
                m = sum(rets) / len(rets)
                vol = (sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
                if vol > 0: cand.append((ci, vol))

            if which == "raw":
                live = [ci for ci, _ in cand]
            elif which == "lowvol":
                cand.sort(key=lambda x: x[1])
                live = [ci for ci, _ in cand[:max(1, int(len(cand) * (1 - cut)))]]
            else:  # ind_neu
                byg = {}
                for ci, vol in cand: byg.setdefault(grp[ci], []).append((ci, vol))
                live = []
                for g, arr in byg.items():
                    arr.sort(key=lambda x: x[1])
                    live += [ci for ci, _ in arr[:max(1, int(len(arr) * (1 - cut)))]]

            if not live: continue
            per = equity / len(live)
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
        if j % 1500 == 0:
            print("  %d/%d %s 净值 %.0f 持仓 %d %.0fs" % (j, ND, d, equity, len(pos), time.time() - t0), flush=True)

    json.dump({"curve": curve, "fees": round(fees), "fills": fills, "which": which, "pool": len(pool)},
              open(os.path.join(HERE, "r4_%s.json" % which), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：净值 %.2f 倍 费用 %.0f 成交 %d %.0fs" % (which, curve[-1][1] / curve[0][1], fees, fills, time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ind_neu")
