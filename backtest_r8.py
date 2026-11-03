"""行业中性化：EP 的超额里，多少是押对行业换来的。

做法：每个调仓日，在每个细分行业内按 EP 排序，取行业内最好的若干只，
让组合的行业权重尽量均匀，而不是让银行地产（EP 天然高）占满。

三组对照，只改一个条件：
  ep_nofilter   纯 EP 排序，全市场取前 k 只（现状）
  ep_ind_neutral 每个行业内取 EP 最高的部分，让行业间等权
  ep_ind_within  控制行业暴露后再排序（行业内 EP 排名，跨行业合并）
"""
import os, json, struct, mmap, time, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DISCLOSURE_LAG = {3: 30, 6: 62, 9: 31, 12: 120}


def load_fund():
    raw = json.load(open(os.path.join(HERE, "fundamentals.json"), encoding="utf-8"))
    out = {}
    for code, rows in raw.items():
        if not isinstance(rows, list): continue
        ser = sorted([(r["d"], r["eps"]) for r in rows if r.get("d") and r.get("eps") is not None])
        if not ser: continue
        fixed = []
        for i, (d, eps) in enumerate(ser):
            m = int(d[5:7])
            if m == 12:
                ttm = eps
            else:
                pf = ps = None
                for j in range(i - 1, -1, -1):
                    dj = ser[j][0]
                    if dj[:4] == str(int(d[:4]) - 1):
                        if dj[5:7] == "12": pf = ser[j][1]
                        if dj[5:7] == d[5:7]: ps = ser[j][1]
                ttm = eps + pf - ps if (pf is not None and ps is not None) else eps
            fixed.append((d, ttm))
        out[code] = fixed
    return out


def avail(d):
    import datetime
    y, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
    return (datetime.date(y, m, dd) + datetime.timedelta(days=DISCLOSURE_LAG.get(m, 90))).isoformat()


def main(which="ep_ind_within", k=50, cash=1_000_000.0, start="2010-06-01"):
    fh_c = open(os.path.join(HERE, "m_close.bin"), "rb")
    fh_o = open(os.path.join(HERE, "m_open.bin"), "rb")
    fh_a = open(os.path.join(HERE, "amt_flat.bin"), "rb")
    mm_c = mmap.mmap(fh_c.fileno(), 0, access=mmap.ACCESS_READ)
    mm_o = mmap.mmap(fh_o.fileno(), 0, access=mmap.ACCESS_READ)
    mm_a = mmap.mmap(fh_a.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm_c, 0)
    def C(i): return struct.unpack_from("<d", mm_c, 16 + 8 * i)[0]
    def O(i): return struct.unpack_from("<d", mm_o, 16 + 8 * i)[0]
    def A(i): return struct.unpack_from("<d", mm_a, 16 + 8 * i)[0]

    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates = meta["dates"]; codes = meta["codes"]
    ci_of = {c: i for i, c in enumerate(codes)}
    ind = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))

    last_j = {}
    for ci in range(NC):
        base = ci * ND
        for j2 in range(ND - 1, -1, -1):
            if C(base + j2) > 0:
                last_j[ci] = j2; break

    prep = {}
    for code, rows in load_fund().items():
        if code in ci_of:
            prep[code] = [(avail(r[0]), r[1]) for r in rows]
    print("覆盖 %d 只，行业 %d 类" % (len(prep), len(set(ind.values()))), flush=True)

    si = next(i for i, d in enumerate(dates) if d >= start)
    pos = {}; cashleft = cash; lastc = {}; curve = []; fees = 0.0
    fills = 0; prev_ym = None; t0 = time.time()
    ind_hist = []
    for j in range(si, ND):
        d = dates[j]
        mv = 0.0
        for ci, sh in list(pos.items()):
            px = C(ci * ND + j)
            if px > 0:
                lastc[ci] = px
            elif j > last_j.get(ci, j):
                continue          # 退市：市值记 0，不产生现金
            mv += sh * lastc.get(ci, 0.0)
        equity = cashleft + mv
        curve.append([d, round(equity, 2)])
        if d[:7] == prev_ym: continue
        prev_ym = d[:7]
        jn = j + 1
        if jn >= ND: break

        snap = {}
        for code, rows in prep.items():
            lo, hi = 0, len(rows) - 1; found = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if rows[mid][0] <= d: found = mid; lo = mid + 1
                else: hi = mid - 1
            if found >= 0: snap[code] = rows[found][1]

        cand = []
        for code, ttm in snap.items():
            ci = ci_of[code]; base = ci * ND
            px = C(base + j)
            if px <= 2.0 or O(base + jn) <= 0: continue
            if not ttm: continue
            cand.append((code, ttm / px, ind.get(code)))
        if not cand: continue

        if which == "ep_nofilter":
            cand.sort(key=lambda x: -x[1])
            live = [c[0] for c in cand[:k]]
        elif which == "ep_ind_neutral":
            # 每个行业取 EP 最高的 3 只，行业间均匀，直到凑满 k
            byind = defaultdict(list)
            for c in cand: byind[c[2]].append(c)
            for g in byind.values(): g.sort(key=lambda x: -x[1])
            live = []
            for depth in range(3):
                for g in byind.values():
                    if depth < len(g): live.append(g[depth][0])
                    if len(live) >= k: break
                if len(live) >= k: break
            live = live[:k]
        elif which == "ep_ind_within":
            # 行业内 EP 百分位排名，跨行业合并 —— 消除行业间 EP 水平差异
            byind = defaultdict(list)
            for c in cand:
                if c[2]: byind[c[2]].append(c)
            scored = []
            for g in byind.values():
                g.sort(key=lambda x: -x[1])
                n = len(g)
                for rank, c in enumerate(g):
                    scored.append((c[0], 1.0 - rank / max(n, 1)))
            scored.sort(key=lambda x: -x[1])
            live = [c[0] for c in scored[:k]]
        else:
            live = [c[0] for c in cand[:k]]
        if not live: continue

        # 记录行业集中度（用于判断是否真的中性化了）
        cnt = defaultdict(int)
        for c in live: cnt[ind.get(c)] += 1
        ind_hist.append(max(cnt.values()) if cnt else 0)

        per = equity / len(live); keep = set(live)
        for ci in list(pos):
            if codes[ci] in keep: continue
            sh = pos.pop(ci); px = O(ci * ND + jn)
            if px <= 0: px = lastc.get(ci, 0.0)
            if px <= 0: continue
            amt = sh * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
            capamt = A(ci * ND + jn) * 0.10
            if capamt > 0 and amt > capamt:
                sellsh = int(capamt / px); amt = sellsh * px
                fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                cashleft += amt - fee; fees += fee; fills += 1
                if sh > sellsh: pos[ci] = sh - sellsh
                continue
            cashleft += amt - fee; fees += fee; fills += 1
        for code in keep:
            ci = ci_of[code]; px = O(ci * ND + jn)
            if px <= 0: continue
            have = pos.get(ci, 0) * px
            if have >= per * 0.9: continue
            want = min(per - have, A(ci * ND + jn) * 0.10 or per - have)
            qq = int(want / px / 100) * 100
            if qq <= 0: continue
            amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
            if amt + fee > cashleft: continue
            cashleft -= amt + fee
            pos[ci] = pos.get(ci, 0) + qq; fills += 1

    json.dump({"curve": curve, "fees": round(fees), "fills": fills, "which": which, "k": k,
               "max_ind_weight": round(sum(ind_hist) / max(len(ind_hist), 1), 2)},
              open(os.path.join(HERE, "r8_%s.json" % which), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：倍数 %.2f 最大行业集中度均值 %.1f/50 %.0fs" % (
        which, curve[-1][1] / curve[0][1], sum(ind_hist) / max(len(ind_hist), 1), time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ep_ind_within")
