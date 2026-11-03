"""容量约束回测：15.87% 到底能不能吃到。

上一轮发现"排除成交额最小 30% 后等权"从 7.52% 跳到 15.87%。
问题是：被排除的恰恰是成交量小的股票，它们买得进、卖得出吗？
这一轮给每笔交易加上硬约束：

    单笔买入金额 <= 该股当日成交额 × CAP
    单笔卖出数量 <= 该股当日成交额 × CAP / 价格

CAP 取 1% / 5% / 10%，看收益随 CAP 收紧掉多少。
掉得越快，说明那 8.35 个百分点越是账面数字。
"""
import os, json, struct, mmap, time, sys, bisect

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.join(HERE, "px_flat.bin")
MA = os.path.join(HERE, "amt_flat.bin")
META = os.path.join(HERE, "px_index_meta.json")
FUND = os.path.join(HERE, "fundamentals.json")

DISCLOSURE_LAG = {3: 30, 6: 62, 9: 31, 12: 120}
EXCLUDE_SMALLEST = 0.30


def load_fund():
    raw = json.load(open(FUND, encoding="utf-8"))
    out = {}
    for code, rows in raw.items():
        if not isinstance(rows, list): continue
        ser = sorted([(r["d"], r["eps"]) for r in rows if r.get("d") and r.get("eps") is not None])
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
        if fixed: out[code] = fixed
    return out


def avail(d):
    import datetime
    y, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
    return (datetime.date(y, m, dd) + datetime.timedelta(days=DISCLOSURE_LAG.get(m, 90))).isoformat()


def main(which="eq_shell", cap=0.10, k=50, cash=1_000_000.0, start="2010-06-01"):
    fp = open(MP, "rb"); mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm, 0)
    B_O = 16; B_C = 16 + 8 * ND * NC
    fa = open(MA, "rb"); ma = mmap.mmap(fa.fileno(), 0, access=mmap.ACCESS_READ)
    def C(i): return struct.unpack_from("<d", mm, B_C + 8 * i)[0]
    def O(i): return struct.unpack_from("<d", mm, B_O + 8 * i)[0]
    def A(i):
        v = struct.unpack_from("<d", ma, 16 + 8 * i)[0]
        return v if v > 0 else 0.0

    meta = json.load(open(META, encoding="utf-8"))
    dates = meta["dates"]; codes = meta["codes"]
    ci_of = {c: i for i, c in enumerate(codes)}
    fs = load_fund()
    prep = {c: [(avail(r[0]), r[1]) for r in rows] for c, rows in fs.items() if c in ci_of}
    print("因子覆盖 %d 只，CAP=%.0f%%" % (len(prep), cap * 100), flush=True)

    si = next(i for i, d in enumerate(dates) if d >= start)
    pos = {}; cashleft = cash; lastc = {}; curve = []; fees = 0.0
    fills = 0; blocked = 0; prev_ym = None; t0 = time.time()
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
                if not ttm or ttm == 0: continue
                adv = 0.0; cnt = 0
                for kk in range(max(0, j - 19), j + 1):
                    a = A(base + kk)
                    if a > 0: adv += a; cnt += 1
                adv = adv / cnt if cnt else 0.0
                cand.append((code, ttm / px, adv))
            if not cand: continue
            if which != "eq_nofilter":
                cand.sort(key=lambda x: x[2])
                cand = cand[int(len(cand) * EXCLUDE_SMALLEST):]
            if not cand: continue
            if which.startswith("ep"):
                cand.sort(key=lambda x: -x[1])
            live = [c for c, _, _ in cand[:k]]
            if not live: continue

            per = equity / len(live)
            keep = set(live)
            for ci in list(pos):
                if codes[ci] in keep: continue
                sh = pos.pop(ci); px = O(ci * ND + jn)
                if px <= 0: px = lastc.get(ci, 0.0)
                if px <= 0: continue
                capsh = int(A(ci * ND + jn) * cap / px)
                if capsh <= 0: pos[ci] = sh; blocked += 1; continue
                sell = min(sh, capsh)
                amt = sell * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                cashleft += amt - fee; fees += fee; fills += 1
                if sh - sell > 0: pos[ci] = sh - sell; blocked += 1
            for code in keep:
                ci = ci_of[code]; px = O(ci * ND + jn)
                if px <= 0: continue
                have = pos.get(ci, 0) * px
                if have >= per * 0.9: continue
                want = per - have
                capamt = A(ci * ND + jn) * cap
                if capamt <= 0: blocked += 1; continue
                use = min(want, capamt)
                qq = int(use / px / 100) * 100
                if qq <= 0: blocked += 1; continue
                amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
                if amt + fee > cashleft: continue
                cashleft -= amt + fee; fees += fee; pos[ci] = pos.get(ci, 0) + qq; fills += 1
        if j % 3000 == 0:
            print("  %d/%d %s 净值 %.0f 持仓 %d %.0fs" % (j, ND, d, equity, len(pos), time.time() - t0), flush=True)

    tag = "%s_cap%02d" % (which, int(cap * 100))
    json.dump({"curve": curve, "fees": round(fees), "fills": fills, "blocked": blocked, "cap": cap, "which": which},
              open(os.path.join(HERE, "r6_%s.json" % tag), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：净值 %.2f 倍 费用 %.0f 成交 %d 受限 %d %.0fs" % (
        tag, curve[-1][1] / curve[0][1], fees, fills, blocked, time.time() - t0))


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "eq_shell"
    c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    main(w, c)
