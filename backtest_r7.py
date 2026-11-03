"""全市场口径回测：5251 只、8816 个交易日。

矩阵拆成独立文件（close / open / amount），因为本机每进程提交上限约 490MB,
合在一个文件里 8*ND*NC*2 就超过上限。

对照设计（只改一个条件）：
  eq_nofilter  同池等权，不排除任何股票，持仓 k 只
  eq_shell     同池等权，排除 20 日均成交额最小的 30%，持仓 k 只
  ep_nofilter  EP 最高 k 只，不排除
  ep_shell     EP 最高 k 只，先排除最小 30%
  ep_rev       反向对照：EP 最低 k 只
  roe          ROE 最高 k 只
"""
import os, json, struct, mmap, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DISCLOSURE_LAG = {3: 30, 6: 62, 9: 31, 12: 120}
EXCL = 0.30


def load_fund():
    raw = json.load(open(os.path.join(HERE, "fundamentals.json"), encoding="utf-8"))
    out = {}
    for code, rows in raw.items():
        if not isinstance(rows, list): continue
        ser = sorted([(r["d"], r["eps"], r.get("bps"), r.get("roe")) for r in rows
                      if r.get("d") and r.get("eps") is not None])
        if not ser: continue
        fixed = []
        for i, (d, eps, bps, roe) in enumerate(ser):
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
            fixed.append((d, ttm, bps, roe))
        out[code] = fixed
    return out


def avail(d):
    import datetime
    y, m, dd = int(d[:4]), int(d[5:7]), int(d[8:10])
    return (datetime.date(y, m, dd) + datetime.timedelta(days=DISCLOSURE_LAG.get(m, 90))).isoformat()


def main(which="ep_shell", k=50, cash=1_000_000.0, start="2010-06-01"):
    fh_c = open(os.path.join(HERE, "m_close.bin"), "rb")
    fh_o = open(os.path.join(HERE, "m_open.bin"), "rb")
    fh_a = open(os.path.join(HERE, "amt_flat.bin"), "rb")
    fh_m = open(os.path.join(HERE, "m_mktcap.bin"), "rb")
    mm_c = mmap.mmap(fh_c.fileno(), 0, access=mmap.ACCESS_READ)
    mm_o = mmap.mmap(fh_o.fileno(), 0, access=mmap.ACCESS_READ)
    mm_a = mmap.mmap(fh_a.fileno(), 0, access=mmap.ACCESS_READ)
    mm_m = mmap.mmap(fh_m.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm_c, 0)
    def C(i): return struct.unpack_from("<d", mm_c, 16 + 8 * i)[0]
    def O(i): return struct.unpack_from("<d", mm_o, 16 + 8 * i)[0]
    def A(i): return struct.unpack_from("<d", mm_a, 16 + 8 * i)[0]
    def M(i): return struct.unpack_from("<d", mm_m, 16 + 8 * i)[0]

    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates = meta["dates"]; codes = meta["codes"]
    assert len(dates) == ND and len(codes) == NC, "meta 与矩阵不一致"
    fs = load_fund()
    prep = {}
    for code, rows in fs.items():
        prep[code] = [(avail(r[0]), r[1], r[2], r[3]) for r in rows]
    ci_of = {c: i for i, c in enumerate(codes)}
    # 每只股票"最后一个有价日"。持仓到这一步就不再按 lastc 兜底，
    # 而是按退市归零处理 —— 否则退市股会用最后一天的价格无限期存活，
    # 这是抬高收益的偏差（实测池内 246 只在回测期间消失）。
    last_j = {}
    for ci in range(NC):
        base = ci * ND
        for j2 in range(ND - 1, -1, -1):
            if C(base + j2) > 0:
                last_j[ci] = j2
                break
    prep = {c: v for c, v in prep.items() if c in ci_of}
    print("覆盖 %d/%d 只，矩阵 %d 天" % (len(prep), NC, ND), flush=True)

    si = next(i for i, d in enumerate(dates) if d >= start)
    pos = {}; cashleft = cash; lastc = {}; curve = []; fees = 0.0
    fills = 0; prev_ym = None; t0 = time.time()
    for j in range(si, ND):
        d = dates[j]
        mv = 0.0
        for ci, sh in list(pos.items()):
            px = C(ci * ND + j)
            if px > 0:
                lastc[ci] = px
            elif j > last_j.get(ci, j):
                # 已过最后一个有价日：退市，市值记 0，不产生现金。
                # 注意不能 del pos[ci] —— 那等于凭最后价把它卖掉了，会凭空多出现金。
                continue
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
            if found >= 0: snap[code] = rows[found]
        cand = []
        for code, (av, ttm, bps, roe) in snap.items():
            ci = ci_of[code]; base = ci * ND
            px = C(base + j)
            if px <= 2.0 or O(base + jn) <= 0: continue
            if not ttm: continue
            # 规模口径：真实总市值（收盘价 × 当时总股本），不再是成交额代理
            cap = M(base + j)
            if cap <= 0: continue
            adv = 0.0; cnt = 0
            for kk in range(max(0, j - 19), j + 1):
                a = A(base + kk)
                if a > 0: adv += a; cnt += 1
            cand.append((code, ttm / px, roe, cap))
        if not cand: continue
        if not which.startswith("eq_nofilter") and which != "ep_nofilter":
            cand.sort(key=lambda x: x[3])
            cand = cand[int(len(cand) * EXCL):]
        if not cand: continue
        if which.startswith("ep_shell") or which == "ep_nofilter":
            cand.sort(key=lambda x: -x[1])
        elif which == "ep_rev":
            cand.sort(key=lambda x: x[1])
        elif which == "roe":
            cand.sort(key=lambda x: -(x[2] or -99))
        live = [c[0] for c in cand[:k]]
        if not live: continue
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
            cashleft -= amt + fee; fees += fee; pos[ci] = pos.get(ci, 0) + qq; fills += 1
        if False:
            print("  %s 净值 %.0f 持仓 %d %.0fs" % (d, equity, len(pos), time.time() - t0), flush=True)

    json.dump({"curve": curve, "fees": round(fees), "fills": fills, "which": which, "k": k},
              open(os.path.join(HERE, "r7_%s.json" % which), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：倍数 %.2f 费用 %.0f 成交 %d %.0fs" % (
        which, curve[-1][1] / curve[0][1], fees, fills, time.time() - t0))


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "ep_shell"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    main(w, k=k)
