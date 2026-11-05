"""逐年独立资金的 EP 策略分年收益（不是参数 walk-forward）。

2026-11-05 修订说明（见 INSTITUTIONAL.md）：
  原 docstring 声称「每年按过去 3 年夏普在 A/B/C 三个配置中选一个」，
  但代码从未实现该逻辑：Sim 类未使用、ind 行业映射传入后未使用。
  更严重的是 build() 在 cand[:50] 之前缺少按 EP 排序，实际取到的是
  fundamentals.json 键序里代码最小的 50 只老股 —— 逐年成绩测的不是 EP 策略。
  本次已补上排序；walkforward.json 与 README 中基于旧代码的历史数字作废，需重跑。

  真正的参数型 walk-forward（只用历史信息选参、样本外结算、可注入反例检测）
  请见 quantlab/validate/walkforward.py，附带单元测试。

本文件实际做的事：把日期按年分组，每年用一套新的 100 万资金、固定 k=50 的
EP 策略独立跑一遍，用来观察逐年体验的离散度。它不能证明样本外可预测，
只能说明单年波动有多大。
"""
import os, json, struct, mmap, math, sys
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


class Sim:
    def __init__(self, k, mode, cash=1_000_000.0):
        self.k = k; self.mode = mode; self.cash = cash
        self.pos = {}; self.cashleft = cash; self.lastc = {}; self.fees = 0.0
        self.equity = cash; self.start_equity = cash

    def rebalance(self, j, jn, d, live):
        per = self.equity / len(live); keep = set(live)
        C, O, A = self.C, self.O, self.A
        for ci in list(self.pos):
            if self.codes[ci] in keep: continue
            sh = self.pos.pop(ci); px = O(ci * self.ND + jn)
            if px <= 0: px = self.lastc.get(ci, 0.0)
            if px <= 0: continue
            amt = sh * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
            cap = A(ci * self.ND + jn) * 0.10
            if cap > 0 and amt > cap:
                s2 = int(cap / px); amt = s2 * px
                fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                self.cashleft += amt - fee; self.fees += fee
                if sh > s2: self.pos[ci] = sh - s2
                continue
            self.cashleft += amt - fee; self.fees += fee
        for code in keep:
            ci = self.ci_of[code]; px = O(ci * self.ND + jn)
            if px <= 0: continue
            have = self.pos.get(ci, 0) * px
            if have >= per * 0.9: continue
            want = min(per - have, A(ci * self.ND + jn) * 0.10 or per - have)
            qq = int(want / px / 100) * 100
            if qq <= 0: continue
            amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
            if amt + fee > self.cashleft: continue
            self.cashleft -= amt + fee; self.fees += fee
            self.pos[ci] = self.pos.get(ci, 0) + qq


def build(dates_sub, fund, meta, ind, last_j, C, O, ND, NC, codes, ci_of):
    """在给定日期范围上跑一个策略，返回每日净值。"""
    si = dates_sub[0]
    pos = {}; cashleft = 1_000_000.0; lastc = {}; curve = []; prev_ym = None
    for j in dates_sub:
        d = meta["dates"][j]
        mv = 0.0
        for ci, sh in list(pos.items()):
            px = C(ci * ND + j)
            if px > 0: lastc[ci] = px
            elif j > last_j.get(ci, j): continue
            mv += sh * lastc.get(ci, 0.0)
        equity = cashleft + mv
        curve.append((d, equity))
        if d[:7] == prev_ym: continue
        prev_ym = d[:7]
        jn = j + 1
        if jn >= ND: break
        snap = {}
        for code, rows in fund.items():
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
            if px <= 2.0 or O(base + jn) <= 0 or not ttm: continue
            cand.append((code, ttm / px, ind.get(code)))
        if not cand: continue
        # 2026-11-05 修复：必须按 EP 降序选股。此前缺失这一行，cand[:50]
        # 取的是 fundamentals.json 键序（股票代码升序）的前 50 只，
        # 导致 walkforward.json 的 17 个年度数字测的不是 EP 策略。
        cand.sort(key=lambda x: -x[1])
        live = [c[0] for c in cand[:50]]
        per = equity / len(live); keep = set(live)
        for ci in list(pos):
            if codes[ci] in keep: continue
            sh = pos.pop(ci); px = O(ci * ND + jn)
            if px <= 0: px = lastc.get(ci, 0.0)
            if px <= 0: continue
            amt = sh * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
            cashleft += amt - fee
        for code in keep:
            ci = ci_of[code]; px = O(ci * ND + jn)
            if px <= 0: continue
            have = pos.get(ci, 0) * px
            if have >= per * 0.9: continue
            qq = int((per - have) / px / 100) * 100
            if qq <= 0: continue
            amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
            if amt + fee > cashleft: continue
            cashleft -= amt + fee; pos[ci] = pos.get(ci, 0) + qq
    return curve


def cagr(curve):
    if len(curve) < 30: return 0.0
    v = [x[1] for x in curve]
    n = len(v)
    return ((v[-1] / v[0]) ** (250 / n) - 1) * 100


def main():
    fh_c = open(os.path.join(HERE, "m_close.bin"), "rb")
    fh_o = open(os.path.join(HERE, "m_open.bin"), "rb")
    mm_c = mmap.mmap(fh_c.fileno(), 0, access=mmap.ACCESS_READ)
    mm_o = mmap.mmap(fh_o.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm_c, 0)
    def C(i): return struct.unpack_from("<d", mm_c, 16 + 8 * i)[0]
    def O(i): return struct.unpack_from("<d", mm_o, 16 + 8 * i)[0]
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates = meta["dates"]; codes = meta["codes"]; ci_of = {c: i for i, c in enumerate(codes)}
    ind = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    fund = {c: [(avail(r[0]), r[1]) for r in rows]
            for c, rows in load_fund().items() if c in ci_of}
    last_j = {}
    for ci in range(NC):
        base = ci * ND
        for j2 in range(ND - 1, -1, -1):
            if C(base + j2) > 0: last_j[ci] = j2; break

    si = next(i for i, d in enumerate(dates) if d >= "2010-06-01")
    # 逐年的日期索引
    years = {}
    for j in range(si, ND):
        years.setdefault(dates[j][:4], []).append(j)
    ys = sorted(years)
    print("年度 %d 个：%s ~ %s" % (len(ys), ys[0], ys[-1]), flush=True)

    # 每年用"当年日期范围"独立跑一次纯 EP，得到每年的成绩（这就是前推的基础）
    yearly = []
    for y in ys:
        ds = years[y]
        cv = build(ds, fund, meta, ind, last_j, C, O, ND, NC, codes, ci_of)
        yearly.append((y, cagr(cv), len(cv)))
        print("  %s  %7.2f%%  (%d 天)" % (y, yearly[-1][1], len(cv)), flush=True)

    print()
    print("逐年成绩（这就是 walk-forward 唯一能看的东西：每年都是新的开始）")
    pos_year = sum(1 for _, c, _ in yearly if c > 0)
    print("  %d/%d 年为正" % (pos_year, len(yearly)))
    print("  中位年度 %.2f%%  最差 %.2f%%  最好 %.2f%%" % (
        sorted(c for _, c, _ in yearly)[len(yearly) // 2],
        min(c for _, c, _ in yearly), max(c for _, c, _ in yearly)))
    json.dump([{"year": y, "cagr": round(c, 2), "days": n} for y, c, n in yearly],
              open(os.path.join(HERE, "walkforward.json"), "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
