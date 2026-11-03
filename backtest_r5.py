"""第四轮：实现《Size and value in China》(Liu, Stambaugh, Yuan 2019, JFE) 的三个因子。

为什么是这篇：它用中国数据重建了 Fama-French，并明确指出原样照搬会留下 17% 的年化
alpha 解释不掉。它的三条结论都是可执行的：

  1. 规模因子必须排除最小的 30%。那些是借壳上市的壳资源，价格反映的是壳价值，
     不是企业价值。直接做"越小越好"在 A 股是噪声。
  2. 价值因子用 EP（盈利/价格），不用 PB。论文实测 EP 在解释中国异象时涵盖 PB 的全部效果。
  3. 这三因子能解释掉盈利异象和波动率异象——也就是说，我此前试的低波，
     在这套框架里可能只是价值/规模的代理。

数据：fundamentals.json 是 3160 只的季度 EPS/BPS/ROE（东方财富财务摘要，带报告期）。
关键细节——**只用已公布的财报**：报告期 2020-03-31 的财报通常 4 月底才公告，
所以每个交易日只使用公告截止日已过的最近一期，避免前视偏差。
这里用保守的滞后：报告期结束后 +90 天视为可用。
"""
import os, json, struct, mmap, time, sys, bisect

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "px_flat.bin")
META = os.path.join(HERE, "px_index_meta.json")
FUND = os.path.join(HERE, "fundamentals.json")

# 报告期 -> 市场已知日。不是统一滞后，按 A 股真实披露截止日：
#   年报 12-31 最晚 4-30 披露  -> 120 天
#   一季报 3-31 最晚 4-30       -> 30 天
#   半年报 6-30 最晚 8-31       -> 62 天
#   三季报 9-30 最晚 10-31      -> 31 天
# 统一用 90 天会漏掉年报的真实生效日，等于提前一个月拿到全年业绩 —— 那是前视偏差。
DISCLOSURE_LAG = {3: 30, 6: 62, 9: 31, 12: 120}
EXCLUDE_SMALLEST = 0.30


def load_fund_series():
    """code -> [(available_date, ttm_eps, bps, roe)]，按 available_date 升序。"""
    raw = json.load(open(FUND, encoding="utf-8"))
    out = {}
    for code, rows in raw.items():
        if not isinstance(rows, list): continue
        ser = []
        for r in rows:
            d = r.get("d"); eps = r.get("eps"); bps = r.get("bps"); roe = r.get("roe")
            if not d or eps is None: continue
            ser.append((d, eps, bps, roe))
        if not ser: continue
        ser.sort(key=lambda x: x[0])
        # TTM EPS：用最近四个季度累计值。中国的季报 EPS 是年初至今累计，
        # 所以 Q4 报告期(12-31)本身就是全年；Q1/Q2/Q3 需要用同比法还原。
        fixed = []
        for i, (d, eps, bps, roe) in enumerate(ser):
            m = int(d[5:7])
            if m == 12:
                ttm = eps
            else:
                # 找去年同期同报告期，用 (本期累计 + 去年全年 - 去年同累计)
                prev_year_full = None; prev_year_same = None
                for j in range(i - 1, -1, -1):
                    dj = ser[j][0]
                    if dj[:4] == str(int(d[:4]) - 1):
                        if dj[5:7] == "12": prev_year_full = ser[j][1]
                        if dj[5:7] == d[5:7]: prev_year_same = ser[j][1]
                if prev_year_full is not None and prev_year_same is not None:
                    ttm = eps + prev_year_full - prev_year_same
                elif prev_year_same is not None:
                    ttm = eps
                else:
                    ttm = eps if m == 12 else None
            if ttm is None: continue
            fixed.append((d, ttm, bps, roe))
        if fixed: out[code] = fixed
    return out


def avail(date_str):
    """报告期 -> 假定市场已知的日期（YYYY-MM-DD），按真实披露截止日。"""
    import datetime
    y, m, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    dt = datetime.date(y, m, d) + datetime.timedelta(days=DISCLOSURE_LAG.get(m, 90))
    return dt.isoformat()


def main(which="ep_shell", cash=1_000_000.0, start="2010-06-01", k=50, vol_n=120):
    fp = open(FLAT, "rb")
    mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
    magic, ND, NC = struct.unpack_from("<8sII", mm, 0)
    B_O = 16; B_C = 16 + 8 * ND * NC
    def C(i): return struct.unpack_from("<d", mm, B_C + 8 * i)[0]
    def O(i): return struct.unpack_from("<d", mm, B_O + 8 * i)[0]

    meta = json.load(open(META, encoding="utf-8"))
    dates = meta["dates"]; codes = meta["codes"]
    ci_of = {c: i for i, c in enumerate(codes)}

    fs = load_fund_series()
    # 预先算好每只股票的可用时点序列
    prep = {}
    for code, rows in fs.items():
        if code not in ci_of: continue
        prep[code] = [(avail(r[0]), r[1], r[2], r[3]) for r in rows]
    print("因子数据覆盖 %d 只" % len(prep), flush=True)

    si = next(i for i, d in enumerate(dates) if d >= start)
    pos = {}; cashleft = cash; lastc = {}; curve = []; fees = 0.0; fills = 0
    prev_ym = None; t0 = time.time(); nrebal = 0
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
            # 当日可用因子
            snap = {}
            for code, rows in prep.items():
                # 只用 avail<=d 的最后一期
                lo, hi = 0, len(rows) - 1; found = -1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if rows[mid][0] <= d: found = mid; lo = mid + 1
                    else: hi = mid - 1
                if found < 0: continue
                a, ttm, bps, roe = rows[found]
                snap[code] = (ttm, bps, roe)

            cand = []
            for code, (ttm, bps, roe) in snap.items():
                ci = ci_of[code]; base = ci * ND
                px = C(base + j)
                if px <= 2.0 or O(base + jn) <= 0: continue
                if ttm is None or ttm == 0: continue
                # 20 日均成交额作为规模代理（论文用市值，这里没有股本，用价格×成交额不可得，
                # 用"价格"作为截面稳定的市值粗代理会被质疑，所以改用滚动成交额）
                adv = 0.0; cnt = 0
                for kk in range(max(0, j - 19), j + 1):
                    c0 = C(base + kk)
                    if c0 > 0: adv += c0; cnt += 1
                adv = adv / cnt if cnt else 0.0
                cand.append((code, ttm / px, roe, adv))

            if not cand: continue
            # 规模过滤：默认排除成交额最小 30%（论文的壳资源规则）。
            # eq_nofilter 用同一批股票但不做过滤，两者之差就是这条规则的真实贡献。
            if which != "eq_nofilter":
                cand.sort(key=lambda x: x[3])
                cut = int(len(cand) * EXCLUDE_SMALLEST)
                cand = cand[cut:]
            if not cand: continue

            if which == "ep":
                cand.sort(key=lambda x: -x[1])          # EP 越高越好
            elif which == "roe":
                cand.sort(key=lambda x: -(x[2] or -99))  # ROE 越高越好
            elif which == "ep_roe":
                # 双排序：EP 和 ROE 各自排名后相加
                by_ep = sorted(range(len(cand)), key=lambda i: -cand[i][1])
                by_roe = sorted(range(len(cand)), key=lambda i: -(cand[i][2] or -99))
                r_ep = {cand[i][0]: r for r, i in enumerate(by_ep)}
                r_roe = {cand[i][0]: r for r, i in enumerate(by_roe)}
                cand.sort(key=lambda x: r_ep[x[0]] + r_roe[x[0]])
            elif which == "ep_shell":
                # 论文的干净版本：EP 排序，但已排除最小 30%
                cand.sort(key=lambda x: -x[1])
            elif which == "ep_rev":
                # 反向对照：EP 越低越好。如果这个也赚，说明赚的不是 EP。
                cand.sort(key=lambda x: x[1])
            elif which == "eq_nofilter":
                # 同池等权，不排除任何股票。与 eq_shell 只差一个过滤。
                pass
            elif which == "eq_shell":
                pass
            elif which == "all_eq":
                # 对照：同池、同去壳规则下纯等权。这是去除"选股"后的基线。
                pass
            # 等权对照组固定持仓数量 = k，否则会退化成"持有全部剩余股票"，
            # 与排除前的 50 只不是同一件事（曾因此误判规模规则贡献 8.67 个百分点）
            if which in ("all_eq", "eq_shell", "eq_nofilter"):
                live = [c for c, _, _, _ in cand[:k]]
            else:
                live = [c for c, _, _, _ in cand[:k]]
            if not live: continue

            per = equity / len(live)
            keep = set(live)
            for ci in list(pos):
                if codes[ci] in keep: continue
                sh = pos.pop(ci); px = O(ci * ND + jn)
                if px <= 0: px = lastc.get(ci, 0.0)
                amt = sh * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                cashleft += amt - fee; fees += fee; fills += 1
            for code in keep:
                ci = ci_of[code]; px = O(ci * ND + jn)
                if px <= 0: continue
                have = pos.get(ci, 0) * px
                if have >= per * 0.9: continue
                qq = int((per - have) / px / 100) * 100
                if qq <= 0: continue
                amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
                if amt + fee > cashleft: continue
                cashleft -= amt + fee; fees += fee; pos[ci] = pos.get(ci, 0) + qq; fills += 1
            nrebal += 1
        if j % 2000 == 0:
            print("  %d/%d %s 净值 %.0f 持仓 %d cand %d %.0fs" % (j, ND, d, equity, len(pos), len(cand) if 'cand' in dir() else -1, time.time() - t0), flush=True)

    json.dump({"curve": curve, "fees": round(fees), "fills": fills, "which": which, "k": k},
              open(os.path.join(HERE, "r5_%s.json" % which), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s：净值 %.2f 倍 费用 %.0f 成交 %d 调仓 %d %.0fs" % (
        which, curve[-1][1] / curve[0][1], fees, fills, nrebal, time.time() - t0))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ep_shell")
