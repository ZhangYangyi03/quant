"""13 个异象横向比较 —— 全向量化版。

上一版慢在：每天对持仓做 Python 循环，每期对 5251 只股票做二分查找。
这一版：价格矩阵一次性加载，截面用 numpy 向量化，账面净值用矩阵乘。
189 期 x 4000 交易日，目标 60 秒内跑完一个信号。

13 个信号出自 xxv-byte/size-and-value-in-china-replication 的 anomaly_summary.md。
本机能算 11 个（cp / asset_growth / accruals / noa 需要资产负债表，我的
fundamentals.json 只有 eps/bps/roe，算不了 —— 这个缺口明确记下）。
"""
import os, json, struct, mmap, time, datetime, sys, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAG = {3: 30, 6: 62, 9: 31, 12: 120}


def load_panel(name):
    f = open(os.path.join(HERE, name), "rb")
    _, ND, NC = struct.unpack_from("<8sII", f.read(16))
    return np.frombuffer(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ),
                         dtype=np.float64, offset=16, count=ND * NC).reshape(NC, ND), ND, NC


def fund_matrix(codes, ci_of, dates, field="eps"):
    """把某一列财务数据摊成 (NC, ND) float32 矩阵，按公告滞后日期对齐。

    性能要点（第一版在这里卡了 4 分钟以上）：
      1. 前向填充不用逐列 Python 循环，用 np.maximum.accumulate 走索引，
         一次算完 8816 列。
      2. TTM 还原用字典索引上一年的年报和同季报，不再每期往回线性搜索。

    field: "eps" 走 TTM 还原，"bps"/"roe" 直接取。
    """
    t0 = time.time()
    raw = json.load(open(os.path.join(HERE, "fundamentals.json"), encoding="utf-8"))
    NC, ND = len(codes), len(dates)
    S = np.full((NC, ND), np.nan, dtype=np.float32)
    di = {d: i for i, d in enumerate(dates)}
    LAGD = {m: datetime.timedelta(days=LAG.get(m, 90)) for m in (3, 6, 9, 12)}
    # 预计算所有出现过的披露日期 + 其滞后落到哪个交易日索引。一次算完，之后是字典查表。
    IDX_CACHE = {}
    _all_d = set()
    for _rows in raw.values():
        if isinstance(_rows, list):
            for _r in _rows:
                if _r.get("d"): _all_d.add((_r["d"], int(_r["d"][5:7])))
    for _d, _m in _all_d:
        _ad = (datetime.date(int(_d[:4]), _m, int(_d[8:10])) + LAGD[_m]).isoformat()
        _k = di.get(_ad)
        if _k is None:
            _k = int(np.searchsorted(dates, _ad))
            if _k >= ND: _k = None
        if _k is not None: IDX_CACHE[(_d, _m)] = _k

    # 关键性能点：公告滞后日期几乎从不落在交易日上（4800 次里 4799 次不命中），
    # 每期都走 searchsorted 会让 5251 只跑 200 秒以上。
    # 改成：把所有可能出现的 (日期, 滞后) 组合预计算一次到交易日索引的映射。
    def to_idx(d):
        k = di.get(d)
        if k is not None: return k
        k = int(np.searchsorted(dates, d))
        return k if k < ND else None

    for code, rows in raw.items():
        ci = ci_of.get(code)
        if ci is None or not isinstance(rows, list): continue
        by = {}
        for r in rows:
            d = r.get("d")
            if not d: continue
            by[(int(d[:4]), int(d[5:7]))] = r
        for (y, m), r in by.items():
            if field == "eps":
                eps = r.get("eps")
                if eps is None: continue
                if m == 12:
                    val = eps
                else:
                    pr = by.get((y - 1, 12)); ps = by.get((y - 1, m))  # 上年同期累计
                    pf = pr.get("eps") if pr else None
                    pss = ps.get("eps") if ps else None
                    val = eps + pf - pss if (pf is not None and pss is not None) else eps
            elif field == "bps":
                val = r.get("bps")
                if val is None: continue
            else:
                val = r.get("roe")
                if val is None: continue
            k = IDX_CACHE.get((r["d"], m))
            if k is None: continue
            S[ci, k] = val

    # 前向填充。分块 + 原地写回，块内也只保留一个 int32 索引数组。
    # 一次持有一整块 float32(177MB) + int32 是不行的，所以块要切得足够小。
    CH = 256
    base_idx = np.arange(ND, dtype=np.int32)[None, :]
    for r0 in range(0, NC, CH):
        r1 = min(NC, r0 + CH)
        blk = S[r0:r1]
        hasnan = np.isnan(blk)
        idx = np.where(hasnan, 0, base_idx)
        np.maximum.accumulate(idx, axis=1, out=idx)
        # 该行从未有过有效值的位置保持 NaN（不能把第 0 列的值向前泄漏）
        first_valid = np.where(~hasnan.any(axis=1), ND, np.argmax(~hasnan, axis=1))
        filled = blk[np.arange(r1 - r0)[:, None], idx]
        cols = base_idx[0][None, :]
        filled = np.where(cols < first_valid[:, None], np.nan, filled)
        S[r0:r1] = filled
        del idx, filled, hasnan
    M = S
    print("    fund_matrix(%s) %.1fs" % (field, time.time() - t0), flush=True)
    return M



def segments(curve, dates, si, n=4):
    """把净值曲线切成 n 等段，各段 CAGR。用来区分"全程有效"和"靠某一段暴赚"。"""
    L = len(curve)
    out = []
    for i in range(n):
        a, b = i * L // n, (i + 1) * L // n - 1
        if b <= a: continue
        d0 = datetime.date(*map(int, dates[si + a].split("-")))
        d1 = datetime.date(*map(int, dates[si + b].split("-")))
        yrs = max((d1 - d0).days / 365.25, 1 / 365)
        c = (curve[b] / curve[a]) ** (1 / yrs) - 1 if curve[a] > 0 else -1
        out.append(round(c * 100, 1))
    return out

def backtest(score_fn, C, O, A, MK, dates, live_mask, k=50, cash=1e6, si=0):
    """逐日回测。交易日只更新净值（一次矩阵广播），交易只在调仓日发生。

    上一版在每一天都跑一遍卖出循环（4000 天 x 50 只），这是它跑不完的原因。
    """
    ND = C.shape[1]
    NC = C.shape[0]
    pos = np.zeros(NC, dtype=np.int64)
    cashleft = cash
    lastc = np.zeros(NC)
    fees = 0.0; fills = 0
    # 调仓日：每月第一个交易日
    rebal = [j for j in range(si + 1, ND) if dates[j][:7] != dates[j - 1][:7] or j == si + 1]
    curve = np.zeros(ND - si)
    prev_ym = None
    for j in range(si, ND):
        col = C[:, j]
        upd = (col > 0) & (pos > 0)
        if upd.any(): lastc[upd] = col[upd]
        equity = cashleft + float(np.dot(pos, lastc))
        curve[j - si] = equity
        if (j - si) % 500 == 0:
            print("    .. %s 净值 %.0f" % (dates[j], equity), flush=True)
        if dates[j][:7] == prev_ym: continue
        prev_ym = dates[j][:7]
        jn = j + 1
        if jn >= ND: break
        sc = score_fn(j)
        valid = np.isfinite(sc) & live_mask[:, j]
        if not valid.any(): continue
        idx = np.where(valid)[0]
        px_next = O[idx, jn]
        ok = px_next > 0
        idx = idx[ok]
        if len(idx) == 0: continue
        order = idx[np.argsort(-sc[idx])][:k]
        keep = set(order.tolist())
        per = equity / max(len(order), 1)

        # 卖：只遍历实际持仓
        held = np.where(pos > 0)[0]
        for ci in held:
            if ci in keep: continue
            px = O[ci, jn]
            if px <= 0: px = lastc[ci]
            if px <= 0: continue
            sh = int(pos[ci]); amt = sh * px
            fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
            cap = A[ci, jn] * 0.10
            if cap > 0 and amt > cap:
                sellsh = int(cap / px); amt = sellsh * px
                fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                cashleft += amt - fee; fees += fee; fills += 1
                pos[ci] = sh - sellsh
                continue
            cashleft += amt - fee; fees += fee; fills += 1; pos[ci] = 0
        # 买
        for ci in order:
            px = O[ci, jn]
            if px <= 0: continue
            have = pos[ci] * px
            if have >= per * 0.9: continue
            cap = A[ci, jn] * 0.10
            want = min(per - have, cap) if cap > 0 else per - have
            qq = int(want / px / 100) * 100
            if qq <= 0: continue
            amt = qq * px; fee = max(5.0, amt * 0.00025) + amt * 0.0001
            if amt + fee > cashleft: continue
            cashleft -= amt + fee; pos[ci] += qq; fills += 1
    return curve, fees, fills


def perf(curve, dates, si):
    v0, v1 = curve[0], curve[-1]
    d0 = datetime.date(*map(int, dates[si].split("-")))
    d1 = datetime.date(*map(int, dates[-1].split("-")))
    yrs = (d1 - d0).days / 365.25
    cagr = (v1 / v0) ** (1 / yrs) - 1 if v1 > 0 else -1.0
    r = curve[1:] / curve[:-1] - 1
    r = r[np.isfinite(r)]
    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = mu / sd * (252 ** 0.5) if sd > 0 else 0.0
    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1))
    return dict(cagr=round(cagr * 100, 2), sharpe=round(float(sharpe), 3),
                mdd=round(mdd * 100, 1), total=round(v1 / v0, 2), years=round(yrs, 1))


def main(only=None, k=50, start="2010-06-01"):
    t_all = time.time()
    C, ND, NC = load_panel("m_close.bin")
    O, _, _ = load_panel("m_open.bin")
    A, _, _ = load_panel("amt_flat.bin")
    MK, _, _ = load_panel("m_mktcap.bin")
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ci_of = {c: i for i, c in enumerate(codes)}
    print("[%5.1fs] 面板加载完 %dx%d" % (time.time() - t_all, NC, ND), flush=True)


    si = next(i for i, d in enumerate(dates) if d >= start)
    live_mask = (C > 2.0) & (A > 0)

    # 每个信号现算一个面板、跑完立刻释放。同一时刻只占一份 (5251,8816) 内存。
    def ret_slice(j0, j1):
        """返回 [j0, j1) 区间的收益率切片，只算这一段，不建全历史矩阵。
        全历史 float32 要 353MB，沙箱里做不到 —— 每个信号只看它需要的窗口。"""
        prev = C[:, j0:j1 - 1]
        cur = C[:, j0 + 1:j1]
        good = prev > 0
        out = np.full((C.shape[0], j1 - j0 - 1), np.nan, dtype=np.float32)
        np.divide(cur, np.where(good, prev, np.float32(1)), out=out, where=good)
        out[~good] = np.nan
        out -= 1
        return out

    def panel_vol(j): return np.nanstd(ret_slice(max(0, j - 20), j + 1), axis=1)
    def panel_max(j): return np.nanmax(ret_slice(max(0, j - 20), j + 1), axis=1)
    def panel_ret20(j): return np.where(C[:, j - 21] > 0, C[:, j] / np.where(C[:, j - 21] > 0, C[:, j - 21], np.nan) - 1, np.nan) if j >= 21 else np.full(C.shape[0], np.nan)
    def panel_ret250(j): return np.where(C[:, j - 251] > 0, C[:, j] / np.where(C[:, j - 251] > 0, C[:, j - 251], np.nan) - 1, np.nan) if j >= 251 else np.full(C.shape[0], np.nan)
    def panel_turn(j, n):
        j0 = max(0, j - n)
        m = MK[:, j0:j + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = A[:, j0:j + 1] / np.where(m > 0, m, np.nan)
        return np.nanmean(t, axis=1)

    FIELD = {"ep": "eps", "bm": "bps", "roe": "roe"}

    def build(name):
        """返回该信号的分数函数；闭包里只留它自己需要的面板。"""
        if name == "size":
            return lambda j: -np.log(np.maximum(MK[:, j], 1))
        if name in ("ep", "bm", "roe"):
            M = fund_matrix(codes, ci_of, dates, field=FIELD[name])
            if name == "ep":  return lambda j: M[:, j] / C[:, j]
            if name == "bm":  return lambda j: M[:, j] / C[:, j]
            return lambda j: M[:, j]
        if name == "volatility":
            return lambda j: -panel_vol(j)
        if name == "max":
            return lambda j: -panel_max(j)
        if name == "reversal":
            return lambda j: -panel_ret20(j)
        if name == "momentum":
            return lambda j: panel_ret250(j)
        if name == "turnover":
            return lambda j: -panel_turn(j, 20)
        if name == "abnormal_turnover":
            def _ab(j):
                t20 = panel_turn(j, 20); t250 = panel_turn(j, 250)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return -np.where(t250 > 0, t20 / np.where(t250 > 0, t250, np.nan), np.nan)
            return _ab
        if name == "illiquidity":
            def _illiq(j):
                j0 = max(0, j - 19)
                r = ret_slice(j0, j + 1)                       # 列对应交易日 j0+1 .. j
                amt = A[:, j0 + 1:j + 1]                        # 对齐到同一批日期
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.nanmean(np.abs(r) / np.maximum(amt / 1e8, 1e-9), axis=1)
            return _illiq
        raise KeyError(name)

    ALL = ["size", "ep", "bm", "roe", "volatility", "max", "turnover",
           "abnormal_turnover", "illiquidity", "reversal", "momentum"]
    todo = [only] if only else ALL
    results = {}
    for name in todo:
        t0 = time.time()
        fn = build(name)
        curve, fees, fills = backtest(fn, C, O, A, MK, dates, live_mask, k=k, si=si)
        p = perf(curve, dates, si)
        p.update(fills=fills, fees=round(fees), k=k,
                 seg=segments(curve, dates, si))
        results[name] = p
        print("[%5.1fs] %-20s CAGR %6.2f%%  Sharpe %5.3f  MDD %6.1f%%  净值 %5.2fx  分段 %s (%.1fs)" % (
            time.time() - t_all, name, p["cagr"], p["sharpe"], p["mdd"], p["total"],
            p["seg"], time.time() - t0), flush=True)
        np.save(os.path.join(HERE, "anomc_%s.npy" % name), curve)
        del fn
        import gc; gc.collect()

    old = {}
    fp = os.path.join(HERE, "anomaly_race.json")
    if only and os.path.exists(fp):
        old = json.load(open(fp, encoding="utf-8"))
    old.update(results)
    json.dump(old, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not only:
        print("\n按 Sharpe 排名：", flush=True)
        for i, (n, r) in enumerate(sorted(results.items(), key=lambda x: -x[1]["sharpe"]), 1):
            print("%2d. %-20s Sharpe %5.3f  CAGR %6.2f%%  MDD %6.1f%%  净值 %5.2fx" % (
                i, n, r["sharpe"], r["cagr"], r["mdd"], r["total"]), flush=True)


if __name__ == "__main__":
    main(only=(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else None),
         k=int(sys.argv[2]) if len(sys.argv) > 2 else 50)
