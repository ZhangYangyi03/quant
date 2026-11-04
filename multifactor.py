"""IC 加权多因子合成（第 2、3 条做法）—— 不选因子，让数据每月重新分配权重。

为什么这样做
------------
上一节证明 8 个因子在时间外有持续性。但它们的 IC 会随时间漂移：
某年 illiquidity 最强，另一年 bm 最强。硬选一个就是赌。
业界做法（Grinold-Kahn，《主动投资组合管理》第 12 章）：
  把每个因子标准化成 z 分数，按 IC/ICIR 加权求和，得到综合打分。

本节严格用"当时能知道的":
  第 t 月的权重 = 截至 t-1 月为止，各因子 IC 的均值 / 标准差（ICIR）
  第 t 月的信号 = 各因子 z 分数 × 那个权重
  这样 t 月用的权重，在 t 月之前就已经算好了 —— 没有用到未来。

对照组
------
  equal          11 个因子等权（不挑、不调）
  icir_t12       最近 12 个月 ICIR 加权（时变）
  icir_all       全历史 ICIR 加权（会有前视，只作上界参考）
  单因子最优     bm（前一节 t 值第二）
"""
import os, json, struct, mmap, time, datetime, sys, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import anomaly_race as AR
import ic_walkforward as IW

FEE_BUY, FEE_SELL = 0.00025, 0.00075
NAMES = ["size", "ep", "bm", "volatility", "max", "turnover",
         "abnormal_turnover", "illiquidity"]          # 只用 |t|>2 的 8 个
FIELDS = {"ep": "eps", "bm": "bps"}


def zscore(x):
    m = np.isfinite(x)
    if m.sum() < 30: return np.full_like(x, np.nan, dtype=np.float64)
    v = x[m].astype(np.float64)
    med = np.median(v)
    # 用中位数/绝对偏差，抗极端值（A 股有大量失真价格）
    mad = np.median(np.abs(v - med)) * 1.4826
    if mad <= 0: return np.full_like(x, np.nan, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[m] = (v - med) / mad
    return np.clip(out, -3, 3)


def main(k=50, start="2010-06-01"):
    t0 = time.time()
    C, ND, NC = AR.load_panel("m_close.bin")
    O, _, _ = AR.load_panel("m_open.bin")
    A, _, _ = AR.load_panel("amt_flat.bin")
    MK, _, _ = AR.load_panel("m_mktcap.bin")
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ci_of = {c: i for i, c in enumerate(codes)}
    si = next(i for i, d in enumerate(dates) if d >= start)
    # 价格跳变清理：A 股单日涨跌停 10%（科创/创业 20%，早年无限制）。
    # 原始价未复权，除权除息会造成前后两天对不上的巨幅跳变
    # （实测单日 >100% 的有 2401 次，最大 174 倍）。这类点位既是假收益，
    # 也会让"低波动/低换手"因子误选到它。逐块扫描量一次，之后统一屏蔽。
    bad = np.zeros((NC, ND), dtype=bool)
    for r0 in range(0, NC, 500):
        blk = C[r0:r0 + 500]
        prev = blk[:, :-1]; cur = blk[:, 1:]
        good = (prev > 0) & (cur > 0)
        r = np.zeros(prev.shape, dtype=np.float32)
        np.divide(cur, np.where(good, prev, np.float32(1)), out=r, where=good)
        r -= 1
        jump = good & ((r > 0.21) | (r < -0.21))
        # 前后两天都跳，说明是真实趋势（如拆股后连续），只屏蔽孤立跳变
        iso = jump.copy()
        iso[:, 1:] &= ~jump[:, :-1]
        iso[:, :-1] &= ~jump[:, 1:]
        bad[r0:r0 + 500, 1:] = iso
        del r, jump, iso, good, prev, cur
    # 被跳变污染的那一天前后各一格都不参与选股
    badin = bad.copy()
    badin[:, 1:] |= bad[:, :-1]
    badin[:, :-1] |= bad[:, 1:]
    print("[%5.1fs] 价格跳变点 %d 个（占 %.3f%%，涉及 %d 只）" % (
        time.time() - t0, int(bad.sum()), 100 * bad.mean(), int(bad.any(axis=1).sum())), flush=True)
    # 新股排除：未上市时价格是 0.00，上市首日在 C>2 过滤下仍可能漏过
    # （实测 603444 吉比特 2017-01-10 收 0.71，次日 12.10，第三天 38.40 ——
    #  连续涨停 20 多个板。策略把它当"低换手小市值"买进来，17 倍就是这么来的，
    #  第二段 218% 几乎全是这个）。上市不满 250 个交易日的股票一律不碰。
    # 只在每个调仓日算一次"当时是否已上市满 250 日"，不留整张 bool 矩阵（46MB 也留不起）
    first_day = np.full(NC, ND, dtype=np.int32)
    for ci in range(NC):
        nz = np.where((C[ci] > 0) | (A[ci] > 0))[0]
        if len(nz): first_day[ci] = nz[0]
    print("[%5.1fs] 新股排除就绪（首日索引已算）" % (time.time() - t0), flush=True)
    MIND = 250

    def live_at(j):
        lst = first_day + MIND <= j
        return (C[:, j] > 2.0) & (A[:, j] > 0) & ~badin[:, j] & lst
    live = None

    rebal = [j for j in range(si + 1, ND) if dates[j][:7] != dates[j - 1][:7]]
    print("[%5.1fs] 调仓日 %d 个" % (time.time() - t0, len(rebal)), flush=True)

    # 每个调仓日的原始因子值 + 下期收益（一次算完，后面反复用）
    FV = {n: [] for n in NAMES}
    FWD = []
    FM = {}

    def ret_slice(j0, j1):
        prev = C[:, j0:j1 - 1]; cur = C[:, j0 + 1:j1]
        good = prev > 0
        out = np.full((C.shape[0], j1 - j0 - 1), np.nan, dtype=np.float32)
        np.divide(cur, np.where(good, prev, np.float32(1)), out=out, where=good)
        out[~good] = np.nan
        out -= 1
        return out

    cur_fld = {"f": None}

    def get(fld):
        if cur_fld["f"] != fld:
            FM.clear(); import gc; gc.collect()
            FM[fld] = AR.fund_matrix(codes, ci_of, dates, field=fld)
            cur_fld["f"] = fld
        return FM[fld]

    def factor_at(name, j):
        if name == "size":   return -np.log(np.maximum(MK[:, j], 1))
        if name == "ep":     return np.where(C[:, j] > 0, get("eps")[:, j] / np.where(C[:, j] > 0, C[:, j], np.nan), np.nan)
        if name == "bm":     return np.where(C[:, j] > 0, get("bps")[:, j] / np.where(C[:, j] > 0, C[:, j], np.nan), np.nan)
        if name == "volatility": return -np.nanstd(ret_slice(max(0, j - 20), j + 1), axis=1).astype(np.float64)
        if name == "max":        return -np.nanmax(ret_slice(max(0, j - 20), j + 1), axis=1).astype(np.float64)
        if name == "turnover":
            j0 = max(0, j - 20); m = MK[:, j0:j + 1]
            return -np.nanmean(A[:, j0:j + 1] / np.where(m > 0, m, np.nan), axis=1)
        if name == "abnormal_turnover":
            j0a = max(0, j - 20); ma = MK[:, j0a:j + 1]
            ta = np.nanmean(A[:, j0a:j + 1] / np.where(ma > 0, ma, np.nan), axis=1)
            j0b = max(0, j - 250); mb = MK[:, j0b:j + 1]
            tb = np.nanmean(A[:, j0b:j + 1] / np.where(mb > 0, mb, np.nan), axis=1)
            return -np.where(tb > 0, ta / np.where(tb > 0, tb, np.nan), np.nan)
        if name == "illiquidity":
            j0 = max(0, j - 19)
            r = ret_slice(j0, j + 1); amt = A[:, j0 + 1:j + 1]
            return np.nanmean(np.abs(r) / np.maximum(amt / 1e8, 1e-9), axis=1)
        raise KeyError(name)

    for n in NAMES:
        if n in FIELDS:
            cur_fld["f"] = None; get(FIELDS[n])
        for ri in range(len(rebal)):
            FV[n].append(zscore(factor_at(n, rebal[ri])).astype(np.float32))
        print("[%5.1fs]   %s 算完" % (time.time() - t0, n), flush=True)

    # 下期收益：调仓日 j 到下一个调仓日 j'，用 j' 的开盘买、j'' 的开盘卖
    for ri in range(len(rebal)):
        jn = rebal[ri] if ri < len(rebal) else ND - 1
        j2 = rebal[ri + 1] if ri + 1 < len(rebal) else ND - 1
        if j2 <= jn:
            FWD.append(np.full(NC, np.nan)); continue
        ok = (O[:, jn] > 0) & (O[:, j2] > 0)
        f = np.full(NC, np.nan)
        f[ok] = O[ok, j2] / O[ok, jn] - 1
        FWD.append(f)

    # 逐期 IC（用于时变权重）
    ICmat = np.full((len(NAMES), len(rebal)), np.nan)
    for ni, n in enumerate(NAMES):
        for ri in range(len(rebal)):
            j = rebal[ri]
            f = FV[n][ri]; y = FWD[ri]
            valid = live_at(j) & np.isfinite(y) & np.isfinite(f)
            if valid.sum() < 200: continue
            ICmat[ni, ri] = IW.spearman(np.where(valid, f, np.nan), np.where(valid, y, np.nan))
    print("[%5.1fs] IC 矩阵算完" % (time.time() - t0), flush=True)

    def weights_at(ri, mode, win=12):
        if mode == "equal":
            return np.ones(len(NAMES)) / len(NAMES)
        if mode == "icir_all":
            h = ICmat[:, :ri]
        else:
            h = ICmat[:, max(0, ri - win):ri]
        w = np.zeros(len(NAMES))
        for ni in range(len(NAMES)):
            v = h[ni][np.isfinite(h[ni])]
            if len(v) < 6: continue
            sd = v.std(ddof=1)
            if sd > 0: w[ni] = v.mean() / sd
        # ICIR 为负的因子权重清零（它最近在反向）
        w = np.where(w > 0, w, 0)
        s = w.sum()
        return w / s if s > 0 else np.ones(len(NAMES)) / len(NAMES)

    def run(mode, win=12, only=None, limit_check=False, impact=0.0):
        nav = 1.0e6; curve = []; AGE = []
        pos = set(); last = {}
        for ri in range(len(rebal)):
            j = rebal[ri]
            w = weights_at(ri, mode, win) if only is None else np.array([1.0 if n == only else 0.0 for n in NAMES])
            score = np.zeros(NC)
            cnt = np.zeros(NC)
            for ni, n in enumerate(NAMES):
                if w[ni] == 0: continue
                f = FV[n][ri]
                ok = np.isfinite(f)
                score[ok] += w[ni] * f[ok]
                cnt[ok] += w[ni]
            score = np.where(cnt > 0, score / np.maximum(cnt, 1e-12), np.nan)
            valid = live_at(j) & np.isfinite(score)
            if valid.sum() < k: curve.append(nav); continue
            idx = np.where(valid)[0]
            order = idx[np.argsort(-score[idx])][:k]
            if limit_check and j > 0:
                # 涨停买不进、跌停卖不出：当日相对前收涨跌幅超过 9.5% 的不可交易。
                # A 股主板 10%，科创/创业 20%，这里用 9.5% 统一近似主板，偏保守。
                pc = C[:, j - 1]
                chg = np.where(pc > 0, C[:, j] / np.where(pc > 0, pc, 1) - 1, 0)
                tradable = (chg < 0.095) & (chg > -0.095)
                order = np.array([c for c in order if tradable[c]], dtype=int)
                if len(order) < k // 2:      # 太多买不进，这期就少买
                    order = order
            newpos = set(order.tolist())
            if ri % 20 == 0 and len(newpos):
                age = np.mean([j - first_day[c] for c in newpos])
                AGE.append(float(age))
            fees = 0.0
            for c in pos - newpos:
                fees += nav / max(len(pos), 1) * FEE_SELL
            for c in newpos - pos:
                b = FEE_BUY
                if impact > 0:
                    # 冲击成本：单笔金额占当日成交额的比例，取其平方根（Almgren 简化式）
                    adv = A[c, j] if A[c, j] > 0 else 1e6
                    stake = nav / max(len(newpos), 1)
                    b += impact * math.sqrt(min(stake / adv, 0.3))
                fees += nav / max(len(newpos), 1) * b
            nav -= fees
            pos = newpos
            # 到下一期结算
            jn = rebal[ri + 1] if ri + 1 < len(rebal) else j
            if jn > j:
                r = np.array([C[c, jn] / C[c, j] - 1 if C[c, j] > 0 and C[c, jn] > 0 else 0.0 for c in pos])
                nav *= (1 + r.mean()) if len(r) else 1
            curve.append(nav)
        # 指标
        sv = np.array(curve)
        d0 = datetime.date(*map(int, dates[rebal[0]].split("-")))
        d1 = datetime.date(*map(int, dates[rebal[-1]].split("-")))
        yrs = max((d1 - d0).days / 365.25, 1/365)
        cagr = (sv[-1] / sv[0]) ** (1 / yrs) - 1
        dr = sv[1:] / sv[:-1] - 1
        sharpe = dr.mean() / dr.std(ddof=1) * math.sqrt(12) if dr.std() > 0 else 0
        peak = np.maximum.accumulate(sv); mdd = float((sv / peak - 1).min())
        L = len(sv); segs = []
        for i in range(4):
            a, b = i * L // 4, (i + 1) * L // 4 - 1
            if b > a: segs.append(round(((sv[b] / sv[a]) ** (12 / (b - a)) - 1) * 100, 1))
        # 逐年
        yr = {}
        for ri in range(1, len(rebal)):
            y = dates[rebal[ri]][:4]
            yr.setdefault(y, []).append(sv[ri] / sv[ri - 1] - 1)
        yearly = {y: round((float(np.prod([1 + x for x in v])) - 1) * 100, 1) for y, v in yr.items()}
        return {"cagr": round(cagr * 100, 2), "sharpe": round(float(sharpe), 3),
                "mdd": round(mdd * 100, 1), "total": round(float(sv[-1] / sv[0]), 2), "seg": segs,
                "yearly": yearly, "avg_age": round(float(np.mean(AGE)), 0) if AGE else None}

    def run_bench(mode="all"):
        """同口径基准：同一个股票池、同样剔除新股、同样月度调仓，但全买（等权）。
        这才能回答"因子到底有没有超额"，而不是"策略 vs 一个含打新的指数"。"""
        nav = 1.0e6; curve = []
        for ri in range(len(rebal)):
            j = rebal[ri]
            valid = live_at(j)
            idx = np.where(valid)[0]
            if len(idx) < 50: curve.append(nav); continue
            if mode == "top3000":
                cap = MK[idx, j]
                idx = idx[np.argsort(-np.nan_to_num(cap))[:3000]]
            jn = rebal[ri + 1] if ri + 1 < len(rebal) else j
            if jn > j:
                p0 = C[idx, j]; p1 = C[idx, jn]
                g = (p0 > 0) & (p1 > 0)
                r = p1[g] / p0[g] - 1 if g.sum() else np.array([0.0])
                nav *= (1 + r.mean())          # 与策略同样不截断，否则基准被低估
            curve.append(nav)
        sv = np.array(curve)
        d0 = datetime.date(*map(int, dates[rebal[0]].split("-")))
        d1 = datetime.date(*map(int, dates[rebal[-1]].split("-")))
        yrs = max((d1 - d0).days / 365.25, 1/365)
        cagr = (sv[-1] / sv[0]) ** (1 / yrs) - 1
        dr = sv[1:] / sv[:-1] - 1
        sharpe = dr.mean() / dr.std(ddof=1) * math.sqrt(12) if dr.std() > 0 else 0
        peak = np.maximum.accumulate(sv); mdd = float((sv / peak - 1).min())
        L = len(sv); segs = []
        for i in range(4):
            a, b = i * L // 4, (i + 1) * L // 4 - 1
            if b > a: segs.append(round(((sv[b] / sv[a]) ** (12 / (b - a)) - 1) * 100, 1))
        yr = {}
        for ri in range(1, len(rebal)):
            y = dates[rebal[ri]][:4]
            yr.setdefault(y, []).append(sv[ri] / sv[ri - 1] - 1)
        yearly = {y: round((float(np.prod([1 + x for x in v])) - 1) * 100, 1) for y, v in yr.items()}
        return {"cagr": round(cagr * 100, 2), "sharpe": round(float(sharpe), 3),
                "mdd": round(mdd * 100, 1), "total": round(float(sv[-1] / sv[0]), 2), "seg": segs,
                "yearly": yearly}

    out = {}
    print("\n%-14s %8s %8s %8s %8s  %s" % ("方案", "CAGR", "Sharpe", "回撤", "净值", "分段"), flush=True)
    print("-" * 78, flush=True)
    for label, mode, win, only in [
        ("等权8因子", "equal", 12, None),
        ("ICIR_t12", "icir_t12", 12, None),
        ("ICIR_t24", "icir_t24", 24, None),
        ("ICIR全历史", "icir_all", 12, None),
        ("单因子bm", "equal", 12, "bm"),
        ("单因子illiquidity", "equal", 12, "illiquidity"),
    ]:
        r = run(mode, win, only)
        out[label] = r
        print("%-14s %7.2f%% %8.3f %7.1f%% %7.2fx  上市日均 %s  %s" % (
            label, r["cagr"], r["sharpe"], r["mdd"], r["total"], r["avg_age"], r["seg"]), flush=True)

    for label, kw in [("ICIR+涨跌停", {"limit_check": True}),
                      ("ICIR+涨跌停+冲击", {"limit_check": True, "impact": 0.1}),
                      ("ICIR+涨跌停+大冲击", {"limit_check": True, "impact": 0.5})]:
        r = run("icir_t12", 12, None, **kw)
        out[label] = r
        print("%-14s %7.2f%% %8.3f %7.1f%% %7.2fx  %s" % (
            label, r["cagr"], r["sharpe"], r["mdd"], r["total"], r["seg"]), flush=True)

    for label, mode in [("基准:同池全部等权", "all"), ("基准:市值前3000等权", "top3000")]:
        r = run_bench(mode)
        out[label] = r
        print("%-14s %7.2f%% %8.3f %7.1f%% %7.2fx  %s" % (
            label, r["cagr"], r["sharpe"], r["mdd"], r["total"], r["seg"]), flush=True)

    print("\n逐年：策略 vs 同池等权基准", flush=True)
    ky = sorted(set(out["ICIR_t12"]["yearly"]) | set(out["基准:同池全部等权"]["yearly"]))
    print("%-6s %10s %10s %10s" % ("年份", "ICIR_t12", "基准", "超额"), flush=True)
    for y in ky:
        a = out["ICIR_t12"]["yearly"].get(y); b = out["基准:同池全部等权"]["yearly"].get(y)
        if a is None or b is None: continue
        print("%-6s %9.1f%% %9.1f%% %9.1f%%" % (y, a, b, a - b), flush=True)

    json.dump(out, open(os.path.join(HERE, "multifactor.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写 multifactor.json", flush=True)


if __name__ == "__main__":
    main()
