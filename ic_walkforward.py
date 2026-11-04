"""滚动前推 IC 检验 —— 回答一个唯一的问题：有没有因子在时间外还有效。

为什么先做这个
--------------
在 11 个因子上做 2010-2026 的全样本回测，那是在同一份数据上挑最好的。
挑出来的第一名很可能只是那段时间的运气。这一节不再挑，只问：
  每个月的信号，和它下个月的收益，在同一个月里有关系吗（rank IC）？
如果 IC 在过去 16 年里大多数时候是正的，那这个信号有时间外的东西；
如果 IC 在 0 上下乱跳，那它排第几都没意义。

口径
----
  rank IC：调仓日 t 的因子值 与 t→t+1 个月收益 的 Spearman 相关
  ICIR   = IC 均值 / IC 标准差（衡量稳定性，不是收益）
  t 值   = IC 均值 / (IC 标准差 / sqrt(n))
  只看 |t| > 2 的，其余一律视为噪声

这一节不产生净值曲线，只产生一张"哪些因子真的有持续性"的表。
"""
import os, json, struct, mmap, time, datetime, sys, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import anomaly_race as AR


def spearman(a, b):
    """去 NaN 后的 Spearman。数据量小（每次几千个点），直接算秩。"""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100: return np.nan
    x, y = a[m], b[m]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    mx, my = rx.mean(), ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = math.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2))
    return num / den if den > 0 else np.nan


def main(k=50, start="2010-06-01"):
    t0 = time.time()
    C, ND, NC = AR.load_panel("m_close.bin")
    O, _, _ = AR.load_panel("m_open.bin")
    A, _, _ = AR.load_panel("amt_flat.bin")
    MK, _, _ = AR.load_panel("m_mktcap.bin")
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ci_of = {c: i for i, c in enumerate(codes)}
    print("[%5.1fs] 面板就绪" % (time.time() - t0), flush=True)

    si = next(i for i, d in enumerate(dates) if d >= start)
    live = (C > 2.0) & (A > 0)

    def ret_slice(j0, j1):
        prev = C[:, j0:j1 - 1]; cur = C[:, j0 + 1:j1]
        good = prev > 0
        out = np.full((C.shape[0], j1 - j0 - 1), np.nan, dtype=np.float32)
        np.divide(cur, np.where(good, prev, np.float32(1)), out=out, where=good)
        out[~good] = np.nan
        out -= 1
        return out

    # 因子值：每个调仓日的原始分（不做中性化，中性化单独一节）
    print("[%5.1fs] 预算因子矩阵..." % (time.time() - t0), flush=True)

    # 调仓日：每月第一个交易日
    rebal = [j for j in range(si + 1, ND) if dates[j][:7] != dates[j - 1][:7]]
    print("调仓日 %d 个：%s ~ %s" % (len(rebal), dates[rebal[0]], dates[rebal[-1]]), flush=True)

    # 财务矩阵按需建、用完即删。三个同时持有要 531MB，沙箱只能给约 480MB。
    FM = {}

    _cur = {"fld": None}

    def _get(fld):
        """取该字段的整块矩阵；同一时刻只保留一块。"""
        if _cur["fld"] != fld:
            FM.clear()
            import gc; gc.collect()
            FM[fld] = AR.fund_matrix(codes, ci_of, dates, field=fld)
            _cur["fld"] = fld
        return FM[fld]

    def factor_at(name, j):
        """返回该日的因子值向量。全部做成同向：越大越好。"""
        if name == "size":   return -np.log(np.maximum(MK[:, j], 1))
        if name == "ep":     return np.where(C[:, j] > 0, _get("eps")[:, j] / np.where(C[:, j] > 0, C[:, j], np.nan), np.nan)
        if name == "bm":     return np.where(C[:, j] > 0, _get("bps")[:, j] / np.where(C[:, j] > 0, C[:, j], np.nan), np.nan)
        if name == "roe":    return _get("roe")[:, j].astype(np.float64)
        if name == "volatility":
            return -np.nanstd(ret_slice(max(0, j - 20), j + 1), axis=1).astype(np.float64)
        if name == "max":
            return -np.nanmax(ret_slice(max(0, j - 20), j + 1), axis=1).astype(np.float64)
        if name == "turnover":
            j0 = max(0, j - 20); m = MK[:, j0:j + 1]
            t = A[:, j0:j + 1] / np.where(m > 0, m, np.nan)
            return -np.nanmean(t, axis=1)
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
        if name == "reversal":
            return -ret_slice(max(0, j - 21), j + 1)[:, -1].astype(np.float64) if j >= 21 else np.full(NC, np.nan)
        if name == "momentum":
            return ret_slice(max(0, j - 251), j + 1)[:, -1].astype(np.float64) if j >= 251 else np.full(NC, np.nan)
        raise KeyError(name)

    NAMES = ["size", "ep", "bm", "roe", "volatility", "max", "turnover",
             "abnormal_turnover", "illiquidity", "reversal", "momentum"]
    IC = {n: [] for n in NAMES}

    # 外循环按因子：一个因子把全部调仓日算完再换下一个，避免同时持有多个财务矩阵
    FIELDS = {"ep": "eps", "bm": "bps", "roe": "roe"}
    for n in NAMES:
        if n in FIELDS:
            _cur["fld"] = None
            _get(FIELDS[n])
        for ri in range(len(rebal) - 1):
            j = rebal[ri]; jn = rebal[ri + 1]
            if jn >= ND: break
            fwd = np.full(NC, np.nan)
            j2 = rebal[ri + 2] if ri + 2 < len(rebal) and rebal[ri + 2] < ND else ND - 1
            ok = (O[:, jn] > 0) & (O[:, j2] > 0)
            fwd[ok] = O[ok, j2] / O[ok, jn] - 1
            valid = live[:, j] & np.isfinite(fwd)
            if valid.sum() < 200: continue
            f = factor_at(n, j)
            ic = spearman(np.where(valid, f, np.nan), np.where(valid, fwd, np.nan))
            if np.isfinite(ic): IC[n].append((dates[j], float(ic)))
        v = np.array([x[1] for x in IC[n]])
        print("[%5.1fs] %-20s %d 期, IC均值 %.4f" % (
            time.time() - t0, n, len(v), v.mean() if len(v) else 0), flush=True)

    out = {}
    print("\n%-20s %6s %8s %8s %8s %8s %8s" % ("因子", "期数", "IC均值", "IC标准差", "ICIR", "t值", "IC>0占比"), flush=True)
    print("-" * 74, flush=True)
    for n in NAMES:
        v = np.array([x[1] for x in IC[n]])
        if len(v) < 10: continue
        mu, sd = v.mean(), v.std(ddof=1)
        icir = mu / sd if sd > 0 else 0
        tstat = mu / (sd / math.sqrt(len(v))) if sd > 0 else 0
        pos = float(np.mean(v > 0))
        out[n] = {"n": int(len(v)), "ic_mean": round(float(mu), 4), "ic_std": round(float(sd), 4),
                  "icir": round(float(icir), 4), "t": round(float(tstat), 2),
                  "pos_rate": round(pos, 3),
                  "series": [[d, round(x, 4)] for d, x in IC[n]]}
        print("%-20s %6d %8.4f %8.4f %8.3f %8.2f %7.1f%%" % (n, len(v), mu, sd, icir, tstat, pos * 100), flush=True)

    print("\n按 |t| 排序（只有 |t|>2 才算有持续性）：", flush=True)
    for i, (n, r) in enumerate(sorted(out.items(), key=lambda x: -abs(x[1]["t"])), 1):
        flag = "**" if abs(r["t"]) > 2 else "  "
        print("%2d. %s %-20s t=%6.2f  ICIR=%6.3f  IC>0 %.1f%%" % (
            i, flag, n, r["t"], r["icir"], r["pos_rate"] * 100), flush=True)

    json.dump(out, open(os.path.join(HERE, "ic_analysis.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写 ic_analysis.json（含每个因子的逐月 IC 序列）", flush=True)


if __name__ == "__main__":
    main()
