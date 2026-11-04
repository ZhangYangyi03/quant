"""fastpanel.py -- 分块流式算因子：全市场、全长历史，进程常驻内存 < 300MB。

为什么改写（2026-11-04 实测）
---------------------------
本机子进程的提交上限 512MB（Windows Job Object, LimitFlags 0x270c,
ProcessMemoryLimit=512MB），且剩余额度被切碎：numpy 一次要 >2MB 的连续内存就
MemoryError，一次 f.read() 超过约 1MB 也会 MemoryError。原来的 anomaly_race /
factor_lab 会 mmap 整张 370MB 的 float64 面板，一次映射就吃掉 370MB，随后任何
分配都是 MemoryError —— 这就是那些会话"算到一半突然没了"的机械原因。

这里的做法
-----------
  1. compact.py：float32 + 按【日期 x 股票】存。某一天=21KB，某 250 天块=5MB。
  2. 一次只装一个 250 天块（4 个字段共 20MB），块内逐日推进，状态都是 O(NC) 的
     累加器 / 环形缓冲；块尾 260 天重叠给滚动窗口预热。
  3. 输出只保留调仓日那一行（196 x 5251 的因子矩阵，每个 4MB）。

口径
----
  因子值只用调仓日当天及以前；收益 = 调仓日收盘 -> 下一个调仓日收盘；
  只在收盘价 > 2 元且有成交的日子里排序（停牌、退市不参与）。

用法：
    python fastpanel.py ic                  # 逐月 rank IC / ICIR / t -> ic_fast.json
    python fastpanel.py ic --start 2016-01-01
"""
import os, sys, json, math, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all='ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compact import Panel

FACTORS = ["size", "turn", "ret20", "ret60", "vol20", "amt20", "hl", "reversal1",
           "abn_turn", "max20",
           # 第二轮扩的因子：日内/隔夜结构、偏度、跳空、量价相关
           "intraday", "overnight", "skew60", "pvcorr", "mom_acc", "vol_ratio", "amt_cv"]


def rolling_features(j0, j1, p):
    NC = p.NC
    ND = j1 - j0
    out = {k: np.full((ND, NC), np.nan, dtype=np.float32) for k in FACTORS}
    out["_close"] = np.full((ND, NC), np.nan, dtype=np.float32)
    rows = {f: p.window(f, j0, j1) for f in ("close", "amt", "open", "mktcap")}
    prev_close = None
    dq_close, dq_amt, dq_turn, ret_hist = [], [], [], []
    # 定形缓冲：abn_turn 要 250 天、vol/max 要 20 天。逐日 np.stack 一次 250x5251
    # 的数组是这里的主要开销（实测 1.5s/200 天），改成"写一行、整体 nanmean"。
    B_TURN = np.full((250, NC), np.nan, dtype=np.float32)
    B_RET = np.full((20, NC), np.nan, dtype=np.float32)
    B_AMT = np.full((20, NC), np.nan, dtype=np.float32)
    B_GAP = np.full((20, NC), np.nan, dtype=np.float32)        # 隔夜跳空
    B_IR = np.full((60, NC), np.nan, dtype=np.float32)         # 日内收益，skew60/pvcorr 用
    B_AM60 = np.full((60, NC), np.nan, dtype=np.float32)
    B_R60 = np.full((60, NC), np.nan, dtype=np.float32)
    B_MOM = []                                                  # (r5 截面, r20 的均值) 滚动
    filled = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(ND):
            C = rows["close"][i].astype(np.float64)
            A = rows["amt"][i].astype(np.float64)
            O = rows["open"][i].astype(np.float64)
            MK = rows["mktcap"][i].astype(np.float64)
            r1 = np.full(NC, np.nan)
            if prev_close is not None:
                ok = (C > 0) & (prev_close > 0)
                r1[ok] = C[ok] / prev_close[ok] - 1.0
            turn = np.where(MK > 0, A / MK, np.nan)
            dq_close.append(C)
            if len(dq_close) > 61: dq_close.pop(0)
            slot = i % 250
            B_TURN[slot] = turn
            B_RET[i % 20] = r1
            B_AMT[i % 20] = A
            B_GAP[i % 20] = np.where((O > 0) & (prev_close > 0), O / np.where(prev_close > 0, prev_close, np.nan) - 1.0, np.nan) if prev_close is not None else np.nan
            B_IR[i % 60] = np.where((O > 0) & (C > 0), C / np.where(O > 0, O, np.nan) - 1.0, np.nan)
            B_AM60[i % 60] = A
            B_R60[i % 60] = r1
            filled += 1

            out["size"][i] = np.log(np.where(MK > 0, MK, np.nan))
            out["turn"][i] = turn
            out["hl"][i] = (np.maximum(C, O) - np.minimum(C, O)) / np.where(O > 0, O, np.nan)
            out["reversal1"][i] = -r1
            out["_close"][i] = C.astype(np.float32)
            for n, key in ((20, "ret20"), (60, "ret60")):
                if len(dq_close) > n:
                    past = dq_close[-1 - n]
                    out[key][i] = np.where((C > 0) & (past > 0), C / past - 1.0, np.nan)
            # ---- 第二轮因子 ----
            if filled >= 1:
                # 日内：(收盘/开盘-1) 的 20 日均值，正=尾盘被买
                out["intraday"][i] = np.nanmean(B_IR[:min(filled, 20)], axis=0)
                # 隔夜：(开盘/昨收-1) 的 20 日均值
                out["overnight"][i] = np.nanmean(B_GAP[:min(filled, 20)], axis=0)
            if filled >= 20:
                base = np.nanmean(B_AMT, axis=0)
                sd = np.nanstd(B_AMT, axis=0)
                out["amt_cv"][i] = np.where(base > 0, sd / base, np.nan)
            # 动量加速度：近 5 日收益 - 近 20 日收益的 1/4；正=加速
            if filled >= 21:
                r5 = np.full(NC, np.nan); r20 = np.full(NC, np.nan)
                if len(dq_close) > 5:
                    p5 = dq_close[-6]
                    r5 = np.where((C > 0) & (p5 > 0), C / p5 - 1.0, np.nan)
                if len(dq_close) > 20:
                    p20 = dq_close[-21]
                    r20 = np.where((C > 0) & (p20 > 0), C / p20 - 1.0, np.nan)
                out["mom_acc"][i] = r5 - r20 / 4.0
            if filled >= 20:
                a5 = np.nanmean(B_AMT[:5], axis=0); a20 = np.nanmean(B_AMT, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["vol_ratio"][i] = np.where(a20 > 0, a5 / a20, np.nan)
            if filled >= 60:
                m = np.nanmean(B_IR, axis=0)
                sd = np.nanstd(B_IR, axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["skew60"][i] = np.where(sd > 0, np.nanmean(((B_IR - m) / np.where(sd > 0, sd, 1)) ** 3, axis=0), np.nan)
                # 量价相关：60 天里 amt 与同日收益的秩相关，简化用标准化后的点积
                ra = B_R60 - np.nanmean(B_R60, axis=0)
                aa = B_AM60 - np.nanmean(B_AM60, axis=0)
                num = np.nansum(ra * aa, axis=0)
                den = np.sqrt(np.nansum(ra * ra, axis=0) * np.nansum(aa * aa, axis=0))
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["pvcorr"][i] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            if filled >= 21:
                out["vol20"][i] = np.nanstd(B_RET, axis=0)
                out["max20"][i] = -np.nanmax(B_RET, axis=0)
            if filled >= 20:
                out["amt20"][i] = np.log1p(np.nanmean(B_AMT, axis=0))
                if filled >= 250:
                    t20 = np.nanmean(B_TURN[:20] if filled == 250 else B_TURN[:filled], axis=0)
                    t250 = np.nanmean(B_TURN, axis=0)
                    out["abn_turn"][i] = np.where(t250 > 0, t20 / t250, np.nan)
            prev_close = C
    del rows
    return out


def spearman(a, b):
    if len(a) < 100:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def main(argv):
    start = "2010-07-01"
    if "--start" in argv:
        start = argv[argv.index("--start") + 1]
    p = Panel()
    dates = p.dates
    si = next(i for i, d in enumerate(dates) if d >= start)
    rb = [j for j in range(si, p.ND - 1) if dates[j + 1][:7] != dates[j][:7]]
    print("调仓日 %d 个: %s ~ %s" % (len(rb), dates[rb[0]], dates[rb[-1]]), flush=True)
    t0 = time.time()
    FEAT = {k: np.full((len(rb), p.NC), np.nan, dtype=np.float32) for k in FACTORS}
    CUR = np.full((len(rb), p.NC), np.nan, dtype=np.float32)
    ALIVE = np.zeros((len(rb), p.NC), dtype=bool)
    rb_pos = {j: i for i, j in enumerate(rb)}
    keep_set = set()
    if "--from" in argv:
        _f0 = argv[argv.index("--from") + 1]
        rb = [j for j in rb if dates[j] >= _f0]
    if "--to" in argv:
        _f1 = argv[argv.index("--to") + 1]
        rb = [j for j in rb if dates[j] < _f1]
    PRE = 260
    SCAN = int(os.environ.get('SCAN', '120'))
    t_f = time.time()
    CH = SCAN
    c0 = max(0, rb[0] - PRE)
    while c0 <= rb[-1]:
        c1 = min(rb[-1] + 1, c0 + CH + PRE)
        F = rolling_features(c0, c1, p)
        for i, j in enumerate(range(c0, c1)):
            k = rb_pos.get(j)
            if k is None:
                continue
            for name in FACTORS:
                FEAT[name][k] = F[name][i]
            CUR[k] = F["_close"][i]
            ALIVE[k] = CUR[k] > 2.0
        del F
        msg = "[%6.1fs] 扫描到 %s  (调仓日 %d/%d)" % (
            time.time() - t0, dates[c1 - 1],
            sum(1 for j in range(rb[0], c1) if j in rb_pos), len(rb))
        print(msg + "   [本块 %.1fs]" % (time.time() - t_f), flush=True)
        t_f = time.time()
        with open(os.path.join(HERE, "_ic_progress.txt"), "a", encoding="utf-8") as pf:
            pf.write(msg + "\n")
        if c1 >= rb[-1] + 1:
            break
        c0 = c1 - PRE
    # 落盘：分位回测要原始因子值和下月收益，这里一次性导出
    FWD = np.full((len(rb), p.NC), np.nan, dtype=np.float32)
    for k in range(len(rb) - 1):
        cn = p.col("close", rb[k + 1])
        with np.errstate(invalid="ignore", divide="ignore"):
            FWD[k] = np.where((CUR[k] > 0) & (cn > 0), cn / CUR[k] - 1.0, np.nan)
    np.savez(os.path.join(HERE, "ic_fast.npz"),
             dates=np.array([dates[j] for j in rb], dtype="U10"),
             alive=ALIVE, fwd=FWD, **{k: FEAT[k] for k in FACTORS})

    # 分块续跑：--from/--to 用日期界定时段，只算这段的调仓日，输出到独立 npz
    if "--from" in argv or "--to" in argv:
        f0 = argv[argv.index("--from") + 1] if "--from" in argv else "1990-01-01"
        f1 = argv[argv.index("--to") + 1] if "--to" in argv else "2099-01-01"
        keep = [k for k, j in enumerate(rb) if f0 <= dates[j] < f1]
        tag = "%s_%s" % (f0, f1)
        out = os.path.join(HERE, "ic_part_%s.npz" % tag)
        # 分块落盘：每块单独一个 npz，单片足够小（<20MB），合并脚本逐片读。
        # 不能整份 savez：那会一次要 30MB 连续内存，在这台机器上直接 MemoryError。
        import zipfile
        CH_M = 12
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            def put(name, arr):
                import io
                buf = io.BytesIO()
                np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)
                zf.writestr(name + ".npy", buf.getvalue())
            put("dates", np.array([dates[rb[k]] for k in keep], dtype="U10"))
            for m0 in range(0, len(keep), CH_M):
                sl = keep[m0:m0 + CH_M]
                tagm = "%04d" % m0
                put("alive_" + tagm, ALIVE[sl])
                put("fwd_" + tagm, FWD[sl])
                for k in FACTORS:
                    put("%s_%s" % (k, tagm), FEAT[k][sl])
                print("  写入 %s 个月 %d~%d" % (tag, m0, m0 + len(sl) - 1), flush=True)
        print("saved %s  (%d 个月)" % (out, len(keep)))
        return

    res = {}
    for name in FACTORS:
        ics = []
        for k in range(len(rb) - 1):
            cn = p.col("close", rb[k + 1])
            with np.errstate(invalid="ignore", divide="ignore"):
                r = np.where((CUR[k] > 0) & (cn > 0), cn / CUR[k] - 1.0, np.nan).astype(np.float64)
            fv = FEAT[name][k].astype(np.float64)
            m = ALIVE[k] & np.isfinite(fv) & np.isfinite(r) & (np.abs(fv) < 1e12)
            if m.sum() < 300:
                continue
            ics.append(spearman(fv[m], r[m]))
        arr = np.array([x for x in ics if np.isfinite(x)])
        if len(arr) < 12:
            continue
        m_, s_ = arr.mean(), arr.std(ddof=1)
        res[name] = {"n": int(len(arr)), "ic_mean": round(float(m_), 4),
                     "icir": round(float(m_ / s_), 4) if s_ > 0 else None,
                     "t": round(float(m_ / (s_ / math.sqrt(len(arr)))), 2) if s_ > 0 else None,
                     "pos_rate": round(float((arr > 0).mean()), 3)}
    print("")
    print("%-12s %5s %9s %8s %7s %7s" % ("factor", "n", "ic_mean", "icir", "t", "pos"))
    for name, v in sorted(res.items(), key=lambda kv: -(kv[1]["t"] or -9)):
        print("%-12s %5d %9.4f %8.3f %7.2f %7.3f" % (name, v["n"], v["ic_mean"],
                                                     v["icir"] or 0, v["t"] or 0, v["pos_rate"]))
    json.dump(res, open(os.path.join(HERE, "ic_fast.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved ic_fast.json  %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main(sys.argv)
