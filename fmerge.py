# -*- coding: utf-8 -*-
"""fmerge.py -- 把 fscan 的分段拼成两张表：IC 表 与 分位组合回测。

输入 fscan_parts/：每个段一个 meta_<id>.npy（调仓日序号）、close_/alive_/<因子>_<id>.npy
输出 ic_fast.json（IC/ICIR/t/正号率）与 fbeta.json（分位组合多空、超额、胜率、换手）。

为什么不在 fastpanel 里一次算完：196 个月 × 5251 只 × 11 因子 = 45MB，
这台机器上一次性算再存必然 MemoryError（见 compact.py 顶部）。分段落盘 + 这里逐段读，
峰值只有"一个段"的几 MB。
"""
import os, sys, glob, json, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compact import Panel
from fastpanel import FACTORS

PARTS = os.path.join(HERE, "fscan_parts")
FEE = 0.00025 + 0.00075
Q = 5


def spearman(a, b):
    if len(a) < 100:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def stat(x):
    x = np.asarray(x, dtype=np.float64)
    m, s = x.mean(), x.std(ddof=1)
    return m, (m / (s / math.sqrt(len(x))) if s > 0 else float("nan"))


def main():
    ids = sorted(int(os.path.basename(f)[5:-4]) for f in glob.glob(os.path.join(PARTS, "meta_*.npy")))
    print("段序号:", ids[0], "..", ids[-1], "共", len(ids))
    if not ids:
        print("fscan_parts/ 是空的：先跑 fscan.py")
        return
    p = Panel()
    dates = p.dates

    # 把所有段拼成一条连续的时间轴（段号升序 = 时间升序，fscan 是按时间顺序落盘的）
    JJ, CL, AL = [], [], []
    for sid in ids:
        jj = np.load(os.path.join(PARTS, "meta_%05d.npy" % sid))
        JJ.append(jj)
        CL.append(np.load(os.path.join(PARTS, "close_%05d.npy" % sid)))
        AL.append(np.load(os.path.join(PARTS, "alive_%05d.npy" % sid)))
    JJ = np.concatenate(JJ).astype(np.int64)
    order_ = np.argsort(JJ, kind="stable")      # 按交易日排序，段文件是分段乱序的
    JJ = JJ[order_]
    CL = np.vstack(CL)[order_]; AL = np.vstack(AL)[order_]
    print("时间轴 %d 个月：%s ~ %s" % (len(JJ), dates[JJ[0]], dates[JJ[-1]]))

    CN_CACHE = {}

    def close_at(j):
        if j not in CN_CACHE:
            if len(CN_CACHE) > 400:
                CN_CACHE.clear()
            CN_CACHE[j] = p.col("close", j)
        return CN_CACHE[j]

    def fwd_ret(m):
        j, j2 = int(JJ[m]), int(JJ[m + 1])
        cn = close_at(j2)
        cur = CL[m].astype(np.float64)
        return np.where((cur > 0) & (cn > 0), cn / np.where(cur > 0, cur, np.nan) - 1.0, np.nan)

    # 每个分段的因子矩阵只读一次（之前每次按月都重开文件，慢到不可用）
    MAT = {}

    def seg_mat(k, sid):
        key = (k, sid)
        if key not in MAT:
            MAT[key] = np.load(os.path.join(PARTS, "%s_%05d.npy" % (k, sid)))
        return MAT[key]

    def load_factor(k, m):
        return seg_mat(k, SEG_OF[m])[OFF_IN_SEG[m]].astype(np.float64)

    SEG_OF, OFF_IN_SEG = [], []
    for sid in ids:
        n = len(np.load(os.path.join(PARTS, "meta_%05d.npy" % sid)))
        SEG_OF.extend([sid] * n); OFF_IN_SEG.extend(range(n))
    SEG_OF = [SEG_OF[i] for i in order_]; OFF_IN_SEG = [OFF_IN_SEG[i] for i in order_]

    NM = len(JJ) - 1
    ic = {k: [] for k in FACTORS}
    for m in range(NM):
        r = fwd_ret(m)
        a = AL[m] & np.isfinite(r)
        for k in FACTORS:
            fv = load_factor(k, m)
            msk = a & np.isfinite(fv) & (np.abs(fv) < 1e12)
            ic[k].append(spearman(fv[msk], r[msk]) if msk.sum() >= 300 else np.nan)
    dirmap = {}
    for k in FACTORS:
        arr = np.array([x for x in ic[k] if np.isfinite(x)])
        dirmap[k] = -1.0 if (len(arr) and arr.mean() < 0) else 1.0
    print("方向（IC 为负则取反）:", {k: ("+" if v > 0 else "-") for k, v in dirmap.items()})

    per = {k: [] for k in FACTORS}
    prev_top = {k: None for k in FACTORS}
    month_dates = []
    for m in range(NM):
        r = fwd_ret(m)
        a0 = AL[m] & np.isfinite(r)
        month_dates.append(dates[JJ[m]])
        for k in FACTORS:
            fv = load_factor(k, m) * dirmap[k]
            msk = a0 & np.isfinite(fv) & (np.abs(fv) < 1e12)
            idx = np.flatnonzero(msk)
            if len(idx) < 500:
                per[k].append(None); continue
            order = idx[np.argsort(fv[idx], kind="stable")]
            b = np.linspace(0, len(order), Q + 1).astype(int)
            top = set(order[b[-2]:b[-1]].tolist())
            bot = order[b[0]:b[1]]
            t_ret = float(np.nanmean(r[list(top)]))
            b_ret = float(np.nanmean(r[bot]))
            mk = float(np.nanmean(r[idx]))
            to = 1.0 if prev_top[k] is None else len(top - prev_top[k]) / max(1, len(top))
            prev_top[k] = top
            per[k].append((t_ret, b_ret, mk, to))

    ic_res, fb_res = {}, {}
    # 胜率按月份对齐：不同因子的 None 位置可能不同，直接比 mean(ex>0) 会算错
    wins = {}
    for k in FACTORS:
        xs = [(m, x) for m, x in enumerate(per[k]) if x is not None]
        wins[k] = xs
    for k in FACTORS:
        arr = np.array([x for x in ic[k] if np.isfinite(x)])
        if len(arr) >= 12:
            m_, s_ = arr.mean(), arr.std(ddof=1)
            ic_res[k] = {"n": int(len(arr)), "ic_mean": round(float(m_), 4),
                         "icir": round(float(m_ / s_), 4) if s_ > 0 else None,
                         "t": round(float(m_ / (s_ / math.sqrt(len(arr)))), 2) if s_ > 0 else None,
                         "pos_rate": round(float((arr > 0).mean()), 3)}
        rows = [x for x in per[k] if x is not None]
        if len(rows) >= 12:
            t_ret = np.array([x[0] for x in rows]); b_ret = np.array([x[1] for x in rows])
            mk = np.array([x[2] for x in rows]); to = np.array([x[3] for x in rows])
            net = t_ret - to * FEE
            ls = t_ret - b_ret
            ex = net - mk
            ls_m, ls_t = stat(ls); ex_m, ex_t = stat(ex)
            fb_res[k] = {"n": len(rows), "ls_m": round(ls_m * 100, 3), "ls_t": round(ls_t, 2),
                         "top_m": round(float(t_ret.mean()) * 100, 3),
                         "excess_m": round(ex_m * 100, 3), "excess_t": round(ex_t, 2),
                         "win": round(float(np.nanmean(ex > 0)), 3),
                         "turn": round(float(to.mean()) * 100, 1),
                         "net_m": round(float(net.mean()) * 100, 3)}
    print("")
    print("%-10s %4s %8s %7s %7s %7s   | %8s %7s %8s %6s %7s" % (
        "factor", "n", "ic_mean", "icir", "t", "pos", "LS月均%", "LS_t", "超额月均%", "胜率", "月换手%"))
    for k in sorted(FACTORS, key=lambda x: -(ic_res.get(x, {}).get("t") or -9)):
        a = ic_res.get(k); b = fb_res.get(k)
        if not a:
            continue
        print("%-10s %4d %8.4f %7.3f %7.2f %7.3f   | %8.3f %7.2f %8.3f %6.3f %7.1f" % (
            k, a["n"], a["ic_mean"], a["icir"] or 0, a["t"] or 0, a["pos_rate"],
            b["ls_m"] if b else 0, b["ls_t"] if b else 0, b["excess_m"] if b else 0,
            b["win"] if b else 0, b["turn"] if b else 0))
    json.dump(ic_res, open(os.path.join(HERE, "ic_fast.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(fb_res, open(os.path.join(HERE, "fbeta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("")
    print("saved ic_fast.json / fbeta.json")


if __name__ == "__main__":
    main()
