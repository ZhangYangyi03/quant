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
    glue = set(x.rsplit("_", 1)[0] for x in os.listdir(PARTS) if x.endswith(".npy"))
    FAC = [k for k in FACTORS if k in glue]          # 只算这段扫描里真的算出来的因子

    # 拼时间轴：只保留"每个因子文件都在"的段，按交易日排序（段号不一定等于时间顺序）
    ok_ids = set()
    for sid in ids:
        if all(os.path.exists(os.path.join(PARTS, "%s_%05d.npy" % (k, sid))) for k in FAC):
            ok_ids.add(sid)
    pairs = []
    for sid in ids:
        if sid not in ok_ids:
            continue
        n = len(np.load(os.path.join(PARTS, "meta_%05d.npy" % sid)))
        pairs.extend([(int(np.load(os.path.join(PARTS, "meta_%05d.npy" % sid))[m]), sid, m) for m in range(n)])
    pairs.sort(key=lambda x: x[0])
    JJ = np.array([x[0] for x in pairs], dtype=np.int64)
    SEG_OF = [x[1] for x in pairs]
    OFF_IN_SEG = [x[2] for x in pairs]
    CLM = {sid: np.load(os.path.join(PARTS, "close_%05d.npy" % sid)) for sid in ok_ids}
    ALM = {sid: np.load(os.path.join(PARTS, "alive_%05d.npy" % sid)) for sid in ok_ids}

    class _View:
        def __getitem__(self, i):
            return CLM[SEG_OF[i]][OFF_IN_SEG[i]]
        def __len__(self):
            return len(SEG_OF)
    CL = _View()
    AL = _View()
    print("时间轴 %d 个月：%s ~ %s，因子 %d 个（跳过 %d 个缺文件的段）" % (
        len(JJ), dates[JJ[0]], dates[JJ[-1]], len(FAC), len(ids) - len(ok_ids)))

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
        cur = np.asarray(CL[m], dtype=np.float64)
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

    # 只保留"每个因子文件都存在"的段，否则新因子的段和旧段混在一起会读空
    ok_ids = []
    for sid in ids:
        if all(os.path.exists(os.path.join(PARTS, "%s_%05d.npy" % (k, sid))) for k in FAC):
            ok_ids.append(sid)
    pairs = []
    for sid in ids:
        n = len(np.load(os.path.join(PARTS, "meta_%05d.npy" % sid)))
        pairs.extend([(int(JJ[m]), sid, m) for m in range(n)])
    pairs = [x for x in pairs if x[1] in set(ok_ids)]
    JJ = np.array([x[0] for x in pairs], dtype=np.int64)
    SEG_OF = [x[1] for x in pairs]; OFF_IN_SEG = [x[2] for x in pairs]

    NM = len(JJ) - 1
    ic = {k: [] for k in FACTORS}
    for m in range(NM):
        r = fwd_ret(m)
        a = np.array(AL[m], dtype=bool) & np.isfinite(r)
        for k in FAC:
            fv = load_factor(k, m)
            msk = a & np.isfinite(fv) & (np.abs(fv) < 1e12)
            ic[k].append(spearman(fv[msk], r[msk]) if msk.sum() >= 300 else np.nan)
    dirmap = {}
    for k in FAC:
        arr = np.array([x for x in ic[k] if np.isfinite(x)])
        dirmap[k] = -1.0 if (len(arr) and arr.mean() < 0) else 1.0
    print("方向（IC 为负则取反）:", {k: ("+" if v > 0 else "-") for k, v in dirmap.items()})

    per = {k: [] for k in FACTORS}
    prev_top = {k: None for k in FACTORS}
    month_dates = []
    for m in range(NM):
        r = fwd_ret(m)
        a0 = np.array(AL[m], dtype=bool) & np.isfinite(r)
        month_dates.append(dates[JJ[m]])
        for k in FAC:
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
            per[k].append((t_ret, b_ret, mk, to, dates[JJ[m]]))

    _MS = [dates[JJ[m]] for m in range(len(JJ) - 1)]     # 每个 per[k] 元素对应的月份字符串
    ic_res, fb_res = {}, {}
    # 胜率按月份对齐：不同因子的 None 位置可能不同，直接比 mean(ex>0) 会算错
    wins = {}
    for k in FAC:
        xs = [(m, x) for m, x in enumerate(per[k]) if x is not None]
        wins[k] = xs
    for k in FAC:
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
    # 分两段：2019 前 / 2019 后，看因子是不是只在某一段成立
    def seg_stats(m_lo, m_hi):
        out = {}
        for k in FAC:
            rows = [x for x in per[k] if x is not None and m_lo <= x[4] < m_hi]
            if len(rows) < 24:
                out[k] = None; continue
            t_ret = np.array([x[0] for x in rows]); b_ret = np.array([x[1] for x in rows])
            mk = np.array([x[2] for x in rows]); to = np.array([x[3] for x in rows])
            ex = (t_ret - to * FEE) - mk
            out[k] = (ex.mean() * 100, ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex))), len(ex))
        return out
    sp = next((m for m in range(len(month_dates)) if month_dates[m] >= "2019-01-01"), len(month_dates))
    A = seg_stats("1990-01-01", "2019-01-01"); B_ = seg_stats("2019-01-01", "2099-01-01")
    print("")
    print("%-10s %14s %14s %14s" % ("factor", "2019前 超额%/t", "2019后 超额%/t", "IC_t"))
    for k in sorted(FAC, key=lambda x: -(ic_res.get(x, {}).get("t") or -9)):
        a, b = A.get(k), B_.get(k)
        fa = ("%7.2f /%5.2f" % (a[0], a[1])) if a else "     n/a     "
        fb = ("%7.2f /%5.2f" % (b[0], b[1])) if b else "     n/a     "
        print("%-10s %14s %14s %14.2f" % (k, fa, fb, ic_res.get(k, {}).get("t") or 0))
    print("")
    print("%-10s %4s %8s %7s %7s %7s   | %8s %7s %8s %6s %7s" % (
        "factor", "n", "ic_mean", "icir", "t", "pos", "LS月均%", "LS_t", "超额月均%", "胜率", "月换手%"))
    for k in sorted(FAC, key=lambda x: -(ic_res.get(x, {}).get("t") or -9)):
        a = ic_res.get(k); b = fb_res.get(k)
        if not a:
            continue
        print("%-10s %4d %8.4f %7.3f %7.2f %7.3f   | %8.3f %7.2f %8.3f %6.3f %7.1f" % (
            k, a["n"], a["ic_mean"], a["icir"] or 0, a["t"] or 0, a["pos_rate"],
            b["ls_m"] if b else 0, b["ls_t"] if b else 0, b["excess_m"] if b else 0,
            b["win"] if b else 0, b["turn"] if b else 0))
    # 面板级缓存（供 mlfit.py 做截面回归/GBDT 用，不重复扫盘）
    # 不再落 fpanel.npz：42MB 的 npz 在这台机器上读一次就炸（解压 + json.load 都超限），
    # mlfit2.py 改成直接读 fscan_parts（fraw.rows 逐行取）。
    if len(JJ) < 150:
        print("\n只有 %d 个月（扫描还没跑完），不覆盖全样本结果文件" % len(JJ))
        return
    json.dump(ic_res, open(os.path.join(HERE, "ic_fast.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(fb_res, open(os.path.join(HERE, "fbeta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("")
    print("saved ic_fast.json / fbeta.json")


if __name__ == "__main__":
    main()
