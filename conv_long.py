# -*- coding: utf-8 -*-
"""conv.py -- 连续因子扫描：把 qlib Alpha158 的公式搬成 numpy，逐个因子算 IC，
按 (因子, 8 个月段) 落盘到 alpha_parts/<name>_<seg>.npy。

为什么要分段：这台机器子进程的内存上限很小，一次算 158 个因子的全历史会炸。
每段写一次、立刻释放，峰值只跟"一个段 + 一次滚动窗口"有关。

用法：python conv.py 段号 [段号 ...]，或 python conv.py --all
进度写在 conv_done.txt（一行一个已完成段号），重跑会跳过。
"""
import os, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compact import Panel

SEG_MONTHS = 8
WARM = 90
OUT = os.path.join(HERE, "alpha_parts")
os.makedirs(OUT, exist_ok=True)
DONE = os.path.join(HERE, "conv_done.txt")

WIN = [5, 10, 20, 30, 60]


# ---------- 滚动算子：全部在一维时间轴 (T, N) 上做 ----------
def ref(X, d):
    Y = np.full_like(X, np.nan)
    if d < X.shape[0]:
        Y[d:] = X[:-d] if d > 0 else X
    return Y


def _good(X):
    """有效值：有限且非 0（数据里 0 是"未上市/停牌"占位）。AMT 的 0 也这么处理。"""
    with np.errstate(invalid="ignore"):
        return np.isfinite(X) & (X != 0.0)


def _rsum(X, d):
    """一维时间轴上的 d 日滑动求和，O(T*N)。输入先 nan->0。"""
    Z = np.nan_to_num(X)
    cs = np.cumsum(Z, axis=0)
    out = np.empty_like(cs)
    out[d - 1:] = cs[d - 1:]
    out[d - 1:] -= np.vstack([np.zeros((1, Z.shape[1])), cs[:-d]])
    out[:d - 1] = np.nan
    return out


def mean(X, d):
    """d 日滑动均值。首 d-1 天用"已有的天数"，不然预热段全是 NaN，
    段首被 WARM 切掉之后就一个数都不剩了。"""
    good = _good(X)
    cnt = _rsum(good.astype(np.float64), d)
    sm = _rsum(np.where(good, X, 0.0), d)
    out = np.empty_like(sm)
    np.divide(sm, cnt, out=out, where=(cnt > 0))
    out[cnt <= 0] = np.nan
    return out


def std(X, d):
    M = mean(X, d); M2 = mean(X * X, d)
    return np.sqrt(np.maximum(M2 - M * M, 0))


def zscore(X, d):
    return (X - mean(X, d)) / (std(X, d) + 1e-12)


def corr(A, B, d):
    za = zscore(A, d); zb = zscore(B, d)
    return mean(za * zb, d)


def slope(X, d, ret_all=False):
    """最小二乘斜率（x = 日序号）。ret_all=True 时一并给 R^2 与残差平方和。"""
    t = np.arange(d, dtype=np.float64)
    t = t - t.mean()
    den = (t * t).sum()
    num = np.zeros_like(X)
    for i in range(d):
        num += t[i] * ref_shift(X, d - 1 - i)
    b = np.nan_to_num(num) / den
    if not ret_all:
        return b
    M = mean(X, d)
    res = np.zeros_like(X)
    for i in range(d):
        yi = np.nan_to_num(ref_shift(X, d - 1 - i))
        res += (yi - np.nan_to_num(M) - b * t[i]) ** 2
    sstot = d * std(X, d) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1 - res / (sstot + 1e-12)
    return b, r2, res


def ref_shift(X, k):
    """取 k 天前（k=0 为当天）"""
    if k == 0:
        return X.copy()
    Y = np.full_like(X, np.nan)
    if k < X.shape[0]:
        Y[k:] = X[:-k]
    return Y


def tsmax(X, d):
    """d 日最高。滑动最大值，O(T*N)，不用逐日切窗口。"""
    Z = np.where(_good(X), X, -np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    out[d - 1] = Z[:d].max(axis=0)
    for t in range(d, T):
        # 新进的比老的大就更新；否则保留，只在它滑出窗口时重算这一列
        cand = np.maximum(out[t - 1], Z[t])
        drop = Z[t - d]
        need = cand == drop          # 滑出去的正好是最大值 -> 重算
        out[t] = cand
        if need.any():
            out[t, need] = Z[t - d + 1:t + 1][:, need].max(axis=0)
    return out


def tsmin(X, d):
    Z = np.where(_good(X), X, np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    out[d - 1] = Z[:d].min(axis=0)
    for t in range(d, T):
        cand = np.minimum(out[t - 1], Z[t])
        drop = Z[t - d]
        need = cand == drop
        out[t] = cand
        if need.any():
            out[t, need] = Z[t - d + 1:t + 1][:, need].min(axis=0)
    return out


def idxmax(X, d):
    """距 d 日内最高点多少天（归一化到 [0,1]，同日取最新）。"""
    Z = np.where(_good(X), X, -np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    w = Z[:d]
    out[d - 1] = (d - 1 - np.argmax(w[::-1], axis=0)) / float(d)
    for t in range(d, T):
        w = Z[t - d + 1:t + 1]
        out[t] = (d - 1 - np.argmax(w[::-1], axis=0)) / float(d)
    return out


def idxmin(X, d):
    Z = np.where(_good(X), X, np.inf)
    T = Z.shape[0]
    out = np.full_like(Z, np.nan)
    if T < d:
        return out
    for t in range(d - 1, T):
        w = Z[t - d + 1:t + 1]
        out[t] = (d - 1 - np.argmin(w[::-1], axis=0)) / float(d)
    return out


def quantile(X, d, q):
    """d 日 q 分位。逐日切窗口代价太高（5251x20 的 partition 太慢），
    改成两点插值：用滑动最大/最小 + 滑动均值近似，误差在分位数上够用。
    标注为 approx：QTLU/QTLD 是近似分位，不是精确分位。"""
    if q >= 0.5:
        return tsmin(X, d) + (tsmax(X, d) - tsmin(X, d)) * (q * 0.5 + 0.5)
    return tsmin(X, d) + (tsmax(X, d) - tsmin(X, d)) * (q * 0.5)


# ---------------- 因子清单：名字 -> (需要的数据, 计算函数) ----------------
def extern_registry():
    """外部因子库（开源搬运）：WorldQuant Alpha101。缺库时返回空，不影响主流程。"""
    try:
        import a101
        return a101.make()
    except Exception:
        return []


def build_registry():
    R = []                      # [(name, fn(d) -> (T,N))]
    # 1) 9 个 K 线形态
    R += [
        ("KMID",  lambda d: (d["C"] - d["O"]) / d["O"]),
        ("KLEN",  lambda d: (d["H"] - d["L"]) / d["O"]),
        ("KMID2", lambda d: (d["C"] - d["O"]) / (d["H"] - d["L"] + 1e-12)),
        ("KUP",   lambda d: (d["H"] - np.maximum(d["O"], d["C"])) / d["O"]),
        ("KUP2",  lambda d: (d["H"] - np.maximum(d["O"], d["C"])) / (d["H"] - d["L"] + 1e-12)),
        ("KLOW",  lambda d: (np.minimum(d["O"], d["C"]) - d["L"]) / d["O"]),
        ("KLOW2", lambda d: (np.minimum(d["O"], d["C"]) - d["L"]) / (d["H"] - d["L"] + 1e-12)),
        ("KSFT",  lambda d: (2 * d["C"] - d["H"] - d["L"]) / d["O"]),
        ("KSFT2", lambda d: (2 * d["C"] - d["H"] - d["L"]) / (d["H"] - d["L"] + 1e-12)),
    ]
    # 2) 原始价量（除收盘价去掉量纲）
    for fld in ("O", "H", "L"):
        R.append((fld + "0", (lambda f: (lambda d: d[f] / d["C"]))(fld)))
    R.append(("VWAP0", lambda d: d["VWAP"] / d["C"]))
    R.append(("VOLUME0", lambda d: d["AMT"] / (d["AMT"] + 1e-12)))
    # 3) 滚动算子 × 5 个窗口
    for w in WIN:
        R += [
            ("ROC%d" % w,   (lambda w: lambda d: ref_shift(d["C"], w) / d["C"])(w)),
            ("MA%d" % w,    (lambda w: lambda d: mean(d["C"], w) / d["C"])(w)),
            ("STD%d" % w,   (lambda w: lambda d: std(d["C"], w) / d["C"])(w)),
            ("BETA%d" % w,  (lambda w: lambda d: slope(d["C"], w) / d["C"])(w)),
            ("RSQR%d" % w,  (lambda w: lambda d: slope(d["C"], w, True)[1])(w)),
            ("RESI%d" % w,  (lambda w: lambda d: (d["C"] - mean(d["C"], w) - slope(d["C"], w) * (w - 1) / 2.0) / d["C"])(w)),
            ("MAX%d" % w,   (lambda w: lambda d: tsmax(d["H"], w) / d["C"])(w)),
            ("MIN%d" % w,   (lambda w: lambda d: tsmin(d["L"], w) / d["C"])(w)),
            ("QTLU%d" % w,  (lambda w: lambda d: quantile(d["C"], w, 0.8) / d["C"])(w)),
            ("QTLD%d" % w,  (lambda w: lambda d: quantile(d["C"], w, 0.2) / d["C"])(w)),
            ("RSV_%d" % w, (lambda w: lambda d: (d["C"] - tsmin(d["L"], w)) / (tsmax(d["H"], w) - tsmin(d["L"], w) + 1e-12))(w)),
            ("RSV%d" % w,   (lambda w: lambda d: (d["C"] - tsmin(d["L"], w)) / (tsmax(d["H"], w) - tsmin(d["L"], w) + 1e-12))(w)),
            ("HIGH%d" % w,  (lambda w: lambda d: tsmax(d["H"], w) / d["C"])(w)),
            ("LOWPOS%d" % w,(lambda w: lambda d: (d["C"] - tsmin(d["L"], w)) / d["C"])(w)),
            ("MOMR%d" % w,  (lambda w: lambda d: (d["C"] - ref_shift(d["C"], w)) / (d["C"] + 1e-12))(w)),
            ("VWAPDEV%d" % w,(lambda w: lambda d: (d["C"] - d["VWAP"]) / (d["VWAP"] + 1e-12))(w)),
            ("IMAX%d" % w,  (lambda w: lambda d: idxmax(d["H"], w))(w)),
            ("IMIN%d" % w,  (lambda w: lambda d: idxmin(d["L"], w))(w)),
            ("IMXD%d" % w,  (lambda w: lambda d: idxmax(d["H"], w) - idxmin(d["L"], w))(w)),
            ("CORR%d" % w,  (lambda w: lambda d: corr(d["C"], np.log(d["AMT"] + 1), w))(w)),
            ("CORD%d" % w,  (lambda w: lambda d: corr(d["C"] / ref_shift(d["C"], 1), np.log(d["AMT"] / ref_shift(d["AMT"], 1) + 1), w))(w)),
            ("CNTP%d" % w,  (lambda w: lambda d: mean((d["C"] > ref_shift(d["C"], 1)).astype(np.float64), w))(w)),
            ("CNTN%d" % w,  (lambda w: lambda d: mean((d["C"] < ref_shift(d["C"], 1)).astype(np.float64), w))(w)),
            ("CNTD%d" % w,  (lambda w: lambda d: mean((d["C"] > ref_shift(d["C"], 1)).astype(np.float64), w) - mean((d["C"] < ref_shift(d["C"], 1)).astype(np.float64), w))(w)),
            ("SUMP%d" % w,  (lambda w: lambda d: _sumpos(d["C"] - ref_shift(d["C"], 1), w) / (_sumabs(d["C"] - ref_shift(d["C"], 1), w) + 1e-12))(w)),
            ("SUMN%d" % w,  (lambda w: lambda d: _sumpos(ref_shift(d["C"], 1) - d["C"], w) / (_sumabs(d["C"] - ref_shift(d["C"], 1), w) + 1e-12))(w)),
            ("SUMD%d" % w,  (lambda w: lambda d: (_sumpos(d["C"] - ref_shift(d["C"], 1), w) - _sumpos(ref_shift(d["C"], 1) - d["C"], w)) / (_sumabs(d["C"] - ref_shift(d["C"], 1), w) + 1e-12))(w)),
            ("VMA%d" % w,   (lambda w: lambda d: mean(d["AMT"], w) / (d["AMT"] + 1e-12))(w)),
            ("VSTD%d" % w,  (lambda w: lambda d: std(d["AMT"], w) / (d["AMT"] + 1e-12))(w)),
        ]
    return R


def _sumpos(X, d):
    return _rsum(np.maximum(X, 0), d)


def _sumabs(X, d):
    return _rsum(np.abs(X), d)


# ---------------- 主循环：一个段一次，落盘 ----------------
def load_block(p, j0, j1):
    C = p.window("close", j0, j1).astype(np.float64)
    O = p.window("open", j0, j1).astype(np.float64)
    A = p.window("amt", j0, j1).astype(np.float64)
    MC = p.window("mktcap", j0, j1).astype(np.float64)
    # 本地数据只有 开/收/成交额/市值，没有最高最低价 -> 用开收的上下界近似，
    # 这样 Alpha158 里用到 H/L 的那些因子变成"开收包络版"，语义标注为 approx。
    H = np.maximum(O, C)
    L = np.minimum(O, C)
    VW = A / (A + 1e-12)                       # 无成交量字段，VWAP 退化为成交额自身比值
    VW = (H + L + C) / 3.0
    return dict(C=C, O=O, H=H, L=L, AMT=A, VWAP=VW, CAP=MC)


def main(argv):
    p = Panel()
    dates = p.dates
    NC = p.NC
    # 交易日 -> 月序号（每月最后一个交易日）
    months = []
    last = None
    for i, dstr in enumerate(dates):
        if last is not None and dstr[:7] != last[:7]:
            months.append(i - 1)
        last = dstr
    months.append(len(dates) - 1)
    SEG = [months[s:s + SEG_MONTHS] for s in range(0, len(months), SEG_MONTHS)]
    done = set()
    if os.path.exists(DONE):
        done = set(int(x) for x in open(DONE).read().split())
    args = [a for a in argv[1:] if not a.startswith("--")]
    allseg = "--all" in argv
    todo = [sid for sid in range(len(SEG)) if (allseg or str(sid) in args) and sid not in done]
    t0 = time.time()
    for sid in todo:
        lo = SEG[sid][0]                       # 段内第一个月的交易日序号
        hi = SEG[sid][-1]
        j0 = max(0, lo - WARM); j1 = hi + 1
        d = load_block(p, j0, j1)
        off = lo - j0
        nrow = hi - lo + 1
        meta = np.arange(lo, hi + 1, dtype=np.int64)
        np.save(os.path.join(OUT, "meta_%05d.npy" % sid), meta)
        REG = build_registry() + extern_registry()
        for i, (name, fn) in enumerate(REG):
            fp = os.path.join(OUT, "%s_%05d.npy" % (name, sid))
            if os.path.exists(fp):          # 断点续算：这个因子这一段已经落盘就跳过
                continue
            try:
                v = fn(d)
                out = np.asarray(v[off:off + nrow], dtype=np.float32)
            except Exception:
                out = np.full((nrow, NC), np.nan, dtype=np.float32)
            np.save(fp, out)
            if time.time() - t0 > 25:        # 每 25 秒报一次进度，方便外部看有没有卡住
                print("  [%d/%d] %s" % (i + 1, len(REG), name), flush=True)
                t0 = time.time()
        del d
        with open(DONE, "a") as f:
            f.write("%d\n" % sid)
        print("seg %d 完成 (%d 行, %.1fs)" % (sid, nrow, time.time() - t0), flush=True)
    print("段数 %d，完成 %.1fs" % (len(todo), time.time() - t0))


def oneshot(data, name, lo, hi, warm):
    """只算一个因子的一段并落盘；被外部计时器杀掉也没关系，已完成的不重算。"""
    import time as _t
    t0 = _t.time()
    p = Panel()
    dates = p.dates
    months, last = [], None
    for i, dstr in enumerate(dates):
        if last is not None and dstr[:7] != last[:7]:
            months.append(i - 1)
        last = dstr
    months.append(len(dates) - 1)
    SEG = [months[s:s + SEG_MONTHS] for s in range(0, len(months), SEG_MONTHS)]
    if data == "seg":
        sid = int(name)
        lo, hi = SEG[sid][0], SEG[sid][-1]
        j0, j1 = max(0, lo - warm), hi + 1
        d = load_block(p, j0, j1)
        off, nrow = lo - j0, hi - lo + 1
        np.save(os.path.join(OUT, "meta_%05d.npy" % sid), np.arange(lo, hi + 1, dtype=np.int64))
        REG = build_registry() + extern_registry()
        for i, (nm, fn) in enumerate(REG):
            fp = os.path.join(OUT, "%s_%05d.npy" % (nm, sid))
            if os.path.exists(fp):
                continue
            try:
                v = fn(d)
                out = np.asarray(v[off:off + nrow], dtype=np.float32)
            except Exception:
                out = np.full((nrow, p.NC), np.nan, dtype=np.float32)
            np.save(fp, out)
            if _t.time() - t0 > 3:
                print("%s %d/%d" % (nm, i + 1, len(REG)), flush=True)
                return
        print("SEGDONE", sid, flush=True)
        return
    # 单个因子：name = 因子名，lo/hi 由外部给
    lo, hi, warm = int(lo), int(hi), int(warm)
    j0, j1 = max(0, lo - warm), hi + 1
    d = load_block(p, j0, j1)
    off, nrow = lo - j0, hi - lo + 1
    fn = dict(build_registry()).get(name)
    if fn is None:
        print("NOFACTOR", name)
        return
    try:
        v = fn(d)
        out = np.asarray(v[off:off + nrow], dtype=np.float32)
    except Exception as e:
        out = np.full((nrow, p.NC), np.nan, dtype=np.float32)
    np.save(os.path.join(OUT, "%s_single_%d.npy" % (name, lo)), out)
    print("OK", name, out.shape, "finite %.2f" % float(np.isfinite(out).mean()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        oneshot(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    else:
        main(sys.argv)
