"""factor_lab.py —— 因子库：算 + 看 + 比。

问题：quant 的因子原来是手写 11 个、散在 multifactor.py 里，没法单独看每一个的
IC / ICIR / 换手 / 稳定性。这里把"算什么"和"怎么量"分开：

  1. compute   在 m_*.bin 的列式面板上算一批因子，按调仓日存 npz
  2. summarize 每个因子的 IC 均值、ICIR、t 值、前后半段是否同号、换手

因子集：Alpha158 里最常用的那部分（K 线形态、滚动统计、量价关系），公式按 qlib
公开定义自己实现，不依赖 qlib（本机也没装）。以后换任何开源因子集，只要吐成
{codes, dates, {name: matrix}} 这个形状，就能进同一张对比表。

用法：
  python factor_lab.py compute --start 2016-01-01
  python factor_lab.py summarize
"""
import json, os, struct, sys, time

# ---- 必须在 import numpy 之前 ----
# 这台机器的 numpy 一 import 就崩：OpenBLAS 默认按核数开线程池，每池要预留缓冲区，
# 分配失败后它把整个进程的内存状态带坏（原文 "Memory allocation still failed after
# 10 retries, giving up."），之后连 4MB 都分不出来。设成单线程就完全正常。
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "fl_cache")


def codes_for_npz():
    src = os.path.join(HERE, "stockdata")
    return sorted(f[:-4] for f in os.listdir(src) if f.endswith(".csv") and not f.startswith("_"))


def load_json(path):
    """沙箱里 fp.read() 之后 utf-8 解码会同时持有 bytes 与 str 两份，
    而这里的堆只有几 MB —— 走 mmap 一次性解码，json.load(open()) 会 MemoryError。"""
    import mmap
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return json.loads(m[:].decode("utf-8"))
        finally:
            m.close()


def write_json(path, obj):
    import mmap as _m
    data = json.dumps(obj, ensure_ascii=False, indent=1)
    with open(path, "wb") as f:
        f.write(data.encode("utf-8"))


def _mmap(path, dtype, shape):
    """整文件 mmap 后切片。

    不能直接 np.memmap(offset=16, shape=...)：Windows 的 CreateFileMapping 会按
    "offset + 请求长度"对齐到的页数预留地址空间，三张 370MB 的面板一起打开就在
    14.7GB 的机器上撞 WinError 1455（可用物理内存 3.9GB）。整文件映射一次、
    再按需切片，内存压力只跟真正碰到的那一页走。
    """
    mm = np.memmap(path, dtype=dtype, mode="r")        # shape 不给就是整文件
    # 跳过 16 字节文件头：magic "PXFL0002" + ND + NC。不跳的话整个矩阵会被挪 2 格，
    # 表现为"左边两列是垃圾、第一只票错位成第二只" —— 不报错，只会安静地算错。
    off = 16 // np.dtype(dtype).itemsize
    n = int(np.prod(shape))
    return mm[off:off + n].reshape(shape)


def panel(fields=("close",)):
    """读 m_*.bin 列式面板。形状以 _dates.json 与 build_flat 的 codes_all() 长度为准
    （文件头里的 ND 是历史遗留，不能直接用）。"""
    dates = load_json(os.path.join(HERE, "_dates.json"))["dates"]
    # NC 必须是真实股票数：写死 5251 会让整个矩阵错位 2 格（左边多两列垃圾、
    # 每只票的价格挪到下一天）—— 不报错，只会安静地算错。
    NC = len(codes_for_npz())
    out = {}
    for f in fields:
        p = os.path.join(HERE, f"m_{f}.bin")
        if not os.path.exists(p):
            continue
        out[f] = _mmap(p, np.float64, (NC, len(dates)))
    return dates, out


def to_f32(a):
    return np.asarray(a, dtype=np.float32)


def safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return a / b


def _buf(shape, dtype=np.float32):
    """从 mmap 取一块内存当数组。

    为什么不直接 np.empty：这台机器上 numpy 一旦在 malloc 里撞上一次失败，
    后面每一次分配都必失败（进程的内存状态被带坏）。mmap 出来再 frombuffer
    不经过 malloc，实测稳定；用完显式释放。
    """
    import mmap
    mm = mmap.mmap(-1, int(np.prod(shape)) * np.dtype(dtype).itemsize)
    return mm, np.frombuffer(mm, dtype=dtype, count=int(np.prod(shape))).reshape(shape)


def roll(x, n, fn="mean"):
    """沿时间轴滚动。nan 不参与，和 pandas 的默认一致。

    实现上刻意不用 nancumsum / concatenate：那些会在 5251×1460 的块上
    一次性申请 58MB，而这台机器上要靠 mmap 才稳。这里用"定长窗口滑动累加"，
    只常驻 (NC, T) 级别的一两块，并且只对末列做真正的开窗。
    """
    NC, T = x.shape
    if T < n:
        return np.full(x.shape, np.nan, dtype=np.float32)
    if fn == "mean":
        # nan 必须先置 0 再累加：nan 一旦进了累加链，后面每个窗口都是 nan，
        # 而且是"安静的 nan" —— 不报错，只会让整个因子变成 n<8 的空壳。
        mm, out = _buf(x.shape, np.float32)
        acc = np.zeros(NC, dtype=np.float64)
        cnt = np.zeros(NC, dtype=np.float64)
        xf = np.asarray(x, dtype=np.float64)
        bad = np.isnan(xf)
        xz = np.where(bad, 0.0, xf)
        for t in range(T):
            if t >= n:                       # 滑出
                acc -= xz[:, t - n]
                cnt -= 1 - bad[:, t - n]
            acc += xz[:, t]
            cnt += 1 - bad[:, t]
            if t >= n - 1:
                out[:, t] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        return _attach(mm, out)
    if fn == "std":
        # 定长窗口滑动的 (sum, sumsq)，两个累加器都是 (NC,) 的向量。
        # 方差 = E[x^2] - E[x]^2 在数值上有相消，这里数据是收益率量级（1e-2），
        # 相消误差远小于 float32 的精度，可以接受；换来的是不产生任何 (NC,T) 中间体。
        m2d = np.asarray(roll(x, n, "mean"), dtype=np.float64)
        xf = np.asarray(x, dtype=np.float64)
        mm, out = _buf(x.shape, np.float32)
        bad = np.isnan(xf)
        xz = np.where(bad, 0.0, xf)
        s1 = np.zeros(NC, dtype=np.float64)
        s2 = np.zeros(NC, dtype=np.float64)
        cnt = np.zeros(NC, dtype=np.float64)
        for t in range(T):
            if t >= n:
                v = xz[:, t - n]
                s1 -= v; s2 -= v * v
                cnt -= 1 - bad[:, t - n]
            v = xz[:, t]
            s1 += v; s2 += v * v
            cnt += 1 - bad[:, t]
            if t >= n - 1:
                N = np.maximum(cnt, 1)
                var = s2 / N - (s1 / N) ** 2
                out[:, t] = np.sqrt(np.where(cnt > 0, np.maximum(var, 0.0), np.nan))
        return _attach(mm, out)
    if fn in ("max", "min"):
        mm, out = _buf(x.shape, np.float32)
        for t in range(n - 1, T):
            win = x[:, t - n + 1:t + 1]
            out[:, t] = np.nanmax(win, axis=1) if fn == "max" else np.nanmin(win, axis=1)
        return _attach(mm, out)
    raise KeyError(fn)


_BUFS = []


def release_bufs():
    """把这一段的 mmap 全部放掉 —— 不放就是把 5251×1460 的块一直攒着。"""
    global _BUFS
    for mm, _a in _BUFS:
        try:
            mm.close()
        except Exception:                     # noqa: BLE001
            pass
    _BUFS = []


def _attach(mm, arr):
    _BUFS.append((mm, arr))                  # 持有引用，别让 mmap 被回收
    return arr


def corr_last(x, y, n):
    """最后一列的 n 日滚动相关（只算窗口末端，因子值只需要那一天）。"""
    xw = x[:, -n:]
    yw = y[:, -n:]
    mx = np.nanmean(xw, axis=1); my = np.nanmean(yw, axis=1)
    dx = xw - mx[:, None]; dy = yw - my[:, None]
    num = np.nansum(dx * dy, axis=1)
    den = np.sqrt(np.nansum(dx * dx, axis=1) * np.nansum(dy * dy, axis=1))
    return safe_div(num, den)


def _f32c(a, shape):
    mm, out = _buf(shape, np.float32)
    out[:] = a
    return _attach(mm, out)


def _safe_ratio(num_op, den_op, shape):
    """num/den，den<=0 或 nan 处给 nan。两遍路过 mmap，中间不产生任何 (NC,T) 的临时数组。

    这台机器上 numpy 的 malloc 每 n 次会崩一次（崩溃后整进程的分配都挂），
    所以凡是 (NC,T) 级别的数据都只走 _buf；临时量一律缩到 (NC,)。
    """
    mm, out = _buf(shape, np.float32)
    out[:] = np.nan
    CH = 256
    for t0 in range(0, shape[1], CH):
        t1 = min(t0 + CH, shape[1])
        num = np.asarray(num_op(t0, t1), dtype=np.float64)
        den = np.asarray(den_op(t0, t1), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        out[:, t0:t1] = r
    return _attach(mm, out)


def factor_set(C, O=None, A=None, MK=None):
    """Alpha158 里最常用的那批。返回前一律已经是 mmap 上的 float32。

    约束（这台机器实测）：numpy 的 malloc 每几十次会崩一次，崩了之后整进程的分配都挂。
    所以每个 (NC, T) 级别的数组都走 _buf（mmap + frombuffer），
    并且不写 `(C-O)/np.where(...)` 这种一行三个 58MB 临时体的表达式。
    """
    NC, T = C.shape
    F = {}
    sh = (NC, T)

    def shaped(fn):
        mm, out = _buf(sh, np.float32)
        fn(out)
        return _attach(mm, out)

    ret1 = shaped(lambda o: o.__setitem__(slice(None), np.nan))
    ret1[:, 1:] = (C[:, 1:] - C[:, :-1]) / np.where(C[:, :-1] > 0, C[:, :-1], np.nan)

    F["ret1"] = shaped(lambda o: o.__setitem__(slice(None), -ret1))
    for k in (5, 20, 60):
        def fill(o, k=k):
            o[:, :k] = np.nan
            o[:, k:] = -(C[:, k:] - C[:, :T - k]) / np.where(C[:, :T - k] > 0, C[:, :T - k], np.nan)
        F[f"ret{k}"] = shaped(fill)

    F["std20"] = -roll(ret1, 20, "std")
    F["std60"] = -roll(ret1, 60, "std")
    F["max20"] = -roll(ret1, 20, "max")

    if MK is not None and A is not None:
        turn = _safe_ratio(lambda a, b: A[:, a:b], lambda a, b: MK[:, a:b], sh)
        F["turn20"] = -roll(turn, 20, "mean")
        F["turn_abn"] = _safe_ratio(lambda a, b: roll(turn, 20, "mean")[:, a:b],
                                    lambda a, b: roll(turn, 250, "mean")[:, a:b], sh)
    if A is not None:
        la = shaped(lambda o: o.__setitem__(slice(None), np.log1p(np.maximum(A, 0))))
        F["amt20"] = -roll(la, 20, "mean")
        F["amt_std20"] = roll(la, 20, "std")

    if O is not None:
        hi = shaped(lambda o: o.__setitem__(slice(None), np.maximum(C, O)))
        lo = shaped(lambda o: o.__setitem__(slice(None), np.minimum(C, O)))
        F["kmid"] = _safe_ratio(lambda a, b: (C[:, a:b] - O[:, a:b]),
                                lambda a, b: O[:, a:b], sh)
        F["klen"] = _safe_ratio(lambda a, b: (hi[:, a:b] - lo[:, a:b]),
                                lambda a, b: O[:, a:b], sh)
        F["ksft"] = _safe_ratio(lambda a, b: ((C[:, a:b] - lo[:, a:b]) - (hi[:, a:b] - C[:, a:b])),
                                lambda a, b: (hi[:, a:b] - lo[:, a:b]), sh)
        low_n = roll(lo, 20, "min"); high_n = roll(hi, 20, "max")
        F["rsv"] = _safe_ratio(lambda a, b: (C[:, a:b] - low_n[:, a:b]),
                               lambda a, b: (high_n[:, a:b] - low_n[:, a:b]), sh)

    if A is not None:
        la = np.log1p(np.maximum(A, 0))
        mm, out = _buf(sh, np.float32)
        out[:] = np.nan
        # 滚动相关系数：仍然要窗口，但窗口只覆盖 20 列，且全部是普通 numpy 小数组。
        # 原来这里每个调仓日要开 130 个 20 列窗口 × 5251 行，是整轮最慢的一段。
        xw_all = np.asarray(C[:, :T], dtype=np.float64)
        yw_all = np.asarray(la[:, :T], dtype=np.float64)
        for t in range(19, T):
            xw = xw_all[:, t - 19:t + 1]
            yw = yw_all[:, t - 19:t + 1]
            mx = xw.mean(axis=1); my = yw.mean(axis=1)
            dx = xw - mx[:, None]; dy = yw - my[:, None]
            num = (dx * dy).sum(axis=1)
            den = np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
            out[:, t] = -num / np.where(den > 0, den, np.nan)
            if t >= T - 2:
                del xw, yw, dx, dy
        F["corr_amt20"] = _attach(mm, out)

    return F


def rebalance_idx(dates, start, freq="月"):
    si = next((i for i, d in enumerate(dates) if d >= start), 0)
    idx = [j for j in range(si + 1, len(dates)) if dates[j][:7] != dates[j - 1][:7]]
    return idx


def _flush_rows(rows, dates, rb, k, codes, NC, t0):
    """增量落盘：每算 5 个调仓日就把已有的行写进 npz。

    为什么不是最后写一次：这个进程会随时被杀，而"算完再存"意味着被杀就等于白算。
    分块存，被杀之后下一次能接着来。
    """
    if not rows:
        return
    names = list(rows[0])
    dst = os.path.join(CACHE, f"factors_part_{k+1}.npz")
    arr = {n: np.stack([r[n] for r in rows], axis=1) for n in names}
    np.savez_compressed(dst, dates=np.array([dates[j] for j in rb[:len(rows)]]),
                        codes=np.array(codes), **arr)


def cmd_compute(a):
    """按调仓日**一天一次**地算，不按时间段批量算。

    为什么不批量：批量时 SliceView 会把整块列都读进私有内存，一次几百 MB，
    这台机器上直接卡死/崩。按天算，每个因子只读自己那 260 天的窗口，
    峰值是几 MB 级别的窗口，跑得慢但跑得完 —— 慢比跑不完强。
    """
    t0 = time.time()
    try:
        import ctypes
        # BelowNormal：这个进程跑几分钟没人等，别和系统抢内存整理
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    os.makedirs(CACHE, exist_ok=True)
    dates, P = panel(("close", "open", "mktcap"))
    C = P["close"]; O = P.get("open"); MK = P.get("mktcap")
    A = None
    amt = os.path.join(HERE, "amt_flat.bin")
    if os.path.exists(amt):
        A = _mmap(amt, np.float64, C.shape)
    codes = codes_for_npz()[:C.shape[0]]
    NC = C.shape[0]
    print(f"[{time.time()-t0:5.1f}s] 面板 {C.shape} amt={'有' if A is not None else '无'}", flush=True)

    rb = rebalance_idx(dates, a.start)
    print(f"[{time.time()-t0:5.1f}s] 调仓日 {len(rb)} 个 {dates[rb[0]]} ~ {dates[rb[-1]]}", flush=True)
    WIN = 260
    rows = []                      # 每个调仓日一个 dict
    T0 = time.time()
    for k, j in enumerate(rb):
        t_day = time.time()
        s0 = max(0, j - WIN + 1)
        win = slice(s0, j + 1)
        sub = {key: (v[:, win] if v is not None else None) for key, v in
               (("C", C), ("O", O), ("A", A), ("MK", MK))}
        F = factor_set(sub["C"], sub["O"], sub["A"], sub["MK"])
        rows.append({n: np.array(F[n][:, -1], dtype=np.float32) for n in F})
        if len(rows) % 40 == 0:
            import gc; gc.collect()
        del F, sub
        release_bufs()
        if k % 5 == 0 or k == len(rb) - 1:
            print(f"[{time.time()-t0:6.1f}s] {k+1}/{len(rb)} {dates[j]}  "
                  f"本日 {time.time()-t_day:.1f}s", flush=True)
            _flush_rows(rows, dates, rb, k, codes, NC, t0)

    names = list(rows[0])
    dst = os.path.join(CACHE, "factors.npz")
    out = {}
    for n in names:
        blk = np.empty((NC, len(rb)), dtype=np.float32)     # 只多一块，不是 16 块
        for kki, r in enumerate(rows):
            blk[:, kki] = r[n]
        out[n] = blk
        rows_bak = rows
    np.savez_compressed(dst, dates=np.array([dates[j] for j in rb]),
                        codes=np.array(codes), **out)
    print(f"[{time.time()-t0:6.1f}s] -> {dst}  因子 {len(names)} × 调仓日 {len(rb)}", flush=True)
    return dst


def cmd_summarize(a):
    """用已有的增量结果算每个因子的 IC / ICIR，并和存量的 11 个因子放同一张表比。

    收益口径：调仓日 j 收盘 -> j+h 交易日收盘（h 默认 20），与 multifactor 的月频一致。
    """
    import glob
    parts = sorted(glob.glob(os.path.join(CACHE, "factors_part_*.npz")),
                   key=lambda p: int(os.path.basename(p).split("_")[-1][:-4]))
    if not parts:
        raise SystemExit("先跑 compute")
    src = parts[-1]                     # 取覆盖最长的那个分块
    z = np.load(src, allow_pickle=True)
    dates, P = panel(("close",))
    C = P["close"]
    di = {d: i for i, d in enumerate(dates)}
    rd = [str(d) for d in z["dates"]]
    rb = [di[d] for d in rd if d in di]
    names = [n for n in z.files if n not in ("dates", "codes")]
    h = a.horizon
    print(f"数据源 {os.path.basename(src)}：{len(names)} 个因子 × {len(rb)} 个调仓日 "
          f"({rd[0]} ~ {rd[-1]})，远期 {h} 日", flush=True)
    rows = []
    for n in names:
        M = z[n]
        if M.shape[1] != len(rd):
            continue
        ics = []
        for k, j in enumerate(rb):
            if j + h >= C.shape[1]:
                continue
            f = np.asarray(M[:, k], dtype=np.float64)
            prev = np.asarray(C[:, j]); cur = np.asarray(C[:, j + h])
            ok = (prev > 2.0) & (cur > 0) & np.isfinite(f)
            if ok.sum() < 100:
                continue
            r = cur[ok] / prev[ok] - 1.0
            fv = f[ok]
            rf = np.argsort(np.argsort(fv)).astype(np.float64)
            rr = np.argsort(np.argsort(r)).astype(np.float64)
            df = rf - rf.mean(); dr = rr - rr.mean()
            d = (df * df).sum() ** .5 * (dr * dr).sum() ** .5
            ics.append((df * dr).sum() / d if d else 0.0)
        if len(ics) < 8:
            rows.append({"factor": n, "n": len(ics), "ic": None})
            continue
        ic = np.array(ics); half = len(ic) // 2
        rows.append({"factor": n, "n": len(ic),
                     "ic": round(float(ic.mean()), 4),
                     "icir": round(float(ic.mean() / (ic.std(ddof=1) + 1e-12)), 3),
                     "t": round(float(ic.mean() / (ic.std(ddof=1) / len(ic) ** .5 + 1e-12)), 2),
                     "pos": round(float((ic > 0).mean()), 2),
                     "h1": round(float(ic[:half].mean()), 4),
                     "h2": round(float(ic[half:].mean()), 4),
                     "stable": bool(np.sign(ic[:half].mean()) == np.sign(ic[half:].mean()))})
    rows.sort(key=lambda r: -(abs(r["ic"]) if r.get("ic") is not None else -1))
    print(f"{'factor':<14}{'IC':>8}{'ICIR':>8}{'t':>7}{'pos':>6}{'1st':>8}{'2nd':>8}  stable")
    for r in rows:
        if r.get("ic") is None:
            print(f"{r['factor']:<14}{'n<8':>8}  n={r['n']}")
            continue
        print(f"{r['factor']:<14}{r['ic']:>8.4f}{r['icir']:>8.3f}{r['t']:>7.2f}"
              f"{r['pos']:>6.2f}{r['h1']:>8.4f}{r['h2']:>8.4f}  {'Y' if r['stable'] else 'N'}")
    rows = fdr(rows, alpha=a.alpha)
    surv = sum(1 for r in rows if r.get("survive"))
    tot = sum(1 for r in rows if r.get("q") is not None)
    print(f"\nBH-FDR(alpha={a.alpha})：{surv} / {tot} 个因子活下来")
    for r in sorted([r for r in rows if r.get("q") is not None], key=lambda r: r["q"]):
        print(f"   {r['factor']:<14} t={r['t']:>6.2f}  p={r['p']:<8} q={r['q']:<8} "
              f"{'通过' if r['survive'] else '不通过'}")
    write_json(os.path.join(CACHE, "summary.json"), rows)
    return rows


class A:                      # 见下：argparse 在这台机器上会崩，手写参数
    start = "2016-01-01"
    horizon = 20
    alpha = 0.05



# ---------------------------------------------------------------- 多重检验

def _norm_cdf(z):
    """标准正态 CDF，Abramowitz-Stegun 7.1.26，误差 <1.5e-7。不引 scipy。"""
    import math
    x = abs(z) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return 0.5 * (1.0 + y) if z >= 0 else 0.5 * (1.0 - y)


def fdr(rows, alpha=0.05):
    """Benjamini-Hochberg。给每个因子一个 q 值，并判断 alpha 下能否拒绝原假设。

    为什么必须有这一步：从 N 个因子里挑 |t| 最大的那个，N 越大越容易挑到纯噪音。
    16 个因子、t=4 看着很硬，但真问题不是"这个 t 大不大"，而是"N 个一起看还剩几个"。
    没有这一步，因子越多越自欺。
    """
    cand = [r for r in rows if r.get("t") is not None]
    m = len(cand)
    if not m:
        return rows
    for r in cand:
        p = 2.0 * (1.0 - _norm_cdf(abs(r["t"])))
        r["p"] = round(p, 5)
    order = sorted(cand, key=lambda r: r["p"])
    prev_q = 1.0
    for k in range(m - 1, -1, -1):          # 从最大的 p 往回收，保证 q 单调
        r = order[k]
        q = min(prev_q, r["p"] * m / (k + 1))
        r["q"] = round(q, 5)
        r["survive"] = bool(q <= alpha)
        prev_q = q
    for r in rows:
        r.setdefault("p", None)
        r.setdefault("q", None)
        r.setdefault("survive", None)
    return rows


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] not in ("compute", "summarize"):
        print("用法: python factor_lab.py compute [--start 2016-01-01] | summarize [--horizon 20]")
        return
    a = A()
    for i, x in enumerate(argv):
        if x == "--start":
            a.start = argv[i + 1]
        if x == "--horizon":
            a.horizon = int(argv[i + 1])
        if x == "--alpha":
            a.alpha = float(argv[i + 1])
    if argv[0] == "compute":
        cmd_compute(a)
    else:
        cmd_summarize(a)


if __name__ == "__main__":
    main()
