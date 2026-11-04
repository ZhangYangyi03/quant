# -*- coding: utf-8 -*-
"""fcalc.py -- 单个因子单次计算（被外部计时器按秒杀掉也不会丢已完成部分）。

和 conv.py 分开的原因：这台机器编译超过约 350 行的文件就 MemoryError，
所以入口各自一个文件。
"""
import os, sys, json, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compact import Panel          # noqa: E402
from ops158 import *              # noqa: E402,F401,F403
from factor158 import build_registry  # noqa: E402

SEG_MONTHS = 8
SKIP = {"MOMR30", "MOMR60"}   # 这两个因子曾由旧 extern 注册表提供，已不再存在
OUT = os.path.join(HERE, "alpha_parts")
os.makedirs(OUT, exist_ok=True)


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
        REG = build_registry()
        for i, (name, fn) in enumerate(REG):
            fp = os.path.join(OUT, "%s_%05d.npy" % (name, sid))
            if os.path.exists(fp) or name in SKIP:   # 断点续算：已落盘或已知不再产出的因子跳过
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
        REG = build_registry()
        for i, (nm, fn) in enumerate(REG):
            fp = os.path.join(OUT, "%s_%05d.npy" % (nm, sid))
            if os.path.exists(fp) or nm in SKIP:
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
