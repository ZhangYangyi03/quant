# -*- coding: utf-8 -*-
"""fscan.py -- 一次只算一个时段，安全落盘。

为什么要分时段
--------------
整个 196 个月 × 5251 只 × 11 个因子 = 45MB 的因子矩阵，在这台机器上（512MB 提交
上限 + 碎片）一次性算完再保存必然 MemoryError。所以：
  * 每次只装 60 个调仓日的因子（11 个 2.5MB 的矩阵，合计 20MB）
  * 每个矩阵单独写成一个 .npy（np.save 逐块写，不需要整份缓冲区）
  * 输出目录 fscan_parts/，文件名 <因子>_<起始月序号>.npy

用法：
  python fscan.py 2017-01-01 2022-01-01     # 算这一段
  python fscan.py 2017-01-01 2022-01-01 8   # 每 8 个月落一次盘（默认 8）
"""
import os, sys, json, math, time, glob
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compact import Panel
from fastpanel import FACTORS, rolling_features

PARTS = os.path.join(HERE, "fscan_parts")


def main(argv):
    f0 = argv[1] if len(argv) > 1 else "2010-01-01"
    f1 = argv[2] if len(argv) > 2 else "2099-01-01"
    CH_M = int(argv[3]) if len(argv) > 3 else 8
    os.makedirs(PARTS, exist_ok=True)
    p = Panel()
    dates = p.dates
    si = next(i for i, d in enumerate(dates) if d >= "2010-07-01")
    rb = [j for j in range(si, p.ND - 1) if dates[j + 1][:7] != dates[j][:7]]
    sel = [k for k, j in enumerate(rb) if f0 <= dates[j] < f1]
    if not sel:
        print("区间内没有调仓日:", f0, f1)
        return
    print("本段 %d 个月: %s ~ %s" % (len(sel), dates[rb[sel[0]]], dates[rb[sel[-1]]]), flush=True )
    t0 = time.time()
    PRE, SCAN = 260, 120
    c0 = max(0, rb[sel[0]] - PRE)
    end_j = rb[sel[-1]]
    buf_date, buf_close, buf_alive = [], [], []
    bufs = {k: [] for k in FACTORS}
    seg_id = [len(glob.glob(os.path.join(PARTS, 'meta_*.npy')))]  # 续跑时从已有段号接着编
    seen = set()          # 后一块会和前一块重叠 PRE 天，不加这个会把早先的调仓日再算一遍

    def flush():
        if not buf_date:
            return
        print("  flush 开始 (%d 个月)" % len(buf_date), flush=True)
        d0 = os.path.join(PARTS, "meta_%05d.npy" % seg_id[0])
        np.save(d0, np.array([int(x) for x in buf_date]))
        np.save(os.path.join(PARTS, "close_%05d.npy" % seg_id[0]), np.vstack(buf_close))
        np.save(os.path.join(PARTS, "alive_%05d.npy" % seg_id[0]), np.vstack(buf_alive))
        for k in FACTORS:
            np.save(os.path.join(PARTS, "%s_%05d.npy" % (k, seg_id[0])), np.vstack(bufs[k]))
        print("  落盘段 %d (%d 个月, %.1fs)" % (seg_id[0], len(buf_date), time.time() - t0), flush=True)
        seg_id[0] += 1
        print("  flush 完成", flush=True)
        buf_date.clear(); buf_close.clear(); buf_alive.clear()
        for k in FACTORS:
            bufs[k].clear()

    in_sel = set(sel)
    rb_pos = {j: k for k, j in enumerate(rb)}
    while c0 <= end_j:
        c1 = min(end_j + 1, c0 + SCAN + PRE)
        F = rolling_features(c0, c1, p)
        for i, j in enumerate(range(c0, c1)):
            k = rb_pos.get(j)
            if k is None or k not in in_sel or j in seen:
                continue
            seen.add(j)
            buf_date.append(j)
            buf_close.append(np.array(F["_close"][i]))
            al = F["_close"][i] > 2.0
            buf_alive.append(al)
            for name in FACTORS:
                bufs[name].append(np.array(F[name][i]))
            if len(buf_date) >= CH_M:
                flush()
        del F
        print("[%6.1fs] 扫描到 %s" % (time.time() - t0, dates[c1 - 1]), flush=True)
        if c1 >= end_j + 1:
            break
        c0 = c1 - PRE
    flush()
    np.save(os.path.join(PARTS, "rb_all.npy"), np.array(rb))
    print("段数 %d，完成 %.0fs" % (seg_id[0], time.time() - t0))


if __name__ == "__main__":
    main(sys.argv)
