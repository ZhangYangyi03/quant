# -*- coding: utf-8 -*-
"""build_hl.py —— 从 stockdata/*.csv 提取调仓执行日的 high/low，供一字板判定。

只落 166 个执行日 x 5251 只的 float32 矩阵（约 3.5MB x2），不重建整张面板。
输出 hl_exec.npz：keys = dates（执行日字符串数组）、high、low。
"""
import os, json, csv, sys
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import fraw
M = fraw.meta_unique()
S = [int(x) for x in M["dates"]]
meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
D, C = meta["dates"], meta["codes"]
E = [min(s + 1, len(D) - 1) for s in S]
want = {D[e]: i for i, e in enumerate(E)}
H = np.full((len(E), len(C)), np.nan, dtype=np.float32)
L = np.full_like(H, np.nan)
code_idx = {c: i for i, c in enumerate(C)}
for n, c in enumerate(C):
    fp = os.path.join(HERE, "stockdata", c + ".csv")
    ci = code_idx[c]
    with open(fp, encoding="utf-8") as f:
        rd = csv.reader(f); next(rd, None)
        for r in rd:
            k = want.get(r[0])
            if k is None or len(r) < 5:
                continue
            try:
                H[k, ci] = float(r[3]); L[k, ci] = float(r[4])
            except ValueError:
                pass
    if (n + 1) % 1000 == 0:
        print("[%.0f%%] %d" % ((n + 1) / len(C) * 100, n + 1), flush=True)
np.savez_compressed(os.path.join(HERE, "hl_exec.npz"),
                    dates=np.array([D[e] for e in E]), high=H, low=L)
print("done", H.shape, "finite high", int(np.isfinite(H).sum()), flush=True)
