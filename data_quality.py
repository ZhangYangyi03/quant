# -*- coding: utf-8 -*-
"""data_quality.py —— 区分「数据损坏」与「停牌/未上市」，前者永久剔除。

背景（2026-11-05）：源数据是 fqt=1 前复权日线，东方财富对部分长期停牌/重组股
返回负价或近零价：600585 在 2012-11-29 收 0.02、11-30 收 0.79（pct=+3850%）。
build_m.py 用 `if v > 0` 把负价写成 0，于是面板里出现 0 -> 0.79 的假涨幅。
任何按价格计价的回测若把这种票放进池子，就会凭空暴富。

但 `close=0 / amt=0` 还有两个完全正常的含义：尚未上市、当日停牌。
把它们一律当"坏数据"会让全市场 65% 的格子被剔除，且持仓在停牌日被错误清零。
本模块因此输出两张互斥的掩码：

  corrupt_mask.bin   数据损坏，永久剔除。命中条件（且当日 amt>0，即确有成交）：
                     open/close 非有限、<=0、< 1.0 元（前复权价近零），
                     或 |close/prev_close - 1| > 22%（A 股涨跌停上限 20% + 余量）
  halt_mask.bin      停牌/未上市/无成交，不是数据错误。命中条件 amt<=0 或 close<=0
                     且非 corrupt。持仓沿用最后已知价，不产生现金、不清零。

策略层的 clean[S] = 该股票在 S 及之前**从未** corrupt（只用过去，无未来函数）。
输出 data_quality.json 记录数量与样例。
"""
import os, json, struct, datetime
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ("close", "amt", "open", "mktcap")
CORRUPT = os.path.join(HERE, "corrupt_mask.bin")
HALT = os.path.join(HERE, "halt_mask.bin")


class P:
    def __init__(self, path):
        self.f = open(path, "rb", buffering=0)
        h = self.f.read(16)
        assert h[:8] == b"CMPC0001"
        self.ND, self.NC = struct.unpack("<II", h[8:16])
        m = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
        self.dates, self.codes = m["dates"], m["codes"]

    def block(self, name, j0, j1):
        n = (j1 - j0) * self.NC
        off = 16 + FIELDS.index(name) * self.ND * self.NC * 4 + j0 * self.NC * 4
        self.f.seek(off)
        raw = self.f.read(n * 4)
        assert len(raw) == n * 4
        return np.frombuffer(raw, dtype="<f4").reshape(j1 - j0, self.NC).astype(np.float32)


def main():
    p = P(os.path.join(HERE, "m_compact.bin"))
    ND, NC = p.ND, p.NC
    for path in (CORRUPT, HALT):
        with open(path, "wb") as f:
            f.seek(ND * NC - 1); f.write(b"\0")
    cm = np.memmap(CORRUPT, dtype=np.uint8, mode="r+", shape=(ND, NC))
    hm = np.memmap(HALT, dtype=np.uint8, mode="r+", shape=(ND, NC))
    prev_close = None
    n_near = n_jump = n_halt = 0
    examples = []
    CH = 200
    for j0 in range(0, ND, CH):
        j1 = min(ND, j0 + CH)
        C = p.block("close", j0, j1)
        O = p.block("open", j0, j1)
        A = p.block("amt", j0, j1)
        traded = np.isfinite(A) & (A > 0)
        halt = (~traded) | (C <= 0) | (O <= 0) | (~np.isfinite(C)) | (~np.isfinite(O))
        near = traded & ((C < 1.0) | (O < 1.0) | (~np.isfinite(C)) | (~np.isfinite(O)))
        # 跨块首日用上一块最后收盘做 prev
        if j1 - j0 > 1:
            pc = np.vstack([prev_close.reshape(1, NC) if prev_close is not None else C[:1], C[:-1]])
        else:
            pc = prev_close.reshape(1, NC) if prev_close is not None else C[:1]
        with np.errstate(all="ignore"):
            r = C / np.where(pc > 0, pc, np.nan) - 1.0
        jump = traded & np.isfinite(r) & (np.abs(r) > 0.22) & (pc > 0) & (C > 0)
        corrupt = near | jump
        corrupt &= ~halt           # 停牌/未上市优先，不算数据损坏
        n_near += int(near.sum()); n_jump += int(jump.sum()); n_halt += int(halt.sum())
        if len(examples) < 20:
            for a, b in np.argwhere(corrupt[:min(30, len(corrupt))]):
                jj = j0 + int(a)
                examples.append({"date": p.dates[jj], "code": p.codes[int(b)],
                                 "prev_close": float(pc[a, b]), "close": float(C[a, b]),
                                 "open": float(O[a, b]), "amt": float(A[a, b]),
                                 "ret_pct": round(float(r[a, b]) * 100, 1) if np.isfinite(r[a, b]) else None})
                if len(examples) >= 20: break
        cm[j0:j1] = corrupt.astype(np.uint8)
        hm[j0:j1] = (halt & ~corrupt).astype(np.uint8)
        prev_close = C[-1].copy()
        print("[%5d/%5d] corrupt=%d halt=%d" % (j1, ND, int(corrupt.sum()), int(halt.sum())), flush=True)
    cm.flush(); hm.flush(); del cm, hm
    tot_c = int(np.memmap(CORRUPT, dtype=np.uint8, mode="r", shape=(ND, NC)).sum())
    tot_h = int(np.memmap(HALT, dtype=np.uint8, mode="r", shape=(ND, NC)).sum())
    meta = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": "m_compact.bin（fqt=1 前复权）", "shape": [ND, NC],
            "corrupt_rules": ["amt>0 且 open/close<=0、<1.0 元或非有限",
                              "amt>0 且 |日收益|>22%（涨跌停上限20%+余量）"],
            "halt_rules": ["amt<=0 或 close<=0 且非 corrupt（停牌/未上市/无成交）"],
            "counts": {"near_zero_or_bad_price_cells": n_near, "jump_gt_22pct_cells": n_jump,
                       "corrupt_mask_cells": tot_c,
                       "corrupt_mask_pct": round(tot_c / (ND * NC) * 100, 3),
                       "halt_mask_cells": tot_h,
                       "halt_mask_pct": round(tot_h / (ND * NC) * 100, 2)},
            "examples": examples,
            "note": "corrupt 永久剔除；halt 只沿用最后已知价，不清零、不产生现金"}
    json.dump(meta, open(os.path.join(HERE, "data_quality.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("done", meta["counts"], flush=True)


if __name__ == "__main__":
    main()
