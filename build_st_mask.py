# -*- coding: utf-8 -*-
"""build_st_mask.py —— 从价格行为反推时变 ST 状态，替换「当前简称回填全历史」的假名单。

背景：东财 fund_parts 的 SECURITY_NAME_ABBR 是当前简称回填到所有历史 REPORT_DATE，
st_history.json 里 202/202 只股票全期同名，判为 UNUSABLE（st_history_verdict.json）。

原理：A 股 ST/*ST 股日涨跌幅限制 ±5%，非 ST 为 ±10%（创业板/科创板 20%，
本项目 universe 以主板为主）。用滚动窗口识别：
  窗口内 最大|日收益| <= 5.2%  且  |收益| 落在 4.7~5.3% 的天数 >= 2
=> 该窗口处于 ST 状态。只凭价格，不依赖任何外部名单，天然 PIT。

输出 st_mask.bin：uint8 (ND, NC)，1=ST，0=非 ST/数据不足；
    st_mask_meta.json：口径、覆盖率、按年统计、抽查。
"""
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ND, NC = None, None
W = 60          # 滚动窗口交易日
NEAR = 0.0485  # 接近 ±5% 的下界（收紧，抑制低波动股假阳性）
CAP = 0.052     # 允许的窗口最大 |收益|（容纳复权/取整噪声）
MIN_HITS = 6    # 窗口内至少几次触及 5% 限制


def main():
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ND, NC = len(dates), len(codes)
    fp = os.path.join(HERE, "m_compact.bin")
    with open(fp, "rb") as f:
        head = f.read(16)
    assert head[:8] == b"CMPC0001" and struct.unpack("<II", head[8:16]) == (ND, NC)
    M = np.memmap(fp, dtype="<f4", mode="r", offset=16, shape=(4, ND, NC))
    bad = np.memmap(os.path.join(HERE, "corrupt_mask.bin"), dtype=np.uint8,
                    mode="r", shape=(ND, NC))
    halt = np.memmap(os.path.join(HERE, "halt_mask.bin"), dtype=np.uint8,
                     mode="r", shape=(ND, NC))
    out = np.zeros((ND, NC), dtype=np.uint8)
    C = np.asarray(M[0], dtype=np.float64)       # close，46M float64 ~ 370MB，可能超内存
    # 改为逐列处理，避免整块 float64
    del C
    n_st_days = 0
    st_codes = set()
    for j in range(NC):
        c = np.asarray(M[0, :, j], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.diff(c) / c[:-1]
        r = np.where(np.isfinite(r) & (c[:-1] > 0) & (np.abs(r) < 0.5), r, np.nan)
        r[bad[1:, j] > 0] = np.nan
        r[halt[1:, j] > 0] = np.nan
        ar = np.abs(r)
        near = np.where(np.isfinite(ar) & (ar >= NEAR) & (ar <= 0.0515), 1.0, 0.0)
        big = np.where(np.isfinite(ar) & (ar > 0.09), 1.0, 0.0)
        # 用 cummax 的滑动版本：分段计算窗口最大，避免 4600 万次 Python 循环
        # 这里用 stride trick 构造 (n-W+1, W) 视图，分批算，内存可控
        n = len(r)
        if n < W:
            continue
        stride = r.strides[0]
        view = np.lib.stride_tricks.as_strided(r, shape=(n - W + 1, W),
                                              strides=(stride, stride))
        # 分块计算，避免一次 8770*60 float64 = 4MB 也没问题，直接算
        wmax = np.nanmax(np.where(np.isfinite(view), np.abs(view), 0.0), axis=1)
        wmin_ok = np.isfinite(view).sum(axis=1) >= 20   # 窗口至少 20 个有效交易日
        cs = np.concatenate([[0.0], np.cumsum(near)])
        cb = np.concatenate([[0.0], np.cumsum(big)])
        t = np.arange(W - 1, n)
        hits = cs[t + 1] - cs[t - W + 1]
        breaks = cb[t + 1] - cb[t - W + 1]
        flag = (hits >= MIN_HITS) & (breaks == 0) & (wmax <= CAP) & wmin_ok
        ids = t[flag] + 1
        out[ids, j] = 1
        v = out[:, j]
        n_st_days += int(v.sum())
        if v.sum() > 0:
            st_codes.add(codes[j])
    np.asarray(out).tofile(os.path.join(HERE, "st_mask.bin"))
    # 按年统计
    byyear = {}
    for y in range(1991, 2027):
        idx = [i for i, d in enumerate(dates) if d[:4] == str(y)]
        if idx:
            byyear[str(y)] = int(out[idx].sum())
    rep = {
        "method": "rolling 60d: max|ret|<=5.2% AND >=2 days in 4.7~5.3% AND no >9% day",
        "ND": ND, "NC": NC, "window": W, "near_lo": NEAR, "cap": CAP,
        "min_hits": MIN_HITS,
        "st_stock_days": n_st_days, "st_codes_ever": len(st_codes),
        "by_year_st_celldays": byyear,
        "note": ("只凭价格反推，天然 PIT；排除 corrupt_mask/halt_mask 日。"
                 "注册制后创业板/科创板 20% 限制不会被误判为 ST，但主板 ST 识别不受影响。"
                 "前复权除权日可能产生假的大幅收益，>9% 直接否决窗口，偏保守。"),
    }
    json.dump(rep, open(os.path.join(HERE, "st_mask_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("[st] codes ever ST", len(st_codes), "stock-days", n_st_days)
    for c in ("000010", "000001", "600000"):
        j = codes.index(c)
        v = out[:, j]
        yrs = sorted({dates[i][:4] for i in range(ND) if v[i]})
        print("[st]", c, "ST years", yrs)

if __name__ == "__main__":
    main()
