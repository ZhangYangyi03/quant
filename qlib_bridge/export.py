# -*- coding: utf-8 -*-
"""export.py —— 把本项目 m_compact.bin 面板导出成 qlib 的本地数据目录。

不下载 qlib 自带行情：数据源仍是我们自己的 stockdata/*.csv。
qlib FileStorage 约定：
    qlib_data/calendars/day.txt        每行一个交易日 YYYY-MM-DD
    qlib_data/instruments/all.txt      每行 code<TAB>start<TAB>end
    qlib_data/features/<code>/<field>.day.bin
        float32 数组，第 0 个元素是 start_index（该股在 calendar 里的起始下标），
        其后为按 calendar 顺序的每日值。注意文件名不带 "$"。

字段：close / open / amount（成交额）/ mktcap（真实市值）。
另外把 corrupt/halt 掩码写进 qlib_data/quality/，供审计层复原「哪些价格不可信」。
"""
import os, sys, json, time
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from compact import Panel

OUT = os.path.join(HERE, "qlib_data")
FIELDS = {"close": "$close", "open": "$open", "amt": "$amount", "mktcap": "$mktcap"}


def write_feature(code, field, vals, start, base=None):
    d = os.path.join(base or os.path.join(OUT, "features"), code.lower())
    os.makedirs(d, exist_ok=True)
    fp = os.path.join(d, field.lstrip("$").lower() + ".day.bin")
    _v = np.asarray(vals, dtype=np.float64)
    _v = np.where(np.isfinite(_v), _v, np.nan)   # NaN 必须保留：corrupt/halt 日不能变成 0
    arr = np.hstack([start, _v]).astype("<f4")
    arr.tofile(fp)


def main():
    t0 = time.time()
    meta = json.load(open(os.path.join(ROOT, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ND, NC = len(dates), len(codes)
    for d in (os.path.join(OUT, "calendars"), os.path.join(OUT, "instruments"), os.path.join(OUT, "quality")):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(OUT, "calendars", "day.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(dates) + "\n")
    P = Panel()
    print("[export] %d stocks x %d days" % (NC, ND), flush=True)

    # 先算 span：用 amt>0 判定每只股票的有效区间，再逐字段写出。
    span = {}
    a_mat = np.empty((ND, NC), dtype=np.float32)
    for j in range(ND):
        a_mat[j] = P.col("amt", j)
    for ci, code in enumerate(codes):
        a = a_mat[:, ci]
        ok = np.flatnonzero(np.isfinite(a) & (a > 0))
        if len(ok):
            span[ci] = (int(ok[0]), int(ok[-1]))
    del a_mat
    print("[span] %d stocks with trades  %.0fs" % (len(span), time.time() - t0), flush=True)
    qmask = {}
    for nm in ("corrupt", "halt"):
        qmask[nm] = np.memmap(os.path.join(ROOT, nm + "_mask.bin"), dtype=np.uint8,
                              mode="r", shape=(ND, NC))
    print("[quality] masks loaded", flush=True)

    # 每份 (ND,NC) float32 占 185MB，绝不能同时驻留 4 份。
    for src in FIELDS:
        mat = np.empty((ND, NC), dtype=np.float32)
        for j in range(ND):
            mat[j] = P.col(src, j)
        print("[load %s] %.0fs, writing" % (src, time.time() - t0), flush=True)
        for ci, code in enumerate(codes):
            if ci not in span:
                continue
            s_, e_ = span[ci]
            seg = mat[s_:e_ + 1, ci].copy()
            if src in ("close", "open"):
                bad = (qmask["corrupt"][s_:e_ + 1, ci] > 0) | (qmask["halt"][s_:e_ + 1, ci] > 0)
                seg[bad] = np.nan
            write_feature(code, FIELDS[src], seg, s_)
            if (ci + 1) % 1000 == 0:
                print("  %s %d/%d %.0fs" % (src, ci + 1, NC, time.time() - t0), flush=True)
        del mat
        print("[done %s] %.0fs" % (src, time.time() - t0), flush=True)

    with open(os.path.join(OUT, "instruments", "all.txt"), "w", encoding="utf-8", newline="\n") as finst:
        for ci, code in enumerate(codes):
            if ci in span:
                s_, e_ = span[ci]
                finst.write("%s\t%s\t%s\n" % (code, dates[s_], dates[e_]))
        # qlib 回测基准：恒 1 的伪股票，不参与选股
        finst.write("__bench__\t%s\t%s\n" % (dates[0], dates[-1]))

    # 质量掩码：只落非零序列，供审计层读回
    for name in ("corrupt", "halt"):
        mm = np.memmap(os.path.join(ROOT, name + "_mask.bin"), dtype=np.uint8,
                       mode="r", shape=(ND, NC))
        for ci, code in enumerate(codes):
            vals = mm[:, ci]
            if not vals.any():
                continue
            write_feature(code, "$" + name, vals.astype(np.float64), 0,
                          base=os.path.join(OUT, "quality"))
    json.dump({"source": "m_compact.bin", "stocks": NC, "days": ND,
               "dates": [dates[0], dates[-1]], "fields": list(FIELDS.values()),
               "note": "价格仍为前复权；corrupt/halt 掩码在 qlib_data/quality/ 下"},
              open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
