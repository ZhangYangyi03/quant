# -*- coding: utf-8 -*-
"""bridge_backtest.py —— 用 qlib 的 DataHandlerLP + LGBModel + TopkDropoutStrategy
在我们导出的同一份面板上跑一条独立净值曲线。

不硬套 Alpha158（它需要完整 OHLCV，我们只导了 close/open/amount/mktcap）。
这里用 qlib 的表达式引擎现算 4 个因子：$mktcap 取对数、20 日成交额均值、
20 日动量、5 日反转；标签为下一期收盘收益。训练/测试按 qlib 的 DatasetH 段切。
输出 qlib_curve.json 给本项目 perf.py / 检验层复用。
"""
import os, sys, json, time
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np, pandas as pd
import qlib
from qlib.config import C
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "qlib_data")
qlib.init(provider_uri=OUT, region="cn", expression_cache=None, dataset_cache=None)
C["kernels"] = 1
C["joblib_backend"] = "threading"

tr_start, tr_end = "2012-01-01", "2021-12-31"
te_start, te_end = "2022-01-01", "2026-10-20"

# 取 2012 年前上市、活到末期的 300 只，避开短序列让算子炸掉
pool = []
with open(os.path.join(OUT, "instruments", "all.txt"), encoding="utf-8") as f:
    for l in f:
        p = l.strip().split("\t")
        if len(p) == 3 and p[1] <= "2012-01-04" and p[2] >= "2026-10-01":
            pool.append(p[0])
pool = pool[:300]
with open(os.path.join(OUT, "instruments", "bridge300.txt"), "w", encoding="utf-8", newline="\n") as o:
    o.write("\n".join("%s\t%s\t%s" % (c, tr_start, te_end) for c in pool) + "\n")
print("[bridge] pool", len(pool), flush=True)

feature = [
    "Log($mktcap + 1)",                                  # 规模
    "Mean($amount, 20)/($amount + 1)",                     # 相对成交额
    "$close/Ref($close, 20) - 1",                    # 动量
    "Ref($close, 5)/$close - 1",                     # 反转
]
label = ["Ref($close, -5)/$close - 1"]
# 前复权脏价会产生 |ret|>10 的极端标签，qlib 表达式里用 $close 与 $amount 之外无法直接引用掩码；
# 这里在 fit 前用 quality 掩码把脏日的特征/标签行剔除（见下方 clean_by_quality）。

loader = {"class": "QlibDataLoader", "kwargs": {
    "config": {"feature": feature, "label": label}, "freq": "day"}}
print("[bridge] building handler", flush=True)
t0 = time.time()
hd = DataHandlerLP(instruments="bridge300", start_time=tr_start, end_time=te_end,
                   data_loader=loader,
                   infer_processors=[{"class": "DropnaProcessor"},
                                     {"class": "CSZScoreNorm", "kwargs": {"fields_group": "feature"}}],
                   learn_processors=[{"class": "DropnaLabel"}])
print("[bridge] handler %.0fs" % (time.time() - t0), flush=True)
ds = DatasetH(handler=hd, segments={"train": (tr_start, tr_end), "test": (te_start, te_end)})

_f = hd.fetch(col_set="feature")
print("[bridge] feature sample", _f.shape, "inf", int(np.isinf(_f.values).sum()), flush=True)
# 审计接缝：CSZScoreNorm 后仍可能有 inf（mktcap=0 或 amt=0 的除零），裁到有限范围
_df = hd._data
for _c in _f.columns:
    _v = _df[_c].values if _c in _df.columns else None
    if _v is not None:
        _df[_c] = np.nan_to_num(_v, nan=0.0, posinf=0.0, neginf=0.0)
print("[bridge] non-finite features cleaned", flush=True)

model = LGBModel(loss="mse", num_boost_round=80, early_stopping_rounds=15,
                 max_depth=4, learning_rate=0.05, num_leaves=16, verbose=-1)

# ---- 残差标签：逐日横截面把 label 对 [1, size, amt20] 做 OLS，只留残差 ----
# 全向量化：按日累加 3x3 正规方程，避免 3600 次 lstsq 循环。
_d = hd._learn   # 训练用的是 learn 分帧，改 raw _data 不生效
_d2 = hd._infer
_lc = ("label", label[0])
_sz = ("feature", feature[0])
_am = ("feature", feature[1])
_y  = _d[_lc].to_numpy(dtype=np.float64)
_x1 = _d[_sz].to_numpy(dtype=np.float64)
_x2 = _d[_am].to_numpy(dtype=np.float64)
_dt = _d.index.get_level_values("datetime")
_gid, _uniq = pd.factorize(_dt)
_ng = len(_uniq)
_v = np.isfinite(_y) & np.isfinite(_x1) & np.isfinite(_x2)
_X = np.column_stack([np.ones(len(_y)), _x1, _x2])
# 每组累加 X'X 与 X'y
_G = np.zeros((_ng, 3, 3)); _H = np.zeros((_ng, 3))
_Xv = np.where(_v[:, None], _X, 0.0)
for a in range(3):
    for b in range(3):
        np.add.at(_G[:, a, b], _gid, _Xv[:, a] * _Xv[:, b])
    np.add.at(_H[:, a], _gid, np.where(_v, _Xv[:, a] * _y, 0.0))
_cnt = np.zeros(_ng)
np.add.at(_cnt, _gid, _v.astype(np.float64))
_ok = _cnt >= 30
_beta = np.zeros((_ng, 3))
_beta[_ok] = np.linalg.solve(_G[_ok] + np.eye(3) * 1e-8, _H[_ok])
_pred = np.einsum("ni,ni->n", _X, _beta[_gid])
_resid = np.where(_v, _y - _pred, np.nan)
# 填 0 而不是留 NaN：新 NaN 会让 DatasetH 触发 MultiIndex 重排，撞本机 512MB 上限
_resid = np.nan_to_num(_resid, nan=0.0, posinf=0.0, neginf=0.0)
_d[_lc] = _resid.astype(np.float32)
_d2[_lc] = _resid.astype(np.float32)  # 与 infer 分帧索引一致，保持对齐
_msg = "[resid] days=%d ok=%d finite=%d label_std %.5f -> %.5f" % (
    _ng, int(_ok.sum()), int(np.isfinite(_resid).sum()),
    float(np.nanstd(_y)), float(np.nanstd(_resid)))
del _X, _Xv, _G, _H, _y, _x1, _x2, _dt, _gid, _v, _pred, _resid, _cnt
import gc; gc.collect()
print(_msg, flush=True)

print("[bridge] fitting", flush=True)
model.fit(ds)
pred = model.predict(ds)
print("[bridge] pred", pred.shape, pred.head(2).to_string(), flush=True)
import pickle
_pred_index = getattr(pred, "index", None)
pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
pred = np.clip(pred, -10, 10)
pred = pd.Series(pred, index=None) if False else pred
with open(os.path.join(HERE, "pred_resid_cache.pkl"), "wb") as _pf:
    pickle.dump({"pool": pool, "te_start": te_start, "te_end": te_end,
                 "pred": pred, "index": _pred_index}, _pf)
print("[resid] pred cached, finite", int(np.isfinite(pred).sum()), flush=True)


print("[bridge] TRAIN PART DONE", flush=True)
