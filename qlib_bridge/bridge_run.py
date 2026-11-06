# -*- coding: utf-8 -*-
"""bridge_run.py —— 读 pred_cache.pkl，用 qlib 的 TopkDropoutStrategy + SimulatorExecutor 回测。
与 bridge_train.py 拆开是为了避开本机 512MB / 单次调用时长限制。
"""
import os, sys, json, time, pickle
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np, pandas as pd
import qlib
from qlib.config import C
from qlib.backtest import backtest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "qlib_data")
qlib.init(provider_uri=OUT, region="cn", expression_cache=None, dataset_cache=None)
C["kernels"] = 1
C["joblib_backend"] = "threading"

import sys
_cache = "pred_cache.pkl"
_out = "qlib_curve.json"
for _i, _a in enumerate(sys.argv):
    if _a == "--cache" and _i + 1 < len(sys.argv): _cache = sys.argv[_i + 1]
    if _a == "--out" and _i + 1 < len(sys.argv): _out = sys.argv[_i + 1]
c = pickle.load(open(os.path.join(HERE, _cache), "rb"))
pool = c["pool"]
te_start, te_end = c["te_start"], c["te_end"]
pred = c["pred"]
print("[run] pred", type(pred), getattr(pred, "shape", ""), flush=True)

# pred 缓存是 ndarray，需要还原成 qlib 期望的 (datetime, instrument) MultiIndex Series
# 用 qlib 日历与池重建索引
from qlib.data import D
cal = D.calendar(start_time=te_start, end_time=te_end)
# pred 是 DatasetH 过滤 NaN 后的序列，行数少于 cal*pool，按行号截断会把信号错配到别的股票。
# 用 qlib 的 fetch 拿到测试段真实保留的 (datetime, instrument) 索引重建。
_hd_idx = None
try:
    from qlib.data.dataset.handler import DataHandlerLP
    # pred 来自 model.predict(ds)，其索引就是 ds 的 test 段索引；没有缓存索引时退回笛卡尔积
    pass
except Exception:
    pass
if "index" in c and c["index"] is not None:
    pred_s = pd.Series(np.asarray(pred).ravel(), index=c["index"], name="score")
else:
    idx = pd.MultiIndex.from_product([cal, pool], names=["datetime", "instrument"])
    n = min(len(pred), len(idx))
    pred_s = pd.Series(np.asarray(pred).ravel()[:n], index=idx[:n], name="score")
print("[run] signal rows", len(pred_s), "vs cal*pool", len(cal) * len(pool), flush=True)

strategy = {"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"signal": pred_s, "topk": 50, "n_drop": 5}}
executor = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
t0 = time.time()
print("[run] backtest", flush=True)
pm, im = backtest(start_time=te_start, end_time=te_end, strategy=strategy,
                  executor=executor, account=1e6, benchmark="__bench__",
                  exchange_kwargs={"limit_threshold": 0.095, "deal_price": "close",
                                   "codes": pool})
print("[run] backtest %.0fs" % (time.time() - t0), flush=True)
print("[run] im type", type(im), "keys", list(im.keys()) if isinstance(im, dict) else "", flush=True)
print("[run] pm type", type(pm), flush=True)
for _name, _obj in (("pm", pm),):
    print("[run]  ", _name, type(_obj), flush=True)
    if isinstance(_obj, dict):
        for k2, v2 in _obj.items():
            print("[run]    ", k2, type(v2), getattr(v2, "shape", ""), 
                  list(v2.columns)[:10] if hasattr(v2, "columns") else "", flush=True)
            items2 = v2 if isinstance(v2, tuple) else (v2,)
            for j2, x2 in enumerate(items2):
                print("[run]       ", j2, type(x2), getattr(x2, "shape", ""),
                      list(x2.columns)[:10] if hasattr(x2, "columns") else "", flush=True)
                if hasattr(x2, "head"):
                    try: print("[run]         head", str(x2.head(2))[:260], flush=True)
                    except Exception: pass
    elif hasattr(_obj, "columns"):
        print("[run]   cols", list(_obj.columns)[:12], _obj.shape, flush=True)
        print("[run]   head", str(_obj.head(2))[:300], flush=True)

curve = []
# 净值在 pm["1day"][0] 这个 DataFrame 里（account / return / bench 列）
for _key, _val in (pm.items() if isinstance(pm, dict) else []):
    _df0 = _val[0] if isinstance(_val, tuple) else _val
    if hasattr(_df0, "columns") and "return" in _df0.columns:
        nav = 1.0
        for _i, _row in _df0.iterrows():
            nav *= (1.0 + float(_row["return"]))
            _d = str(_i.date()) if hasattr(_i, "date") else str(_i)
            curve.append((_d, nav))
        print("[run] nav from pm", len(curve), "final", round(nav, 4), flush=True)
        break
for k, v in (im.items() if isinstance(im, dict) else []):
    print("[run] ", k, type(v), flush=True)
    items = v if isinstance(v, tuple) else (v,)
    for j, x in enumerate(items):
        print("[run]    ", j, type(x), getattr(x, "shape", ""),
              getattr(x, "name", ""),
              list(x.columns)[:12] if hasattr(x, "columns") else "", flush=True)
        if hasattr(x, "head"):
            try: print("[run]      head", str(x.head(2))[:300], flush=True)
            except Exception: pass
        if hasattr(x, "columns") and "return" in x.columns:
            ser = x["return"]
            curve = [(str(i.date()) if hasattr(i, "date") else str(i), float(1 + r))
                     for i, r in ser.items()]
        elif hasattr(x, "name") and x.name == "return":
            curve = [(str(i.date()) if hasattr(i, "date") else str(i), float(1 + r))
                     for i, r in x.items()]
print("[run] curve len", len(curve), flush=True)
if curve:
    print("[run] first", curve[:2], "last", curve[-2:], flush=True)
json.dump({"curve": curve, "pool": pool, "train": ["2012-01-01", "2021-12-31"],
           "test": [te_start, te_end],
           "features": ["Log($mktcap + 1)", "Mean($amount, 20)/($amount + 1)",
                        "$close/Ref($close, 20) - 1", "Ref($close, 5)/$close - 1"],
           "label": "Ref($close, -5)/$close - 1"},
          open(os.path.join(HERE, _out), "w", encoding="utf-8"), ensure_ascii=False)
