# -*- coding: utf-8 -*-
"""qlib_audit.py —— 让任意 qlib 策略/模型一键过三层审计。

用法：
    from qlib_audit import qlib_audit
    rep = qlib_audit(handler=hd, model=m, segment=("2022-01-04","2022-12-30"),
                     topk=50, cost_bp=3.65)
    print(rep["qlib_nav"], rep["audit"]["executable_alpha_nav"])

它做三件事：
  1. 从 handler 的MultiIndex 里取出真实 (datetime, instrument) 索引，
     重建 (T, N) 价格/开盘/成交额面板，避免按行号截断；
  2. 用 model.predict 的原始索引（不是 np.asarray 后丢索引）对齐信号；
  3. 调 audit_backtest.audit() 出三层判定，同时跑 qlib 原生 backtest 作对照。
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_backtest import audit

FIELDS = {"close": "$close", "open_": "$open", "amount": "$amount"}


def _panel_from_handler(hd, cal, codes, fields=("close", "open_", "amount")):
    """从 qlib handler 取 (T,N) 面板。优先用 handler._data；不同 qlib 版本布局不同，
    失败时回退到 D.features() 逐字段取，保证跨平台。"""
    idx = pd.to_datetime(cal)
    out = {}
    try:
        df = hd._data
        for f in fields:
            col = FIELDS[f]
            # qlib 的 feature 列名可能是 ('feature', '$close') 或直接 '$close'
            candidates = [c for c in df.columns if (isinstance(c, tuple) and c[-1] == col) or c == col]
            if not candidates:
                raise KeyError(col)
            s = df[candidates[0]]
            out[f] = s.unstack("instrument").reindex(index=idx, columns=codes).to_numpy(dtype=np.float64)
        return out
    except Exception:
        pass
    from qlib.data import D
    for f in fields:
        col = FIELDS[f]
        df = D.features(codes, [col], start_time=str(cal[0]), end_time=str(cal[-1]), freq="day")
        df.columns = [col] * len(df.columns)
        out[f] = df[col].unstack("instrument").reindex(index=idx, columns=codes).to_numpy(dtype=np.float64)
    return out


def _signal_from_pred(pred, cal, codes):
    """保留 pred 的原始 MultiIndex，重建 (T,N) 信号网格，并返回显式索引供 L1 核对。"""
    if isinstance(pred, pd.Series):
        s = pred.copy()
    elif isinstance(pred, pd.DataFrame):
        s = pred.iloc[:, 0].copy()
    else:
        raise TypeError("model.predict() 必须返回带 (datetime,instrument) MultiIndex 的 Series/DataFrame")
    s.index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(d).normalize(), str(c)) for d, c in s.index],
        names=["datetime", "instrument"])
    s = s[~s.index.get_level_values("instrument").isin(["__bench__"])]
    idx = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in cal])
    grid = s.unstack("instrument").reindex(index=idx, columns=codes)
    expected = [(str(pd.Timestamp(d))[:10], str(c)) for d in cal for c in codes]
    return grid.to_numpy(dtype=np.float64), expected, s.index


def qlib_audit_dataset(dataset, model, segment, topk=50, cost_bp=3.65,
                       account=1_000_000, run_qlib_backtest=True,
                       exchange_kwargs=None):
    """dataset: qlib DatasetH；model: 已 fit 或可 fit 的模型。"""
    hd = dataset.handler
    # 只取 segment 范围内的日历与股票池
    df = hd._data
    dt = df.index.get_level_values("datetime")
    mask = (dt >= pd.Timestamp(segment[0])) & (dt <= pd.Timestamp(segment[1]))
    sub = df.loc[mask]
    cal = sorted({str(d)[:10] for d in sub.index.get_level_values("datetime")})
    codes = sorted({str(c) for c in sub.index.get_level_values("instrument")
                    if str(c) != "__bench__"})
    panel = _panel_from_handler(hd, cal, codes)
    pred = model.predict(dataset)
    SIG, expected, raw_index = _signal_from_pred(pred, cal, codes)
    rep = audit(SIG, panel["close"], panel["open_"], panel["amount"],
                cal, codes, topk=topk, cost_bp=cost_bp,
                signal_index=expected)
    out = {
        "segment": list(segment), "T": len(cal), "N": len(codes),
        "signal_pairs": int(len(raw_index)),
        "expected_pairs": len(cal) * len(codes),
        "audit": rep,
    }
    if run_qlib_backtest:
        try:
            nav = _qlib_native_nav(pred, segment, codes, topk, account, exchange_kwargs)
            out["qlib_nav"] = nav
            out["nav_gap"] = round(float(nav - rep["executable_alpha_nav"]), 4)
        except Exception as e:
            out["qlib_nav_error"] = str(e)[:200]
    return out


def _qlib_native_nav(pred, segment, codes, topk, account, exchange_kwargs):
    from qlib.backtest import backtest
    kw = dict(limit_threshold=0.095, deal_price="close", codes=codes,
              open_cost=0.0003, close_cost=0.0013, min_cost=0.0, impact_cost=0.0)
    if exchange_kwargs:
        kw.update(exchange_kwargs)
    strategy = {"class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"signal": pred, "topk": topk, "n_drop": max(1, topk // 10),
                           "only_tradable": True}}
    executor = {"class": "SimulatorExecutor",
                "module_path": "qlib.backtest.executor",
                "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}
    pm, _ = backtest(start_time=segment[0], end_time=segment[1],
                     strategy=strategy, executor=executor, account=account,
                     benchmark="__bench__", exchange_kwargs=kw)
    d = pm["1day"][0]
    return round(float((1 + d["return"]).prod()), 4)
