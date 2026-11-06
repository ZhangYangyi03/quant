# qlib 原生回测一键过审（qlib_audit.py）

## 做了什么

`audit_backtest.py` 原本只能审计自写回测；`qlib_audit.py` 把它接到了 qlib 的
`DatasetH` + `model.predict()` 上：

- 从 handler 的 MultiIndex 重建 (T, N) 的 close/open/amount 面板，**不按行号截断**；
- 用 `model.predict()` 的原始 (datetime, instrument) 索引对齐信号，
  并自动过滤 qlib 塞进来的 `__bench__`；
- 把真实索引传给 L1 做显式核对（能抓"整体滚动一格"这类错配）；
- 可选跑 qlib 原生 `backtest()` 作对照，默认关闭（动态池上较慢，见下）。

## 2022 年实测：同一模型，两个数字

| 口径 | 净值 |
|---|---|
| qlib 原生 backtest（`TopkDropoutStrategy`, top50, 真实成本） | **1.3365** |
| 三层审计后可执行净值（O2O + 排除涨停/停牌） | **0.9415** |
| 差 | **-39.5pp** |

三层判定：`L1 索引核对通过`，`L2 前视不通过`（lag0 1.3512 vs lag1 1.1017），
`L3 不可买不通过`。也就是说 qlib 报的 +33.7% 扣掉三层假阳性后是 **-5.9%**。

复跑：

    python qlib_bridge/run_qlib_audit.py    # 三层审计，约 38s（pred 已缓存）
    python qlib_bridge/native_backtest_2022.py       # qlib 原生对照，约 19s
    # 或用库
    from qlib_audit import qlib_audit_dataset
    rep = qlib_audit_dataset(ds, model, ("2022-01-04","2022-12-30"),
                             topk=50, cost_bp=3.65, run_qlib_backtest=False)

## 已知坑

- `_signal_from_pred` 必须把 pred 索引统一成 `pd.Timestamp` 再 reindex，
  用 `str(d)[:10]` 字符串会与 `pd.to_datetime(cal)` 对不上，导致信号全 NaN。
- `run_qlib_backtest=True` 时 handler 构建 + 审计 + 原生回测串在一起会超时；
  原生对照拆成 `native_backtest_2022.py` 独立跑，19s 完成。
- 本机 `miniconda3` 的 numpy 导入即 OOM，全程用 `.qlibenv/Scripts/python.exe`。

## 意义

qlib 自己的 `qlib/contrib/evaluate/backtest.py` 和 `risk_analysis` 只算收益/回撤/IC，
不校验信号索引、时间对齐和不可买股票。`qlib_audit.py` 是补在这一层上的：
**任何用 qlib 跑出来的漂亮曲线，都该先过一遍这个 wrapper。**
