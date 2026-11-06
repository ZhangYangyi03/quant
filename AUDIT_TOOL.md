# audit_backtest：回测可信度审计工具

## 它解决什么

qlib / backtrader / vnpy / Lean 都校验行情、校验成本，但**没有一家**校验下面三件事。
本项目自己踩过这三个坑，每次都把"看起来很强的策略"削掉一大块：

| 层 | 假阳性来源 | 本项目实测 |
|---|---|---|
| L1 | 信号索引错配：pred 行数 < calendar x universe，按行号截断静默配错股票 | 净值 1.1298 -> 4.2774 |
| L2 | 前视：当日收盘信号赚当日收益；幸存者偏差：池子只含活到末期的股票 | 2022 +34.7% -> +10.2% |
| L3 | 不可买股票：涨停/停牌股按收盘价成交，而它们贡献几乎全部超额 | 2022 +10.2% -> -5.5% |

## API

    from audit_backtest import audit, multi_year_verdict

    r = audit(signal=SIG, close=C, open_=O, amount=A,
              cal=TD, codes=pool, topk=50, cost_bp=3.65,
              signal_index=EXPECTED_INDEX)     # 强烈建议传
    print(r["verdict"], r["fails"], r["executable_alpha_nav"])

    v = multi_year_verdict([(2022, r22), (2023, r23), ...])
    print(v["verdict"])

`signal` / `close` / `open_` / `amount` 都是 (T, N) numpy 数组，
NaN 表示缺失；`signal_index` 是 (datetime, instrument) 二元组列表，
传入后 L1 会逐格核对是否落在 (cal x codes) 网格上、有无重复/越界。

L1 不传索引是**降级模式**：统计特征抓不到"整体滚动一格"这类错配，
结果里会明确写 `index_checked: false` 和警告，不会假装通过。

三层各自只判"回测是否可信"，不判"策略是否赚钱"——
随机数据净值低于 1 是可信的回测，不该被误判为不可信。

L2 另加**统计指纹**：逐 lag 算信号与未来收益的横截面 RankIC。
真 alpha 在 lag>=1 仍有 IC 且衰减平缓；只有 lag=0 强、lag=1 塌到近 0
（`ic_collapse=True`）说明信号含当日价格信息，即使没传索引也能判前视。
合成数据里完美前视的 RankIC 是 lag0 1.0 -> lag1 -0.0003。
本项目真实五年数据 RankIC 稳定在 0.04~0.06 且衰减平缓，`ic_collapse=False`，
说明 lag=0 的虚高主要来自 L3 的涨停不可买，不是信号偷看——两层互相印证。

`multi_year_verdict()` 加跨年度判据：单年通过不算数，
要求所有年三层都过、多数年为正、且去掉最好一年后合计仍 > 1。

## 一键复跑

    # 逐年训练（DY_YEAR=2022..2026）
    DY_YEAR=2022 python qlib_bridge/train_dyn_year.py
    # 逐年三层审计
    DY_YEAR=2022 python qlib_bridge/audit_year.py
    # 跨年度判定
    python audit_all.py
    # 合成数据单元测试（8 个用例）
    python test_audit_backtest.py

## 当前判定

    python audit_all.py
    -> 无稳定可执行 alpha：2/5 年为正，去掉 2025 后合计 0.8642

逐年可执行净值（O2O + 排除涨停/停牌 + 3.65bp 成本）：
2022 0.9451 / 2023 1.2014 / 2024 0.7948 / 2025 1.6189 / 2026 0.9576。

## 单元测试覆盖（test_audit_backtest.py）

    A 干净随机数据        -> 三层通过
    B 信号整体滚动一格    -> 降级模式明确警告无法可靠识别
    C 完美前视信号        -> L2 不通过（lag0 净值 1099 vs lag1 0.98）
    D 涨停股虚假成交      -> L3 不通过（过滤掉水 0.174）
    E 跨年度 2/5 正       -> 无稳定 alpha
    F 显式索引错配        -> L1 不通过
    G 显式索引正确        -> 三层通过
    H 完美前视 IC 指纹    -> lag0 RankIC 1.0, lag1 -0.0003, ic_collapse=True

注意：本机 `miniconda3` 的 numpy 导入即 OOM，用 `.qlibenv/Scripts/python.exe` 跑。
