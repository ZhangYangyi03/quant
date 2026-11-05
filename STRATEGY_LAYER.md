# 策略层与执行层补齐（M162）

生成时间 2026-11-05T15:06:09。本文只报告本次改动，历史日志见 README.md / INSTITUTIONAL.md。

## 改了什么

原来两条收益叙事互相无法比较：README 的 EP 叙事、tca 的 22.7%→alpha 2.3%、
mlfit2 的 size+amt20 月度 1.27%，用的是不同面板、不同成本、不同基准、不同执行日；
执行层只有一个近似冲击成本，没有订单生命周期、盘前风控或成交回报对账。

现在 `strategy_layer.py` 把它们放进同一条管线：

    面板    m_compact.bin（5251 只 x 8816 日，close/open/amt/mktcap，真实市值）
    截面    fscan_parts 的 166 个唯一月末（fraw.meta_unique，已修重复落盘）
    信号    size+amt20 / EP(PIT 财务) / 同池同 k 固定哈希随机基准
    执行    S 月末收盘出信号 -> E=S+1 开盘成交，持有到下一 E 开盘换仓
    成本    quantlab.model.cost.compute_cost（佣金/过户/印花分段/校准冲击）
    风控    quantlab.execution.PreTradeRisk（单票4%/行业20%/ADV10%/涨跌停/现金/整手/T+1）
    撮合    quantlab.execution.PaperBroker + reconcile 逐笔对账
    归因    MKT/SMB/HML 自建可交易组合 + Newey-West HAC

新增文件：`strategy_layer.py`、`data_quality.py`、`survivorship_probe.py`、
`quantlab/execution/`（oms.py / broker.py）、`tests/test_execution_layer.py`、
`unified_result.json`、`order_plan_*.jsonl`、`fills_*.jsonl`、`corrupt_mask.bin`、
`halt_mask.bin`。测试从 94 个增加到 104 个，全绿。

## 中途发现并修掉的错误（比收益数字更重要）

源数据用 `fqt=1` 前复权日线。东方财富对部分长期停牌/重组股返回负价或近零价：
600585 在 2012-11-29 收 0.02、11-30 收 0.79（pct=+3850%）。`build_m.py` 用
`if v > 0` 把负价写成 0，面板里于是出现 0 -> 0.79 的假涨幅。size+amt20 恰好
专挑这类微盘股，第一版跑出 44.69% 年化，全是数据错误。

`data_quality.py` 把两类情况分开：

    corrupt  有成交但 open/close<=0、<1 元、或 |日收益|>22%（涨跌停上限20%+余量）
             共 168235 格，占 0.363%
    halt     amt<=0 或 close<=0（停牌/未上市/无成交），不是数据错误
             共 29929482 格，占 64.65%

严格口径永久剔除出现过 corrupt 的股票（clean 只用过去，无未来函数）；持仓遇到
halt 沿用最后已知价，遇到 corrupt 按 0 记市值但不产生现金、也不"卖出"。

## 统一后的结果（2012-09 ~ 2026-11，166 个月末，含全成本）

    组合                         CAGR    夏普   年化波动    MDD     成本bps   IS(bps)  对账失败
    size+amt20 · top50 · 100万        38.26%   1.20   32.22%   -51.13%     26.1     47.2  0
    同上 · 不永久剔除corrupt                38.67%   1.21   32.16%   -50.03%     26.4     47.3  0
    同上 · 前20%分位锚点                    30.47%   1.03   31.53%   -52.65%     11.4     40.6  0
    同上 · AUM 1亿                      29.33%   1.16   25.71%   -48.80%     36.8     51.2  0
    EP · top50 · 100万                17.48%   0.77   26.02%   -46.72%     12.1     27.2  0
    同池同k随机等权基准                       -0.31%   0.12   26.76%   -82.73%     10.1      6.1  0

严格 38.26% 与宽松 38.67% 只差 0.41pp，说明「永久剔除 corrupt」造成的
幸存者偏差不是收益的主要来源；`survivorship_probe.py` 可复算。

## 归因：alpha 没有看上去那么多

    组合                         年化alpha  NW-t   beta_MKT  beta_SMB  beta_HML   R2
    size+amt20 · top50 · 100万           9.41%   1.91     0.824     0.728     0.077  0.806
    同上 · 不永久剔除corrupt                   9.48%   1.83     0.825     0.735     0.084  0.799
    同上 · 前20%分位锚点                       4.64%   1.17     0.879     0.630     0.014  0.863
    同上 · AUM 1亿                         2.48%   0.51     0.701     0.531     0.290  0.753
    EP · top50 · 100万                   0.19%   0.05     0.750     0.035     0.804  0.735
    同池同k随机等权基准                         -3.88%  -1.36     0.789    -0.058     0.022  0.841

size+amt20 的 30pp 超额里，三因子解释掉大部分：R2=0.806，beta_MKT=0.824、
beta_SMB=0.728，残差 alpha 9.41%，Newey-West t=1.91（不到 2）。
EP 的 alpha 0.19%、t=0.05，基本被 beta_HML=0.804 解释干净。
AUM 1 亿时 alpha 掉到 2.48%、t=0.51。

分段：size+amt20 在 2013-2018 是 55.12%，2019-2026 是 27.56%；
EP 从 29.54% 掉到 9.56%。收益高度依赖早期小盘/壳行情。

## 执行层

`quantlab/execution/` 是订单生命周期，不是回测撮合器：

    目标权重 -> OrderPlan（先卖后买） -> PreTradeRisk -> PaperBroker/CsvBroker
             -> FillRecord -> reconcile（漏单/超量/重复/未知成交/拒单后成交）

两处硬约束：

- 决策价与执行价隔夜会跳空，盘前按收盘价过的现金约束必须按真实开盘价再过一遍，
  否则会撮合出现金为负的成交。`PaperBroker.send` 在整手网格上二分求解。
- 风控拒绝的单本来就不该有成交；`reconcile` 只把 APPROVED 漏单算异常，
  券商拒单归类为 CANCELLED。本次 7 条曲线、约 6 万笔订单，对账失败 0 次。

`order_plan_*.jsonl` / `fills_*.jsonl` 逐笔带 decision_price、exec_price、
fee 分项与 IS(bps)，可直接接 TCA 或券商回报。`CsvBroker` 导出人工复核用 CSV，
不联网、不下真单。

## 还欠什么（不假装做了）

- 没有 high/low，涨跌停用「开盘触板」近似，偏保守；没有真实一字板判定。
- 没有历史 ST 名单，ST 股未剔除。
- 财务公告日用法定披露截止日兜底，不是真实公告日。
- HML 用可见 EPS/收盘价作价值代理，不是账面市值比。
- 未做参数寻优的 Deflated Sharpe / PBO / placebo；`quantlab.validate` 里有
  检验器，但还没接到这条真实面板的管线上。
- 前复权数据本身仍带复权因子断层风险；根治要重拉不复权价 + 自行处理分红送转，
  成本很高，本次没做。
- 策略仍是样本内研究，不是可上线的 alpha：残差 alpha t<2、分段衰减明显、
  1 亿规模后 alpha 不显著。

## 复算

    python data_quality.py          # 生成 corrupt/halt 掩码
    python strategy_layer.py        # 主结果（约 40 秒）
    python survivorship_probe.py    # 宽/严口径对照
    python -m unittest discover -s tests -v
