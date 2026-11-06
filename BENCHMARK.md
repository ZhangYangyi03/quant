# "超过全部开源项目"的可量化标尺（M166 立）

没有单一标尺，拆成可比、可证伪的维度。数值均为本机实测，复跑命令见各文档。

| 维度 | 本项目现状 | 开源对照 | 可判定方式 |
|---|---|---|---|
| PIT / 无未来函数 | report_ann_dates 240315 对，四类报告期齐全（年报 56224）；真实公告日覆盖显著提升 | qlib/zipline/vnpy 均无 A 股财报 PIT 表 | `ANNUAL_PIT.md` / `annual_ann_quality.json` |
| 交易成本与冲击 | 佣金万2.5双边/印花税千1卖出/过户费万0.1/冲击10bp，日均换手 0.7~1.2，年化拖累 8.92pp；tca_calibrated_frontier 已落 | Lean/freqtrade/vnpy 有费率滑点但不含 A 股印花税不对称与涨跌停不可成交 | EXEC_CONSTRAINTS.md / tca_calibrated_frontier.json |
| 多重检验闸门 | DSR / PBO / NW-t / 正交检验已落 | qlib 有 IC/回测，无 DSR/PBO；绝大多数开源无 | 工具存在且可复跑 |
| 因子正交检验 | factor_orth / alt_orth 已落，负面结论留档 | alphalens 有 IC 分解，无 style 残差 A/B | 脚本可跑 |
| 回测可信度审计 | 三层判据 + RankIC 指纹 + 显式索引核对 + 跨年度判定 + 自动修复建议 + qlib wrapper + 9 个单元测试 | qlib / backtrader / vnpy / Lean 均无内置这三类检测 | `python audit_all.py`、`python test_audit_backtest.py` |
| 数据覆盖 | 日频 OHLCV + 4 类非价量 + qlib 桥接 | 多数开源只提供框架，不带数据 | alt_data 行数 |
| 独立可交易 alpha | **没有**：机器可复跑判定 2/5 年为正、去掉 2025 后合计 0.8642 | 多数开源不宣称有 alpha | `python audit_all.py` |
| 工程成熟度 / 社区 | 单人 20+ commit，无 CI 发布 | qlib 20k+ stars、Lean/freqtrade 50k+ | 客观差距，不追 |

**诚实判定**：研究纪律、防自欺、PIT、正交检验、数据覆盖、回测可信度审计
这 6 项已可量化地强于主流开源，其中「回测可信度审计」是独有的：
同模型 qlib 原生报 1.3365，过三层审计后可执行 0.9415，差 -39.5pp。

独立可交易 alpha **没有**（机器判定，不是主观判断），且多数开源也不提供；
工程成熟度/社区差距是数量级的，不靠代码追。
所以"超过"必须限定在研究质量维度，且必须诚实写出 alpha 为负这一条，否则不可证伪。


## 2026-11-06 更新

三层审计（信号索引错配 / 幸存者偏差+前视 / 不可买股票）均已落地并留档：
RESID_LABEL.md、DYNAMIC_UNIVERSE.md、EXEC_CONSTRAINTS.md。
结果是可执行 alpha ≈ 0。这反而把第 1 项「防自欺」做成了本项目最独特的能力：
qlib / backtrader / vnpy / Lean 都没有内置的这三类检测。
