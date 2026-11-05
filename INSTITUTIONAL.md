# quant 仓库机构级改造说明与验收口径

> 日期：2026-11-05
> 范围：在保留全部历史研究脚本的前提下，新增 `quantlab/` 研究栈、修复 P0 缺陷、建立可自动执行的验收。
> 环境约束：本机 `Python 3.13.14`，**没有 numpy / pandas / scipy**，因此 `quantlab/` 全部使用**纯标准库**实现，零第三方依赖。

---

## 0. 一句话结论

原仓库是一份**高质量的研究日志**（策略思路、异常排序、成本敏感性、事件试验都记下来了），
但**不是一份可复现的代码库**：关键数字与产出它的代码之间没有绑定，三个核心数字各自对应一个实现缺陷。

本次改造没有去「调参数把收益做高」——那是研究问题，不是工程问题。
改造交付的是**一套能自动证伪的研究基础设施**：任何一条净值曲线都必须能追溯到
唯一的配置指纹、数据指纹和随机种子，并且必须通过无未来函数验收才能被引用。

**必须明确：这不等于「可以实盘」。** 第 9 节列出距离真实机构生产环境仍缺的东西。

---

## 1. 原项目诊断回顾

### 1.1 三条断掉的数字链（P0）

| README 中的数字 | 实现缺陷 | 后果 |
|---|---|---|
| `CAGR 24.14% / 31.46 倍` | 全仓找不到对应的结果 JSON；最接近的 `r8_ep_nofilter.json` 是 `24.25% / 31.90x` | 数字无法溯源到任何一次运行 |
| `超额 17.45pp`（基准 6.69%） | `backtest_r7.py` 的 `eq_nofilter` 分支**既不排序也不抽样**，`cand[:k]` 落在 `fundamentals.json` 键序的前 k 只固定老股上（`r7_eq_nofilter.json` 仅 993 笔成交 vs EP 组 10085 笔） | 基准不是「同池、同费用、同频率的可执行基准」，超额无意义 |
| `逐年中位 11.19%` | `walkforward.py` 缺少按 EP 降序排序，`cand[:50]` 取的是股票代码升序的前 50 只；docstring 承诺的参数 walk-forward 从未实现 | 17 个年度数字测的不是 EP 策略 |

### 1.2 工程缺口

- 16 个 `.py` / 3 个 `.bat` 硬编码旧机器路径（`D:\Users\china\...`、`C:\Users\china\miniconda3\python.exe`），换台机器即不可用。
- README 声称含 0.1% 滑点，代码里从未实现；印花税全期按 0.1% 收（2023-08-28 起已减半至 0.05%）；一字涨停买不进、T+1、停牌、整手、ADV 容量均未建模。
- 用 `log(价格)` 代理市值；用 `i % 20 == 0` 切调仓日而非真实交易日。
- 无 `LICENSE`、无 `requirements.txt` / `pyproject.toml`、无 CI、无单元测试（`test_events.py` 是研究脚本，无 `assert`）。
- 12 个 `.npy` / `.bin` 大矩阵未随仓库提供，历史结果**无法重跑**。
- `METHOD.md` / `ROADMAP.md` / `README.md` 描述的是三代不同系统，文档与代码不一致。

---

## 2. 改造范围与边界

### 做了什么

| 层 | 模块 | 关键能力 |
|---|---|---|
| 配置 | `quantlab/config.py` | frozen dataclass 集中所有业务数字；`fingerprint()` 把配置哈希写进结果 |
| 时钟 | `quantlab/clock.py` | `AsOf` / `LookaheadError`，越界读取抛异常而非靠自觉 |
| PIT 数据 | `quantlab/data/pit.py` | 财报按**公告日**生效、报告期截止日兜底、修订版本按 asof 可见、TTM EPS 只用已公告记录 |
| 股票池 | `quantlab/data/universe.py` | 退市股在退市前保留（反幸存者偏差）、ST 时间窗、板块/行业 |
| 日历 | `quantlab/data/calendar.py` | 真实交易日历，月/周/季调仓日落在真实交易日 |
| 面板 | `quantlab/data/panel.py` | `Market.view(date)` 生成**物理上不含未来**的 `CrossSection`；`Panel.history` 用二分 + 排序缓存 |
| 合成数据 | `quantlab/data/synth.py` | 确定性生成，刻意注入退市、停牌、一字涨跌停、财报修订、ST 窗口 |
| 成本 | `quantlab/model/cost.py` | 佣金/最低佣金/过户费/印花税分段/平方根冲击，冲击与固定滑点二选一避免重复计费 |
| 因子 | `quantlab/model/factors.py` | MAD 去极值、z-score、平均秩、OLS 行业/风格中性化、Spearman rank IC |
| 组合 | `quantlab/model/portfolio.py` | 等权 / 逆波动率 / 带约束最小方差；统一强制「非负 + 和为 1 + 单票上限」 |
| 风险 | `quantlab/model/risk.py` | 风格暴露、收缩协方差、因子风险 vs 特异性风险分解 |
| 执行 | `quantlab/backtest/execution.py` | T+1、一字涨跌停、停牌、100 股整手、ADV 容量、现金约束；每笔拒绝都记原因 |
| 账本 | `quantlab/backtest/accounting.py` | 唯一记账入口，现金非负 / 持仓非负 / 退市不凭空产生现金 |
| 主循环 | `quantlab/backtest/engine.py` | 事件驱动：t 日收盘出信号 → t+1 日开盘撮合；同池随机 K 等权基准 |
| 统计 | `quantlab/validate/stats.py` | CAGR/Sharpe/Sortino/MDD/Calmar/偏度峰度、Newey-West HAC t、DSR、PBO/CSCV、bootstrap Reality Check |
| 验证 | `quantlab/validate/walkforward.py` | 真参数 walk-forward（只用历史选参）+ 平行世界/蜜罐泄漏检测 |
| 验收 | `quantlab/validate/no_lookahead.py` | 截断重跑 + 蜜罐反例 + 截面 PIT 不变量 |
| 留痕 | `quantlab/reporting.py` | 配置指纹 + 数据指纹 + 种子 + 环境 + 完整净值写入同一份 JSON |

### 没做什么（刻意的边界）

- **不改历史研究脚本的结论**：`backtest_r*.py` / `r7_*.json` / `anomaly_race.json` 原样保留为历史记录，只修实现缺陷并加作废声明。
- **不重算历史收益**：缺少 `.bin` 行情矩阵和 `stockdata/` 原始数据，无法重跑；不编造替代数字。
- **不宣称策略有效**：演示跑在合成数据上，只验证代码正确性。
- **不接实盘**：无券商接口、无风控合规、无 OMS/EMS、无实盘监控。

---

## 3. 关键设计决策

### 3.1 未来函数从「纪律问题」变成「物理不可能」

策略函数唯一能拿到的是 `CrossSection`（某日收盘的信息集），其中：
- `bars` 只有当日的 Bar；`history` 用二分上界 `bisect_right(keys, asof.date)` 截取，**不可能**返回未来 K 线；
- `fundamentals.visible(code, asof)` 只返回 `announced_at <= asof` 的财报记录；
- 直接调 `Panel.get(code, future_date, AsOf(past))` 抛 `LookaheadError`；
- 信号在 t 日收盘产生，订单进 `pending` 队列，t+1 日开盘才撮合 —— 信号与成交强制隔夜。

财报可见性口径：一季报 `04-30`、半年报 `08-31`、三季报 `10-31`、年报次年 `04-30`；
修订版本单独存一条 `Report(version=2)`，公告日更晚，asof 查询只取当时可见的最高版本。

### 3.2 基准必须是可执行的对照组

`run_benchmark()` 用**同池、同 k、同调仓频率、同费用**的随机抽样等权组合。
随机分数由 `sha256(seed | asof日期 | 代码)` 生成：每期强制换仓、无选股偏差、同种子完全可复现。
这直接修复历史 `eq_nofilter` 固定篮子的问题。

### 3.3 成本口径集中且可审计

| 项目 | 口径 |
|---|---|
| 佣金 | 2.5 bps 双边，单笔最低 ¥5 |
| 过户费 | 1 bp 双边 |
| 印花税 | 仅卖出；`< 2023-08-28` 10 bps，`>=` 5 bps |
| 滑点/冲击 | 二选一；默认平方根冲击 `eta·sigma·sqrt(Q/ADV)`，避免与固定 10 bps 重复计费 |
| 容量 | 单票单日成交额 ≤ `ADV × 10%` |

### 3.4 可复现性与留痕

任何 `Result` 都带 `config_fingerprint`（配置 JSON 的 SHA256 前 16 位）、
`data_fingerprint`（行情面板哈希）、`seed`。`reporting.result_record()` 把这些连同
`config` 全量、完整净值曲线、逐项费用、拒单原因写进同一份产物。
改一个参数 → 指纹变化 → 旧结果自动失效，不再出现「旧 JSON 被继续引用」。

### 3.5 检验器本身必须可被证伪

一个只会说 PASS 的验收器毫无价值。`verify_no_lookahead()` 除了截断重跑，
还会故意构造一个**偷看未来财报**的蜜罐信号，要求验收器**必须**把它抓出来；
若连明显作弊都检测不到，整轮验收判 FAIL。同理 `detect_lookahead()` 用平行世界
（未来收益取反）检测选参函数是否依赖未来，同时覆盖 `future` 形参和闭包两条作弊路径。

---

## 4. 修复清单

### 上一轮的三个 P0（已修，回归测试锁死）

1. `walkforward.py`：补上 `cand.sort(key=lambda x: -x[1])`；docstring 改为如实说明
   「这是分年独立资金收益，不是参数 walk-forward」，并声明历史数字作废。
2. `backtest_r7.py`：`eq_*` 分支加入按 `(调仓月, 代码)` 的确定性哈希抽样，每期换仓。
3. 全仓 16 个 `.py` + 3 个 `.bat`：硬编码旧机器路径改为
   `HERE = os.path.dirname(os.path.abspath(__file__))` / `%~dp0` / `python`；
   README 顶部加入作废声明。

### 本轮搭建过程中发现并修复的新缺陷

| 位置 | 问题 | 修复 |
|---|---|---|
| `validate/__init__.py` | 导入尚不存在的 `no_lookahead.py`，整个验证层 import 失败 | 新建模块 |
| `validate/walkforward.py` | `detect_lookahead` 给普通 `choose` 传 `future=`，必抛 `TypeError` | `_call_choose` 用 `inspect.signature` 判断，并新增原地翻转以抓闭包作弊，`finally` 恢复数据 |
| `validate/stats.py` | `deflated_sharpe(n_trials=1)` 触发 `_norm_ppf(0)` 的 ValueError | 单次试验不做多重检验折减，`sr0 = 0` |
| `validate/stats.py` | Euler 常数写成负号 | 改为 `0.5772156649...` |
| `model/portfolio.py` | 等权/逆波动率无视单票上限；最小方差分支 `min(v, cap)` 后不再归一化，权重和 < 1 等于凭空留现金 | 统一 `_enforce_cap()`，投影到 capped simplex 后归一化 |
| `model/cost.py` | 自定义 `math_sqrt()` 不存在 | 用 `math.sqrt` |
| `backtest/execution.py` | 现金不足时用「原始大单的费用」折算可买股数，会误拒本可成交的小单 | 在整手网格上二分求满足 `amount + cost(amount) <= cash` 的最大手数 |
| `backtest/accounting.py` | 退市减记对 `mv` 重复乘清偿率 | 去掉重复乘数；退市只减记市值、不动现金、不 pop 仓位 |
| `data/panel.py` | `history()` 每次全量排序 O(D log D)，事件驱动循环里成为瓶颈 | 缓存排序键 + `bisect_right`，O(log D)；增量 `add` 时失效缓存 |
| `backtest/__init__.py` | 未导出 `run_benchmark` | 补齐导出 |

---

## 5. 验收口径（可复现命令）

```bash
# 全部单元 / 集成 / 回归测试
python -m unittest discover -s tests -v

# 端到端演示 + 六道验收闸门 + 结果落盘
python run_institutional.py

# 自定义规模
python run_institutional.py --codes 40 --days 520 --out results/demo.json
```

自定义规模时 `top_k` 与 walk-forward 的 `min_history / oos_len` 会按数据量自适应
（否则股票池小于持仓数时 G5「基准非固定篮子」会误判失败）；
`--codes < 10` 或 `--days < 120` 会被拒绝，因为无未来函数验收样本不足。
默认 760 天时仍为 `min_history=120 / oos_len=20 / top_k=20`。

当前结果：**91 个测试全部通过**（约 0.9 秒），`compileall` 无语法错误。
产物：`results/institutional_run.json`（约 120 KB，含完整净值曲线与全部指纹）。

### 六道闸门（`run_institutional.py`，任一失败退出码非 0）

| 闸门 | 判据 |
|---|---|
| G1 无未来函数 | 截断重跑净值前缀逐日一致（差 0.0）+ 蜜罐作弊被抓到 + 未来数据确实被删除 |
| G2 选参不依赖未来 | 平行世界 `changed_ratio == 0` |
| G3 真 walk-forward | ≥ 3 个样本外期，且历史窗口严格递增（120 → 720） |
| G4 完全可复现 | 同配置同数据两次运行净值曲线逐位一致 |
| G5 基准非固定篮子 | 基准成交过的不同代码数 > top_k，且与策略共享同一 `data_fingerprint` |
| G6 运行留痕完整 | 配置指纹 / 数据指纹 / 种子均非空 |

### 测试覆盖分布

| 文件 | 覆盖 |
|---|---|
| `tests/test_data_pit.py` | PIT 时钟与越界、披露截止日、公告日至上可见性、修订版本不可提前见、TTM EPS、退市/ST/幸存者偏差、真实调仓日、面板无未来、合成反例注入、truncate |
| `tests/test_model.py` | 印花税分段/最低佣金/冲击与滑点互斥/平方根律、因子预处理与中性化、rank IC、权重三约束、协方差与风险分解 |
| `tests/test_execution.py` | 停牌/一字涨跌停/T+1/整手/ADV 容量/现金二分/逐项费用、账本不变量、退市减记不生现金 |
| `tests/test_backtest.py` | 逐位可复现、配置指纹、基准换仓、拒单归因、审计记录、信号纯度、成本敏感性 |
| `tests/test_validation.py` | 绩效、NW t、DSR、PBO、RC、真 walk-forward、泄漏检测三种路径、无未来函数验收正例与反例 |
| `tests/test_regressions.py` | 锁死三个 P0 修复、全仓无硬编码路径、`.bat` 可移植、全模块编译、交付物齐全 |

---

## 6. 当前演示输出（**合成数据，禁止当作投资结论**）

```
合成标的 60 只 / 交易日 760 天 / K 线 44155 根（含退市 5 只）
config_fingerprint = a016c221a1379633   data_fingerprint = 1b149afb718ce299

                    EP 策略      随机K基准
CAGR %              15.6667       7.8258
Sharpe               0.8512       0.4846
最大回撤 %          -16.6726     -23.8969

Newey-West t(超额) = 2.5139   DSR = 0.8814   PBO = 0.2857   RC p = 0.523

G1 截断点 2021-09-30：删除未来 K 线 16735 根 / 财报 365 条；
   截断重跑最大净值差 0.0；蜜罐作弊净值差 139836.93（被抓到）
G3 walk-forward：31 个样本外期，历史窗口 120 -> 720
G1–G6 全部 PASS
```

合成数据的价格漂移由一个固定的潜在质量因子驱动，E/P 因此**天然具有预测力**——
这是为了让管线能跑出「有信号」的情形，不是任何真实市场结论。
统计检验在这里只演示接口能算、数值落在合理区间，**不作显著性主张**。

---

## 7. 历史数字当前状态

| 数字 | 状态 | 恢复条件 |
|---|---|---|
| `24.14% / 31.46 倍` | **作废**，无对应结果文件 | 找到或重跑产出它的那次运行，并补齐配置/数据指纹 |
| `超额 17.45pp`（基准 6.69%） | **作废**，基准构造错误 | 用 `run_benchmark()` 或修复后的 `eq_*` 重跑，同池同 k 同费用 |
| `逐年中位 11.19%` | **作废**，缺 EP 排序 | 用修复后的 `walkforward.py` 重跑；参数型 walk-forward 改用 `quantlab/validate/walkforward.py` |
| `r7_*` / `r8_*` / `anomaly_race.json` | **保留但冻结**，不可引用为结论 | 缺 `.bin` 矩阵与 `stockdata/`，需接入真实 PIT 数据源后全量重建 |

README 顶部已加显著声明；`walkforward.py` 与 `backtest_r7.py` 的修复处均有
`2026-11-05 修复` 注释指向本文档。

---

## 8. 如何接入真实数据

`quantlab` 的 feed 层刻意不联网、不读数据库，由上层注入，便于精确构造反例。
接入真实数据时需要实现：

1. **行情**：按 `Bar` 填充 `Panel`（date/code/OHLC/amount/volume/mktcap/suspended/limit_up/limit_down）。
   涨跌停与停牌必须是**当日实际状态**，不能用「当日涨跌幅 ≥ 9.9%」事后推算。
2. **财报**：每条 `Report` 必须带**实际公告日** `announced_at`，修订单独一条并递增 `version`。
   用报告期当公告日是典型未来函数。
3. **股票池**：`Listing` 带上市/退市日、ST 时间窗、板块、行业；退市股历史行情必须保留。
4. **日历**：真实交易所交易日历（含临时休市），不要用 `business_days()` 的周一至周五近似。
5. **重跑**：先跑 `verify_no_lookahead()`，再谈任何收益数字；把 `results/*.json` 纳入版本管理。

---

## 9. 距离真正的机构生产环境还差什么（诚实清单）

改造解决的是**研究可复现性与统计严谨性**，不是生产就绪。仍缺：

- **真实 PIT 数据源**：含退市、公司行为（分红/送转/配股/拆并股）、复权口径、停复牌、ST 变更历史；目前只有合成数据。
- **成本与执行精度**：无盘口/L2、无开盘集合竞价、无量价撮合队列、无部分成交排队模型；平方根冲击是简化近似，未做参数校准。
- **组合与风控**：无行业/风格暴露硬约束、无换手/跟踪误差约束、无杠杆与融券、无压力测试与情景分析、无合规白名单与受限证券校验。
- **统计**：DSR 的 `n_trials` 在演示里只是 5 个信号；真实研究必须记录**全部试过的**试验数（含被丢弃的）。`pbo` 是 CSCV 的简化实现，未做组合净化与样本加权。
- **工程**：无 `LICENSE`、无依赖锁定、无 CI、无 `pyproject.toml`、无类型检查/覆盖率门槛；纯标准库实现便于零依赖但性能上限低，全市场日频需换 numpy/BLAS 后端（接口已按可替换设计）。
- **运维**：无 OMS/EMS、无实盘/模拟盘对账、无净值监控告警、无数据质量门禁。
- **文档一致性**：`METHOD.md` / `ROADMAP.md` / `UNIVERSE.md` 仍描述三代不同的历史系统，尚未统一。

以上任何一项都不是本次「改造到华尔街水平」已交付的内容，不应被 README 的措辞误导。

---

## 10. 本次新增/修改的文件

**新增**
```
quantlab/                          # 完整包（config, clock, data, model, backtest, validate, reporting）
run_institutional.py               # 端到端演示与六道闸门
INSTITUTIONAL.md                   # 本文档
tests/__init__.py
tests/test_data_pit.py  tests/test_model.py       tests/test_execution.py
tests/test_backtest.py  tests/test_validation.py  tests/test_regressions.py
results/institutional_run.json     # 演示产物（可重复生成）
quant-项目评价.md                   # 上一轮的项目评价报告（仓库同级目录）
```

**修改**
```
README.md                          # 顶部作废声明 + 新框架入口说明
walkforward.py                     # P0：补 EP 排序 + 如实 docstring
backtest_r7.py                     # P0：eq_* 确定性哈希抽样
16 个 .py / 3 个 .bat              # 硬编码路径 -> 可移植路径
```

**保留不动**：所有历史研究脚本、历史 JSON / NPY 结果、`METHOD.md`、`ROADMAP.md`、
`UNIVERSE.md`、`EVENTS.md`（`UNIVERSE.md:72` 仍有一处历史文档中的旧路径描述，属研究日志原貌）。
