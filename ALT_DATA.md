# 非价量数据补入（2026-11-05，M165）

## 一句话结论

四类免费数据全拉回来了、不用注册，但按 `factor_orth.json` 的同一口径检验后，
**对 [1, size, amt20] 正交后全部没有残差信息**。这是负面结论，值得记下来。

## 需要注册吗

不需要。东财 datacenter 与巨潮 cninfo 的公开接口全部免 token、免注册、纯 urllib 可拉。
唯一需要注册的是 Tushare Pro（分析师一致预期），本次没做，优先级最低。

## 拉到了什么

| 数据集 | 文件 | 行数 | 起点 | 源 |
|---|---|---|---|---|
| 龙虎榜 | alt_data/lhb.jsonl | 244054 | 2010 | 东财 RPT_DAILYBILLBOARD_DETAILSNEW |
| 机构持仓 | alt_data/orghold.jsonl | 247966 | 2010 | 东财 RPT_F10_MAIN_ORGHOLD |
| 股东户数 | alt_data/holdernum.jsonl | 343343 | 2013 | 东财 RPT_HOLDERNUM_DET |
| 两融月度截面 | alt_data/margin_monthly.jsonl | 335468 | 2012 | 东财 RPTA_WEB_RZRQ_GGMX |

两融逐日全量是千万级（2025 单季 25 万条），所以只抓 179 个月末截面，够月度调仓用。
面板 alt_data/alt_panel.json：179 个月 × 5251 只 × 6 特征。

时点纪律：lhb / margin 收盘后才公布，一律滞后 1 个月；holdernum / orghold 按
HOLD_NOTICE_DATE / NOTICE_DATE 真实公告日可见。无记录 = NaN，绝不填 0。

## 正交检验（alt_orth.json，口径同 factor_orth.json）

| 特征 | 原始 IC | t | 残差 IC | t | 被两因子解释 | 覆盖中位数 |
|---|---|---|---|---|---|---|
| lhb_net_pct | -0.0076 | -1.22 | +0.0037 | 0.71 | 1.6% | 438 |
| lhb_count | -0.0545 | -5.42 | +0.0069 | 0.86 | 12.2% | 438 |
| margin_pct | -0.0218 | -1.66 | -0.0034 | -0.48 | 24.5% | 1475 |
| margin_buy_pct | -0.0532 | -3.69 | -0.0052 | -0.90 | 27.4% | 1475 |
| holder_concent | +0.0088 | 1.44 | +0.0089 | 0.66 | 2.6% | 819 |
| org_chg | +0.0289 | 2.86 | +0.0004 | 0.05 | 2.6% | 2488 |

对照已有价量因子：amt_cv 残差 IC +0.0644 / t=11.70，max20 +0.0381 / t=6.91。
新数据连零头都不到。原始 IC 显著的那几个，本质就是小盘 + 高换手的另一种写法
（corr_size：margin_pct +0.415、margin_buy_pct +0.273），一正交就剥干净。

边界：只测了月度低频、单变量线性 IC、六个最朴素构造。事件的高频衰减、非线性
组合、以及与价量因子的交互都没验。别把"这六个没用"读成"这类数据没用"。

## 顺带查清的 PIT 缺口

report_ann_dates.json 有 184379 个 (股票, 报告期) 对，但**完全没有年报（12-31）的
真实公告日**——orghold 里 74327 个 PIT 没有的对，其中 59699 是年报。这就是之前
"PIT 真实公告日覆盖 54.4%" 的主体，剩下全靠法定截止日 4-30 兜底。

orghold 的 NOTICE_DATE 覆盖率 99.9%，但与 PIT 逐对比较：一致 110613、不一致 62826，
差异基本 1–6 天且方向是 orghold 更晚。两者不是同一份文件（机构持仓公告 vs 财报
首次披露），所以不能直接覆盖 PIT，只能当独立佐证。

巨潮 cninfo 可以按日期扫出年报真实披露日（实测 2011-04 深市 1600 条、54 页），
能造独立表；但 17 年 × 12 月 × 深沪两市要上万次请求，本次没做。

## 复跑

    python fetch_alt_data.py lhb holdernum orghold   # 全量，约 5 分钟
    python fetch_margin_monthly.py                   # 月末截面，约 2 分钟，可断点续传
    python alt_panel.py                              # 落面板，19 秒
    python alt_orth.py                               # 正交检验，7 秒

脚本都有断点续传（按 _year / _day 跳过已完成部分）。
