# 大数据重建（核心层不需要，个股 / qlib 线需要）

仓库里**不含**以下目录，它们是本机重建产物（约 1.9GB）：

    stockdata/                  5258 个全市场日线 CSV
    store/                     6321 个列式二进制 + 索引
    qlib_bridge/qlib_data/     5252 个 qlib 特征文件
    m_compact.bin               紧凑面板（strategy_layer / 2 个测试用）

只跑 ETF 审计 / 三层审计 / 110 测试 / `verdict_all.py` **不需要这些**。
`python check_env.py` 会明确告诉你哪一段缺。

## 重建顺序

    python fetch_stocks.py                 # 东财全市场日线 → stockdata/*.csv
    python build_store.py                  # stockdata/ → store/*.bin + *.idx
    python compact.py                      # → m_compact.bin
    python qlib_bridge/export.py           # → qlib_bridge/qlib_data/
    python qlib_bridge/run_all.py          # 导出→训练→回测→对账（可选 qlib 层）

`fetch_stocks.py` 支持断点续抓（`stockdata/_progress.json`），
`build_store.py` 跳过已转换的文件，可中断重来。

## 现实约束（2026-11-07 实测）

- **东财 `push2his.eastmoney.com` 此刻不通**（HTTPS RemoteDisconnected /
  HTTP 502），所以 `fetch_stocks.py` 现在跑不出新数据。新浪/腾讯的备用源
  能给 1900 天以内的 ETF 日线，但不给全市场股票 + 前复权 + 成交额，
  不能直接顶替。要重建得等东财恢复，或另找给全市场前复权行情的源。
- `build_store.py` / `compact.py` 是纯本机转换，不依赖网络，有 `stockdata/`
  就能跑。
- `export.py` 依赖 `m_compact.bin`；qlib 层还需
  `pip install -r requirements-qlib.txt`，Linux 上 pyqlib 0.9.7 要源码编译。
- 全市场 5258 只、十余年日线的抓取与转换是小时级任务，不是分钟级。

## 为什么没打包上传

1.9GB 二进制不适合放 GitHub（单文件 100MB 上限、LFS 需额外配置），
且数据源许可与复权口径本就该由使用者自己确认。这里选择把**重建脚本**
和**自检工具**入库，而不是把一个来源不明的二进制塞给别人。

## 缺数据时哪些会跳过

`tests/test_strategy_layer.py` 的两个用例依赖 `m_compact.bin`，
文件不在时自动 `skip`（`unittest.skipUnless`），其余 108 个照跑。
本机数据齐全时这两个用例真跑、不跳过。

    # 干净 clone 实测（系统 Python，无 qlib，无大数据）
    python -m unittest discover -s tests
    # Ran 110 tests ... OK (skipped=2)
