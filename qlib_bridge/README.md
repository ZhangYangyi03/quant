# qlib_bridge —— 本项目面板 ↔ microsoft/qlib 的接缝

合二为一的做法：qlib 当地基（数据层 + 滚动切分 + 模型 + 回测），
本项目审计层（corrupt/PIT/对账/统计检验）留在外部，两套实现跑同一份面板然后对数。

## 目录

    .qlibenv/                 可选的本机 3.11 venv；新环境用 requirements-qlib.txt
    qlib_bridge/export.py    把 m_compact.bin 导成 qlib 的 calendars/instruments/features
    qlib_bridge/qlib_data/   导出的数据目录（不进 git，2.1GB）
    qlib_bridge/bridge_backtest.py
                             qlib 的 DataHandlerLP + LGBModel + TopkDropoutStrategy 回测
    qlib_bridge/qlib_curve.json
                             qlib 侧净值曲线，供本项目 perf.py / 统计检验复用

## 环境坑（本机实测）

    1. 项目根曾有 _queue.py，遮蔽 stdlib queue，让 logging.handlers 陷入死循环，
       任何 qlib import 都卡死。已改名为 legacy_queue_runner.py.txt。
    2. numpy/mlflow 首次 import 时编译 .pyc 撞 512MB 提交上限。
       解法：python -m compileall 预编译 + PYTHONDONTWRITEBYTECODE=1 + OPENBLAS=1。
    3. qlib 的 DataLoader 默认多进程，撞 Windows 进程配额（WinError 1816）。
       解法：C["kernels"]=1、joblib_backend="threading"。
    4. qlib 回测默认基准 SH000300，我们没导指数；造一根恒 1 的 __bench__ 伪股票。
    5. qlib 文件名约定是 close.day.bin（不带 $），表达式 $close 会自己去掉 $。
    6. 前复权脏价：导出时把 corrupt/halt 日的 close/open 置 NaN，
       qlib 的 DropnaProcessor 自动丢行；但跨期 Ref 仍可能读到未清洗的复权断层，
       标签用 5 日收益并把训练输入里的 inf 裁到 0。

## 数据格式（qlib FileStorage）

    calendars/day.txt              每行一个交易日
    instruments/all.txt            code<TAB>start<TAB>end
    features/<code>/<field>.day.bin
        float32，第 0 个元素 = 该股在 calendar 里的 start_index，其后为每日值
    quality/<code>/<mask>.day.bin  corrupt / halt 掩码（非零序列才写）

字段：close / open / amount / mktcap。

## 当前状态（2026-11-05 定稿）

一层命令跑完整条链：

    python -u export.py          # 导出，26s
    python -u bridge_train.py    # handler + LGB，29s
    python -u bridge_run.py      # TopkDropout 回测，25s
    python -u reconcile.py       # 对账，6s
（每个脚本前设 OPENBLAS/OMP/MKL_NUM_THREADS=1、PYTHONDONTWRITEBYTECODE=1，
 cwd 必须是 qlib_bridge/，不能是项目根。）

对账结果 reconcile_report.json：

    price_check   232 对 (股票, 日期, 字段)，max_abs_err = 0.0  → ok
    quality       corrupt 日 close 全部 NaN（11/11）             → ok
    qlib 曲线     1159 个交易日、57 个月，期末净值 1.1298
    月度相关      vs size_amt20_top50  0.755（同号 75.4%）
                  vs ep_top50          0.714（同号 71.9%）
                  vs random_k_equal    0.816（同号 89.5%）

相关都高，是因为两套都吃同一份 A 股 beta；但 qlib 侧只用 300 只池 + 日频
TopkDropout + LGB，本项目是 5251 只池 + 月度调仓 + 手写因子，绝对收益不该
一样，也不该一样。这里对的是数据层与方向，不是策略收益。

对账过程抓到并修掉两个真 bug：
  1. export 把 corrupt/halt 日的 close 写成 0（np.nan_to_num 的 nan=0.0），
     qlib 把 0 当正常价，污染标签与回测。改成保留 NaN，qlib 侧净值从
     1.0944 变 1.1298。
  2. qlib 回测默认基准 SH000300，我们没导指数；造恒 1 的 __bench__ 伪股票，
     并由 export.py 自动写进 instruments/all.txt。

## 刻意不合的部分

OrderPlan / 风控 / 撮合 / 逐笔 reconcile 继续留在本项目 execution_layer.py：
A 股整手、T+1、涨跌停排队 qlib 的 SimulatorExecutor 默认不按这套来，
搬过去等于重写，且会丢掉"故意作弊必须被抓出来"的蜜罐验收。
