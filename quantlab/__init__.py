"""quantlab — 机构级、可复现的量化研究栈（纯标准库实现）。

设计原则：
  1. Point-in-time 优先：任何数据访问必须携带 asof，越界抛 LookaheadError。
  2. 执行真实：T+1、涨跌停、停牌、整手、ADV 容量、冲击成本，每笔拒绝都记账。
  3. 统计严谨：样本外、多重检验校正、PBO、安慰剂、无未来函数验收。
  4. 可审计：配置指纹 + 数据哈希 + 随机种子 + 结果落盘，任何数字可追溯到一次运行。

研究脚本（backtest_r7.py 等）保留作为历史记录；新代码请基于本包。
"""
__version__ = "1.0.0"

from .config import Config, CostConfig, ExecutionConfig, BacktestConfig, ValidationConfig
from .clock import AsOf, LookaheadError

__all__ = [
    "Config", "CostConfig", "ExecutionConfig", "BacktestConfig",
    "ValidationConfig", "AsOf", "LookaheadError", "__version__",
]
