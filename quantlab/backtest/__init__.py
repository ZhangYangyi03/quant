"""回测层：事件驱动撮合、组合账本、主循环。"""
from .accounting import Book
from .execution import Executor, Order, Fill, REJECT
from .engine import run_backtest, run_benchmark, Result, default_signal

__all__ = ["Book", "Executor", "Order", "Fill", "REJECT",
           "run_backtest", "run_benchmark", "Result", "default_signal"]
