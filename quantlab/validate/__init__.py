"""验证层：绩效与统计检验、真 walk-forward、无未来函数验收。"""
from .stats import (performance, newey_west_t, deflated_sharpe, pbo,
                    bootstrap_max_sharpe_p, annualized_sharpe)
from .walkforward import (WalkForwardResult, run_walkforward, detect_lookahead,
                          WalkForwardLeakError)
from .no_lookahead import verify_no_lookahead

__all__ = [
    "performance", "newey_west_t", "deflated_sharpe", "pbo",
    "bootstrap_max_sharpe_p", "annualized_sharpe",
    "WalkForwardResult", "run_walkforward", "detect_lookahead",
    "WalkForwardLeakError", "verify_no_lookahead",
]
