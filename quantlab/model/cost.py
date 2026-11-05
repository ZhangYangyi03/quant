"""交易成本与冲击模型。

修复历史项目里的三个口径问题：
  1. README 声称含 0.1% 滑点，代码里从未实现 —— 现在显式建模，且与冲击二选一避免重复计费。
  2. 印花税全期按 0.1% 收 —— 现在按 2023-08-28 分段（10bp -> 5bp），且只对卖出收。
  3. 佣金/过户费/最低佣金的方向与取整 —— 现在集中在一个函数里，逐项可审计。

冲击用平方根律：impact_bps = eta * sigma * sqrt(participation) * 10000。
"""
from dataclasses import dataclass
import math

from ..config import CostConfig


@dataclass(frozen=True)
class CostBreakdown:
    commission: float
    transfer: float
    stamp: float
    slippage: float
    impact: float

    @property
    def total(self) -> float:
        return self.commission + self.transfer + self.stamp + self.slippage + self.impact


def impact_bps(participation: float, cfg: CostConfig) -> float:
    """参与率 -> 冲击（基点）。参与率被 ADV 上限截断后传入。"""
    if participation <= 0:
        return 0.0
    return cfg.impact_eta * cfg.impact_sigma * math.sqrt(participation) * 10000.0


def max_notional(adv: float, cfg: CostConfig) -> float:
    """单票单日可成交金额上限。adv<=0 表示无数据，返回 0 由执行层拒单。"""
    return max(0.0, adv * cfg.adv_cap)


def compute_cost(side: str, amount: float, trade_date: str, adv: float,
                 cfg: CostConfig) -> CostBreakdown:
    """返回某笔成交的完整费用拆解。side 为 BUY 或 SELL。"""
    if side not in ("BUY", "SELL"):
        raise ValueError("side 必须是 BUY 或 SELL")
    if amount <= 0:
        return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)

    commission = max(cfg.min_commission, amount * cfg.commission_bps / 10000.0)
    transfer = amount * cfg.transfer_bps / 10000.0
    rate = cfg.stamp_bps_after if trade_date >= cfg.stamp_cut_date else cfg.stamp_bps_before
    stamp = amount * rate / 10000.0 if side == "SELL" else 0.0

    participation = min(1.0, amount / adv) if adv > 0 else 0.0
    if cfg.use_impact_model:
        impact = amount * impact_bps(participation, cfg) / 10000.0
        slippage = 0.0
    else:
        impact = 0.0
        slippage = amount * cfg.slippage_bps / 10000.0
    return CostBreakdown(commission=commission, transfer=transfer, stamp=stamp,
                         slippage=slippage, impact=impact)
