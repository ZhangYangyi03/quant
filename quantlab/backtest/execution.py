"""撮合规则：任何一笔成交都必须通过这里，任何拒绝都必须留下原因。

A 股特有约束全部建模：
  T+1（当日买入不可卖）· 一字涨跌停 · 停牌 · 100 股整手 · ADV 容量上限 · 现金约束。
历史项目里「调仓日一字板买不进」被列为已知缺口，这里补上了。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import Config
from ..data.panel import Bar
from ..model.cost import compute_cost, max_notional, CostBreakdown
if TYPE_CHECKING:
    from .accounting import Book


class REJECT:
    NO_BAR = "no_bar"
    SUSPENDED = "suspended"
    LIMIT_UP = "limit_up_cannot_buy"
    LIMIT_DOWN = "limit_down_cannot_sell"
    T_PLUS_1 = "t_plus_1_locked"
    LOT = "below_lot_size"
    CAPACITY = "adv_capacity_zero"
    CASH = "insufficient_cash"
    NO_POSITION = "no_position"


@dataclass(frozen=True)
class Order:
    code: str
    side: str            # BUY | SELL
    shares: int
    reason: str = "rebalance"


@dataclass(frozen=True)
class Fill:
    order: Order
    date: str
    price: float
    shares: int
    cost: CostBreakdown | None
    status: str          # FILLED | PARTIAL | REJECTED
    reason: str = ""


class Executor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def execute(self, order: Order, trade_date: str, bar: Bar | None,
                adv: float, book: Book) -> Fill:
        ec, cc = self.cfg.execution, self.cfg.cost
        if bar is None:
            return self._reject(order, trade_date, REJECT.NO_BAR)
        if not bar.tradable:
            return self._reject(order, trade_date, REJECT.SUSPENDED)

        px = bar.open
        if ec.require_open_gt_zero and px <= 0:
            return self._reject(order, trade_date, REJECT.SUSPENDED)
        if order.side == "BUY" and bar.limit_up:
            return self._reject(order, trade_date, REJECT.LIMIT_UP)
        if order.side == "SELL" and bar.limit_down:
            return self._reject(order, trade_date, REJECT.LIMIT_DOWN)

        shares = order.shares
        if order.side == "SELL":
            avail = book.available_shares(order.code)
            if avail <= 0:
                return self._reject(order, trade_date,
                                    REJECT.T_PLUS_1 if ec.t_plus_1 else REJECT.NO_POSITION)
            shares = min(shares, avail)

        cap = max_notional(adv, cc)
        if cap <= 0:
            return self._reject(order, trade_date, REJECT.CAPACITY)
        shares = min(shares, int(cap / max(px, 1e-9)))
        shares = (shares // ec.lot) * ec.lot
        if shares <= 0:
            return self._reject(order, trade_date, REJECT.LOT)

        amount = shares * px
        cost = compute_cost(order.side, amount, trade_date, adv, cc)
        if order.side == "BUY" and amount + cost.total > book.cash + 1e-6:
            # 费用本身随成交额变化，不能拿「原始大单的费用」去折算可买股数，
            # 否则现金远小于下单额时会误拒本来可以成交的小单。
            # 在整手网格上二分出满足 amount + cost(amount) <= cash 的最大手数。
            lo, hi = 0, shares // ec.lot
            while lo < hi:
                mid = (lo + hi + 1) // 2
                trial = mid * ec.lot * px
                trial_cost = compute_cost("BUY", trial, trade_date, adv, cc)
                if trial + trial_cost.total <= book.cash + 1e-6:
                    lo = mid
                else:
                    hi = mid - 1
            if lo <= 0:
                return self._reject(order, trade_date, REJECT.CASH)
            shares = lo * ec.lot
            amount = shares * px
            cost = compute_cost(order.side, amount, trade_date, adv, cc)

        status = "PARTIAL" if shares < order.shares else "FILLED"
        return Fill(order=order, date=trade_date, price=px, shares=shares,
                    cost=cost, status=status, reason=order.reason)

    @staticmethod
    def _reject(order: Order, trade_date: str, why: str) -> Fill:
        return Fill(order=order, date=trade_date, price=0.0, shares=0,
                    cost=None, status="REJECTED", reason=why)
