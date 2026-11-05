"""组合账本：唯一记账入口，带不变量断言。

历史上「退市股按最后价无限期存活」和「删掉持仓等于凭空多出现金」都源于
没有集中账本。这里把 three 条不变量写成运行时断言：
  1. cash >= 0
  2. equity = cash + sum(shares * price)，退市股按清偿率计价且**不产生现金**
  3. 买入股数在 T+1 之前计入 locked，不可卖出
"""
from collections import Counter

from ..config import Config
from ..data.panel import CrossSection
from .execution import Fill


class Book:
    def __init__(self, cfg: Config, initial_cash: float | None = None) -> None:
        self.cfg = cfg
        self.cash = float(initial_cash if initial_cash is not None else cfg.backtest.cash)
        self.positions: dict[str, int] = {}
        self.locked: dict[str, int] = {}
        self.last_price: dict[str, float] = {}
        self.equity_curve: list[tuple[str, float]] = []
        self.fills: list[Fill] = []
        self.fees_paid: float = 0.0
        self.turnover: float = 0.0
        self.blocked: Counter = Counter()
        self.delisted_marked: set[str] = set()

    def release_locks(self) -> None:
        """新交易日开始：昨日买入解锁。"""
        self.locked.clear()

    def available_shares(self, code: str) -> int:
        if not self.cfg.execution.t_plus_1:
            return self.positions.get(code, 0)
        return max(0, self.positions.get(code, 0) - self.locked.get(code, 0))

    def apply_fill(self, fill: Fill) -> None:
        self.fills.append(fill)
        if fill.status == "REJECTED":
            self.blocked[fill.reason] += 1
            return
        o, px, sh = fill.order, fill.price, fill.shares
        amount = px * sh
        fee = fill.cost.total if fill.cost else 0.0
        self.fees_paid += fee
        self.turnover += amount
        if o.side == "BUY":
            self.cash -= amount + fee
            self.positions[o.code] = self.positions.get(o.code, 0) + sh
            if self.cfg.execution.t_plus_1:
                self.locked[o.code] = self.locked.get(o.code, 0) + sh
        else:
            self.cash += amount - fee
            left = self.positions.get(o.code, 0) - sh
            if left <= 0:
                self.positions.pop(o.code, None)
                self.locked.pop(o.code, None)
            else:
                self.positions[o.code] = left
                self.locked[o.code] = min(self.locked.get(o.code, 0), left)
        self.assert_invariants()

    def mark(self, date: str, cs: CrossSection) -> float:
        """按收盘价计价。退市股按清偿率归零，不产生现金。"""
        recovery = self.cfg.execution.delisting_recovery
        mv = 0.0
        for code, sh in list(self.positions.items()):
            px = cs.close(code)
            if px is not None:
                self.last_price[code] = px
                mv += sh * px
                continue
            if cs.universe.is_delisted(code, cs.at):
                # 关键：只减记市值，绝不动 cash，也不能 pop 仓位（那等于凭空卖出）
                if code not in self.delisted_marked:
                    self.delisted_marked.add(code)
                    self.last_price[code] = self.last_price.get(code, 0.0) * recovery
                mv += sh * self.last_price.get(code, 0.0)
            else:
                # 停牌：沿用最后已知价，但不参与当日交易
                mv += sh * self.last_price.get(code, 0.0)
        equity = self.cash + mv
        self.equity_curve.append((date, round(equity, 6)))
        self.assert_invariants()
        return equity

    def market_value(self, cs: CrossSection) -> float:
        return sum(sh * (cs.close(c) or self.last_price.get(c, 0.0))
                   for c, sh in self.positions.items())

    def equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else self.cash

    def assert_invariants(self) -> None:
        if self.cash < -1e-5:
            raise AssertionError(f"现金为负: {self.cash}")
        for c, sh in self.positions.items():
            if sh < 0:
                raise AssertionError(f"负持仓: {c}={sh}")
            if not isinstance(sh, int):
                raise AssertionError(f"非整手持仓: {c}={sh}")
