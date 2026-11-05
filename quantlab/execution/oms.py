# -*- coding: utf-8 -*-
"""订单计划、盘前风控、成交回报与对账。

一条订单从生成到成交的完整状态：
  PLANNED -> APPROVED / REJECTED(pretrade) -> FILLED / PARTIAL / REJECTED(broker) / CANCELLED
对账 reconcile() 只认 (order_id, code, side)，能查出漏单、重复成交、超额成交。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from hashlib import sha256
import json
import math

from ..backtest.execution import REJECT
from ..model.cost import compute_cost, max_notional, CostBreakdown


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    code: str
    side: str                 # BUY | SELL
    shares: int
    reason: str
    decision_date: str
    decision_price: float     # 决策日收盘价，用于实现缺口 IS
    exec_date: str            # 计划成交日（下一交易日开盘）
    target_weight: float = 0.0
    status: str = "PLANNED"   # PLANNED | APPROVED | REJECTED | FILLED | PARTIAL | CANCELLED
    reject_reason: str = ""
    filled_shares: int = 0
    exec_price: float = 0.0
    fee: dict = field(default_factory=dict)
    is_bps: float = 0.0       # implementation shortfall（决策价 -> 成交价 + 费用）

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FillRecord:
    order_id: str
    code: str
    side: str
    shares: int
    price: float
    date: str
    fee: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def make_order_id(decision_date: str, code: str, side: str, reason: str) -> str:
    blob = "|".join((decision_date, code, side, reason))
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


class OrderPlan:
    """目标权重 -> 订单列表。先卖后买，卖出只用 T+1 可用股份。"""

    def __init__(self, equity: float, lot: int = 100, min_ticket: float = 2000.0,
                 max_weight: float = 0.10):
        self.equity = float(equity)
        self.lot = int(lot)
        self.min_ticket = float(min_ticket)
        self.max_weight = float(max_weight)
        self.orders: list[OrderRecord] = []

    def build(self, target: dict[str, float], positions: dict[str, int],
              available: dict[str, int], prices: dict[str, float],
              decision_date: str, exec_date: str) -> list[OrderRecord]:
        px = {c: prices.get(c) or 0.0 for c in set(positions) | set(target)}
        out: list[OrderRecord] = []
        for code in sorted(set(positions) - set(target)):
            sh = int(available.get(code, 0))
            if sh > 0 and px.get(code, 0) > 0:
                out.append(self._rec(code, "SELL", sh, "drop_from_target",
                                     decision_date, exec_date))
        for code in sorted(target):
            price = px.get(code, 0.0)
            if price <= 0:
                continue
            w = min(max(0.0, float(target[code])), self.max_weight)
            want = w * self.equity
            have = int(positions.get(code, 0)) * price
            diff = want - have
            if diff < -self.min_ticket:
                sh = min(int(-diff / price), int(available.get(code, 0)))
                if sh > 0:
                    out.append(self._rec(code, "SELL", sh, "trim_to_target",
                                         decision_date, exec_date, w))
            elif diff > self.min_ticket:
                sh = int(diff / price)
                if sh > 0:
                    out.append(self._rec(code, "BUY", sh, "raise_to_target",
                                         decision_date, exec_date, w))
        # 整手对齐在撮合器里做，这里保留原始意向股数，便于对账看差值
        self.orders = out
        return out

    def _rec(self, code, side, shares, reason, d_date, e_date, w=0.0):
        return OrderRecord(order_id=make_order_id(d_date, code, side, reason),
                           code=code, side=side, shares=int(shares), reason=reason,
                           decision_date=d_date, decision_price=0.0,
                           exec_date=e_date, target_weight=float(w))


class PreTradeRisk:
    """盘前风控：把会改变组合性质的订单在发出之前拦下来。

    检查项：单票上限、行业集中度、ADV 参与率、现金、整手、涨跌停/停牌、T+1。
    返回 (批准列表, 拒绝列表)，拒绝原因沿用 backtest/execution.REJECT，附加自定义项。
    """
    CONCENTRATION = "single_name_cap_breach"
    INDUSTRY = "industry_cap_breach"
    PARTICIPATION = "participation_cap_breach"

    def __init__(self, cfg, max_weight: float = 0.04, max_industry_weight: float = 0.20,
                 max_participation: float = 0.10):
        self.cfg = cfg
        self.max_weight = float(max_weight)
        self.max_industry_weight = float(max_industry_weight)
        self.max_participation = float(max_participation)

    def check(self, orders: list[OrderRecord], *, prices: dict[str, float],
              adv: dict[str, float], cash: float, equity: float,
              industries: dict[str, str], bar_flags: dict[str, dict],
              positions: dict[str, int], turnover_band: float | None = None):
        from dataclasses import replace
        approved, rejected = [], []
        # 现金按订单顺序滚动：卖出回款可用于后续买入，否则月度换仓会被
        # "卖出尚未到账" 误拒掉大半买单（实盘 OMS 必须遵守成交顺序）。
        cash_left = float(cash)
        # 先按目标权重算行业集中度（用订单的 target_weight，卖出单为 0）
        ind_w: dict[str, float] = {}
        for o in orders:
            if o.target_weight > 0:
                g = industries.get(o.code, "__unknown__")
                ind_w[g] = ind_w.get(g, 0.0) + o.target_weight
        for o in orders:
            px = float(prices.get(o.code) or 0.0)
            if px <= 0:
                rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.NO_BAR)); continue
            flags = bar_flags.get(o.code, {})
            if flags.get("suspended") or flags.get("open", px) <= 0:
                rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.SUSPENDED)); continue
            if o.side == "BUY" and flags.get("limit_up"):
                rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.LIMIT_UP)); continue
            if o.side == "SELL" and flags.get("limit_down"):
                rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.LIMIT_DOWN)); continue
            if o.side == "SELL" and int(positions.get(o.code, 0)) <= 0:
                # 无持仓的卖单直接丢弃，不进入对账簿：它不是一笔待发订单
                continue
            sh = (int(o.shares) // self.cfg.execution.lot) * self.cfg.execution.lot
            if sh <= 0:
                rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.LOT)); continue
            amount = sh * px
            a = float(adv.get(o.code) or 0.0)
            cap = max_notional(a, self.cfg.cost)
            if cap <= 0:
                rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.CAPACITY)); continue
            if amount > cap:
                sh = int(cap / px) // self.cfg.execution.lot * self.cfg.execution.lot
                amount = sh * px
                if sh <= 0:
                    rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.CAPACITY)); continue
            part = amount / a if a > 0 else 0.0
            if part > self.max_participation:
                rejected.append(replace(o, status="REJECTED", reject_reason=self.PARTICIPATION)); continue
            if o.side == "BUY":
                fee = compute_cost("BUY", amount, o.exec_date, a, self.cfg.cost)
                if amount + fee.total > cash_left + 1e-6:
                    rejected.append(replace(o, status="REJECTED", reject_reason=REJECT.CASH)); continue
                cash_left -= amount + fee.total
            else:
                fee_sell = compute_cost("SELL", amount, o.exec_date, a, self.cfg.cost)
                cash_left += amount - fee_sell.total
            # 单票上限：目标权重本身不得超过上限（避免把已持仓误算成新增暴露）
            if o.side == "BUY" and o.target_weight > self.max_weight + 1e-9:
                rejected.append(replace(o, status="REJECTED", reject_reason=self.CONCENTRATION)); continue
            g = industries.get(o.code, "__unknown__")
            if ind_w.get(g, 0.0) > self.max_industry_weight + 1e-9:
                rejected.append(replace(o, status="REJECTED", reject_reason=self.INDUSTRY)); continue
            approved.append(replace(o, shares=sh, status="APPROVED",
                                    decision_price=float(flags.get("decision_price", px))))
        return approved, rejected


def reconcile(plan: list[OrderRecord], fills: list[FillRecord],
              cancelled: set[str] | None = None) -> dict:
    """对账：PLANNED/APPROVED 里每一单都必须有对应成交或明确的券商拒单，
    且不得超量、重复、或被拒单后仍成交。"""
    by_id: dict[str, list[FillRecord]] = {}
    for f in fills:
        by_id.setdefault(f.order_id, []).append(f)
    plan_ids = {o.order_id: o for o in plan}
    cancelled = cancelled or set()
    missing, over, duplicate, unknown, filled, n_cancel = [], [], [], [], [], 0
    for oid, o in plan_ids.items():
        fs = by_id.get(oid, [])
        qty = sum(f.shares for f in fs)
        if not fs and oid in cancelled:
            n_cancel += 1
        elif not fs and o.status != "REJECTED":
            # 盘前风控拒绝的单本来就不该有成交；只有 APPROVED/PLANNED 漏单才是异常
            missing.append(oid)
        elif qty > o.shares:
            over.append({"order_id": oid, "planned": o.shares, "filled": qty})
        elif len(fs) > 1:
            duplicate.append({"order_id": oid, "n_fills": len(fs)})
        else:
            filled.append(oid)
    for oid in by_id:
        if oid not in plan_ids:
            unknown.append(oid)
    approved_ids = {o.order_id for o in plan if o.status == "APPROVED"}
    rejected_but_filled = sorted({o.order_id for o in plan if o.status == "REJECTED"} & set(by_id))
    return {"planned": len(plan), "approved": len(approved_ids),
            "filled": len(filled), "cancelled": n_cancel,
            "missing": missing, "over": over,
            "duplicate": duplicate, "unknown_fills": unknown,
            "rejected_but_filled": rejected_but_filled,
            "clean": not (missing or over or duplicate or unknown or rejected_but_filled)}


def order_plan_fingerprint(plan: list[OrderRecord]) -> str:
    blob = json.dumps([o.as_dict() for o in plan], sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return sha256(blob.encode("utf-8")).hexdigest()[:16]
