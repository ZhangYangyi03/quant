# -*- coding: utf-8 -*-
"""券商适配层：只做 paper 撮合与 CSV 导出，绝不联网下真单。

PaperBroker 用执行日开盘价成交，费用走 quantlab.model.cost.compute_cost，
与回测撮合器完全同口径；因此 strategy_layer.py 的净值与这里的对账能逐笔核对。
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import csv
import json
import os

from ..model.cost import compute_cost, CostBreakdown
from .oms import OrderRecord, FillRecord


class BrokerAdapter:
    def send(self, orders: list[OrderRecord], **ctx) -> list[FillRecord]:
        raise NotImplementedError


class PaperBroker(BrokerAdapter):
    """历史开盘价撮合。ctx 需要 prices / adv / date / cfg / cash。

    决策价与执行价隔夜会跳空，所以盘前用收盘价过的现金约束必须在这里按
    **真实开盘价**再过一遍；否则会撮合出现金为负的成交（历史项目的隐性漏洞）。
    未成交的批准单写入 last_rejects，供 reconcile 归类为 CANCELLED。
    """

    def __init__(self, strict_lot: bool = True):
        self.strict_lot = strict_lot
        self.last_rejects: dict[str, str] = {}

    def send(self, orders: list[OrderRecord], *, prices: dict[str, float],
             adv: dict[str, float], date: str, cfg,
             cash: float = 0.0,
             bar_flags: dict[str, dict] | None = None) -> list[FillRecord]:
        from ..backtest.execution import REJECT
        from ..model.cost import max_notional as _cap_notional
        out: list[FillRecord] = []
        self.last_rejects = {}
        flags = bar_flags or {}
        cash_left = float(cash)
        for o in orders:
            if o.status != "APPROVED":
                continue
            px = float(prices.get(o.code) or 0.0)
            if px <= 0:
                self.last_rejects[o.order_id] = REJECT.NO_BAR; continue
            f = flags.get(o.code, {})
            if f.get("suspended"):
                self.last_rejects[o.order_id] = REJECT.SUSPENDED; continue
            if o.side == "BUY" and f.get("limit_up"):
                self.last_rejects[o.order_id] = REJECT.LIMIT_UP; continue
            if o.side == "SELL" and f.get("limit_down"):
                self.last_rejects[o.order_id] = REJECT.LIMIT_DOWN; continue
            sh = (o.shares // cfg.execution.lot) * cfg.execution.lot if self.strict_lot else o.shares
            if sh <= 0:
                self.last_rejects[o.order_id] = REJECT.LOT; continue
            a = float(adv.get(o.code) or 0.0)
            cap = _cap_notional(a, cfg.cost)
            if cap <= 0:
                self.last_rejects[o.order_id] = REJECT.CAPACITY; continue
            sh = min(sh, int(cap / px))
            sh = (sh // cfg.execution.lot) * cfg.execution.lot
            if sh <= 0:
                self.last_rejects[o.order_id] = REJECT.CAPACITY; continue
            if o.side == "BUY":
                # 在整手网格上二分出满足 amount + cost(amount) <= cash 的最大手数
                lo, hi = 0, sh // cfg.execution.lot
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    trial = mid * cfg.execution.lot * px
                    cb0 = compute_cost("BUY", trial, date, a, cfg.cost)
                    if trial + cb0.total <= cash_left + 1e-6:
                        lo = mid
                    else:
                        hi = mid - 1
                if lo <= 0:
                    self.last_rejects[o.order_id] = REJECT.CASH; continue
                sh = lo * cfg.execution.lot
            amount = sh * px
            cb: CostBreakdown = compute_cost(o.side, amount, date, a, cfg.cost)
            fee = {"commission": cb.commission, "transfer": cb.transfer,
                   "stamp": cb.stamp, "slippage": cb.slippage,
                   "impact": cb.impact, "total": cb.total}
            cash_left += (-amount - cb.total) if o.side == "BUY" else (amount - cb.total)
            out.append(FillRecord(order_id=o.order_id, code=o.code, side=o.side,
                                  shares=int(sh), price=float(px), date=date, fee=fee))
        return out


class CsvBroker(BrokerAdapter):
    """把批准订单写成券商可导入的 CSV（不发送）。用于人工复核。"""

    def __init__(self, path: str):
        self.path = path

    def send(self, orders: list[OrderRecord], **ctx) -> list[FillRecord]:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["order_id", "code", "side", "shares", "decision_date",
                        "exec_date", "reason", "target_weight"])
            for o in orders:
                if o.status != "APPROVED":
                    continue
                w.writerow([o.order_id, o.code, o.side, o.shares,
                            o.decision_date, o.exec_date, o.reason,
                            round(o.target_weight, 6)])
        return []


def fill_is_bps(o: OrderRecord, f: FillRecord) -> float:
    """实现缺口：成交价相对决策收盘价的偏离 + 全部费用，单位 bps（买入为正成本）。"""
    if o.decision_price <= 0:
        return 0.0
    move = f.price / o.decision_price - 1.0
    if o.side == "SELL":
        move = -move
    fee_rate = float(f.fee.get("total", 0.0)) / max(f.shares * f.price, 1.0)
    return (move + fee_rate) * 10000.0
