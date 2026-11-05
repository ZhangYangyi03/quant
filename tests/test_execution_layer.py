# -*- coding: utf-8 -*-
"""执行层回归测试：现金二次约束、拒单对账、风控拒绝、CSV 导出。"""
import os, sys, unittest, tempfile, json
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from dataclasses import replace
from quantlab.config import Config
from quantlab.execution import OrderPlan, PreTradeRisk, OrderRecord, FillRecord, reconcile
from quantlab.execution.broker import PaperBroker, CsvBroker, fill_is_bps
from quantlab.execution.oms import order_plan_fingerprint


def cfg():
    c = Config()
    return replace(c, cost=replace(c.cost, use_impact_model=False, min_commission=0.0,
                                   slippage_bps=0.0, adv_cap=1.0))


class TestOrderPlan(unittest.TestCase):
    def test_sell_before_buy(self):
        p = OrderPlan(1e6, lot=100)
        o = p.build({"B": 0.5, "C": 0.5}, {"A": 1000}, {"A": 1000},
                    {"A": 10.0, "B": 10.0, "C": 10.0}, "2020-01-31", "2020-02-03")
        self.assertEqual([x.side for x in o[:1]], ["SELL"])
        self.assertEqual(o[0].code, "A")

    def test_fingerprint_stable(self):
        a = [OrderRecord("x", "000001", "BUY", 100, "r", "d", 10.0, "d2")]
        b = [OrderRecord("x", "000001", "BUY", 100, "r", "d", 10.0, "d2")]
        self.assertEqual(order_plan_fingerprint(a), order_plan_fingerprint(b))


class TestPreTradeRisk(unittest.TestCase):
    def test_cash_rolling(self):
        """卖出回款必须可用于后续买入，否则月度换仓会被误拒。"""
        c = cfg()
        risk = PreTradeRisk(c, max_weight=1.0, max_industry_weight=1.0,
                            max_participation=1.0)
        orders = [
            OrderRecord("s", "A", "SELL", 1000, "drop", "d", 10.0, "d2", status="PLANNED"),
            OrderRecord("b", "B", "BUY", 900, "raise", "d", 10.0, "d2",
                        target_weight=0.5, status="PLANNED"),
        ]
        ok, bad = risk.check(orders, prices={"A": 10.0, "B": 10.0},
                             adv={"A": 1e9, "B": 1e9}, cash=0.0, equity=10000.0,
                             industries={"A": "x", "B": "y"},
                             bar_flags={"A": {"open": 10.0}, "B": {"open": 10.0}},
                             positions={"A": 1000})
        self.assertEqual(len(ok), 2, bad)

    def test_industry_cap(self):
        c = cfg()
        risk = PreTradeRisk(c, max_weight=1.0, max_industry_weight=0.2)
        o = [OrderRecord("b", "B", "BUY", 100, "raise", "d", 10.0, "d2",
                         target_weight=0.3, status="PLANNED")]
        ok, bad = risk.check(o, prices={"B": 10.0}, adv={"B": 1e9}, cash=1e6,
                             equity=1e6, industries={"B": "bank"}, bar_flags={},
                             positions={"B": 0})
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0].reject_reason, PreTradeRisk.INDUSTRY)

    def test_limit_up_reject(self):
        c = cfg()
        risk = PreTradeRisk(c, max_weight=1.0, max_industry_weight=1.0)
        o = [OrderRecord("b", "B", "BUY", 100, "raise", "d", 10.0, "d2",
                         target_weight=0.5, status="PLANNED")]
        ok, bad = risk.check(o, prices={"B": 10.0}, adv={"B": 1e9}, cash=1e6,
                             equity=1e6, industries={"B": "x"},
                             bar_flags={"B": {"limit_up": True, "open": 10.0}},
                             positions={"B": 0})
        self.assertEqual(bad[0].reject_reason, "limit_up_cannot_buy")


class TestPaperBroker(unittest.TestCase):
    def test_gap_up_cash_constraint(self):
        """决策价 10 元过风控，执行价 20 元时不得撮合出现金为负。"""
        c = cfg()
        b = PaperBroker()
        o = [OrderRecord("b", "B", "BUY", 1000, "raise", "d", 10.0, "d2",
                         status="APPROVED")]
        fills = b.send(o, prices={"B": 20.0}, adv={"B": 1e12}, date="d2",
                       cfg=c, cash=10000.0, bar_flags={"B": {"open": 20.0}})
        self.assertEqual(len(fills), 1)
        self.assertLessEqual(fills[0].shares * fills[0].price + fills[0].fee["total"], 10000.0 + 1e-6)
        self.assertEqual(fills[0].shares % c.execution.lot, 0)

    def test_broker_rejects_are_cancellations(self):
        c = cfg()
        b = PaperBroker()
        o = [OrderRecord("b", "B", "BUY", 100, "raise", "d", 10.0, "d2",
                         status="APPROVED")]
        fills = b.send(o, prices={"B": 10.0}, adv={"B": 1e9}, date="d2",
                       cfg=c, cash=1e6,
                       bar_flags={"B": {"open": 10.0, "limit_up": True}})
        self.assertEqual(fills, [])
        self.assertIn("b", b.last_rejects)
        r = reconcile(o, fills, set(b.last_rejects))
        self.assertTrue(r["clean"])
        self.assertEqual(r["cancelled"], 1)

    def test_reconcile_flags_overfill_and_unknown(self):
        plan = [OrderRecord("a", "B", "BUY", 100, "r", "d", 10.0, "d2", status="APPROVED")]
        self.assertFalse(reconcile(plan, [FillRecord("a", "B", "BUY", 200, 10.0, "d2")])["clean"])
        self.assertFalse(reconcile(plan, [FillRecord("z", "B", "BUY", 100, 10.0, "d2")])["clean"])

    def test_is_bps_sign(self):
        o = OrderRecord("a", "B", "BUY", 100, "r", "d", 10.0, "d2")
        f = FillRecord("a", "B", "BUY", 100, 10.1, "d2", {"total": 1.0})
        self.assertGreater(fill_is_bps(o, f), 100.0)  # 涨 1% + 费用

    def test_csv_broker(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "orders.csv")
            b = CsvBroker(path)
            o = [OrderRecord("a", "B", "BUY", 100, "r", "d", 10.0, "d2", status="APPROVED")]
            self.assertEqual(b.send(o), [])
            self.assertTrue(os.path.exists(path))
            self.assertIn("order_id", open(path, encoding="utf-8-sig").read())


if __name__ == "__main__":
    unittest.main(verbosity=2)
