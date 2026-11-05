"""执行层验收：A 股交易约束与账本不变量。"""
import unittest

from quantlab.clock import AsOf
from quantlab.config import Config, ExecutionConfig
from quantlab.backtest.accounting import Book
from quantlab.backtest.execution import Executor, Order, REJECT
from quantlab.data.panel import Bar, CrossSection
from quantlab.data.pit import PITFundamentals
from quantlab.data.universe import Listing, Universe


CODE = "000001.SZ"
DATE = "2023-01-10"


def _bar(**kw):
    base = dict(date=DATE, code=CODE, open=10.0, high=10.5, low=9.5,
                close=10.2, amount=1e7, volume=1e6, mktcap=1e9,
                suspended=False, limit_up=False, limit_down=False)
    base.update(kw)
    return Bar(**base)


class ExecutorCase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.exe = Executor(self.cfg)
        self.book = Book(self.cfg, initial_cash=1_000_000.0)

    def run_fill(self, order, bar=None, adv=1e7):
        return self.exe.execute(order, DATE, bar or _bar(), adv, self.book)


class TestRejections(ExecutorCase):
    def test_no_bar(self):
        f = self.exe.execute(Order(CODE, "BUY", 100), DATE, None, 1e7, self.book)
        self.assertEqual(f.status, "REJECTED")
        self.assertEqual(f.reason, REJECT.NO_BAR)

    def test_suspension(self):
        f = self.run_fill(Order(CODE, "BUY", 100), _bar(suspended=True, volume=0))
        self.assertEqual(f.reason, REJECT.SUSPENDED)

    def test_one_word_limit_up_blocks_buy_but_not_sell(self):
        hot = _bar(limit_up=True)
        self.assertEqual(self.run_fill(Order(CODE, "BUY", 100), hot).reason,
                         REJECT.LIMIT_UP)
        self.book.positions[CODE] = 1000
        self.assertNotEqual(self.run_fill(Order(CODE, "SELL", 100), hot).status,
                            "REJECTED")

    def test_one_word_limit_down_blocks_sell(self):
        self.book.positions[CODE] = 1000
        f = self.run_fill(Order(CODE, "SELL", 100), _bar(limit_down=True))
        self.assertEqual(f.reason, REJECT.LIMIT_DOWN)

    def test_adv_capacity_zero(self):
        f = self.run_fill(Order(CODE, "BUY", 100), adv=0.0)
        self.assertEqual(f.reason, REJECT.CAPACITY)

    def test_below_lot_size(self):
        f = self.run_fill(Order(CODE, "BUY", 100), adv=500.0)
        self.assertEqual(f.reason, REJECT.LOT)


class TestExecutionRules(ExecutorCase):
    def test_lot_rounding_down_to_100(self):
        # 159 股不是整手 -> 只能成交 100 股，并且必须如实标记为部分成交
        f = self.run_fill(Order(CODE, "BUY", 159))
        self.assertEqual(f.shares, 100)
        self.assertEqual(f.status, "PARTIAL")
        whole = self.run_fill(Order(CODE, "BUY", 100))
        self.assertEqual(whole.status, "FILLED")

    def test_adv_cap_partial_fill(self):
        # adv=10000 -> cap=1000 元 -> 最多 100 股
        f = self.run_fill(Order(CODE, "BUY", 10_000), adv=10_000.0)
        self.assertEqual(f.shares, 100)
        self.assertEqual(f.status, "PARTIAL")

    def test_cash_constraint_never_overdrafts(self):
        poor = Book(self.cfg, initial_cash=1500.0)
        f = self.exe.execute(Order(CODE, "BUY", 100_000), DATE, _bar(), 1e7, poor)
        self.assertNotEqual(f.status, "REJECTED")
        poor.apply_fill(f)
        self.assertGreaterEqual(poor.cash, 0.0)
        self.assertLessEqual(f.shares * f.price + f.cost.total, 1500.0 + 1e-6)

    def test_insufficient_cash_rejected_with_reason(self):
        broke = Book(self.cfg, initial_cash=3.0)
        f = self.exe.execute(Order(CODE, "BUY", 100), DATE, _bar(), 1e7, broke)
        self.assertEqual(f.reason, REJECT.CASH)

    def test_t_plus_1_lock_and_release(self):
        f = self.run_fill(Order(CODE, "BUY", 1000))
        self.book.apply_fill(f)
        self.assertEqual(self.book.available_shares(CODE), 0)
        blocked = self.run_fill(Order(CODE, "SELL", 1000))
        self.assertEqual(blocked.reason, REJECT.T_PLUS_1)
        self.book.release_locks()
        self.assertEqual(self.book.available_shares(CODE), 1000)
        ok = self.run_fill(Order(CODE, "SELL", 1000))
        self.assertNotEqual(ok.status, "REJECTED")

    def test_t_plus_1_can_be_disabled(self):
        cfg = Config(execution=ExecutionConfig(t_plus_1=False))
        book = Book(cfg, initial_cash=1_000_000.0)
        exe = Executor(cfg)
        book.apply_fill(exe.execute(Order(CODE, "BUY", 1000), DATE, _bar(), 1e7, book))
        self.assertEqual(book.available_shares(CODE), 1000)

    def test_every_fill_carries_itemized_cost(self):
        f = self.run_fill(Order(CODE, "BUY", 1000))
        self.assertIsNotNone(f.cost)
        self.assertGreater(f.cost.total, 0)
        self.assertEqual(f.cost.total,
                         f.cost.commission + f.cost.transfer + f.cost.stamp
                         + f.cost.slippage + f.cost.impact)


class TestBookInvariants(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.exe = Executor(self.cfg)

    def test_cash_never_negative(self):
        book = Book(self.cfg, initial_cash=100.0)
        with self.assertRaises(AssertionError):
            book.cash = -1.0
            book.assert_invariants()

    def test_delisted_position_is_written_down_without_creating_cash(self):
        uni = Universe()
        uni.add(Listing(CODE, "2020-01-01", delisted_on="2023-01-01"))
        cs = CrossSection(at=AsOf(DATE), members=(), bars={}, history={},
                          fundamentals=PITFundamentals(), universe=uni)
        book = Book(self.cfg, initial_cash=100_000.0)
        book.positions[CODE] = 1000
        book.last_price[CODE] = 10.0
        equity = book.mark(DATE, cs)
        self.assertEqual(book.cash, 100_000.0, "退市不得凭空产生现金")
        self.assertEqual(equity, 100_000.0, "清偿率为 0 时市值必须归零")
        self.assertIn(CODE, book.positions, "不得靠 pop 仓位假装卖出")
        self.assertIn(CODE, book.delisted_marked)

    def test_fees_and_blocked_are_booked(self):
        book = Book(self.cfg, initial_cash=1_000_000.0)
        ok = self.exe.execute(Order(CODE, "BUY", 1000), DATE, _bar(), 1e7, book)
        bad = self.exe.execute(Order(CODE, "BUY", 1000), DATE, None, 1e7, book)
        book.apply_fill(ok)
        book.apply_fill(bad)
        self.assertGreater(book.fees_paid, 0.0)
        self.assertEqual(book.blocked[REJECT.NO_BAR], 1)
        self.assertEqual(len(book.fills), 2)
        self.assertAlmostEqual(book.turnover, ok.shares * ok.price)


if __name__ == "__main__":
    unittest.main()
