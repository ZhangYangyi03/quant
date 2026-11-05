"""数据层验收：PIT 时钟、财报可见性、股票池状态、交易日历、合成反例。"""
import unittest

from quantlab.clock import AsOf, LookaheadError, asof
from quantlab.data.calendar import TradingCalendar
from quantlab.data.panel import Bar, Panel
from quantlab.data.pit import Report, PITFundamentals, disclosure_deadline
from quantlab.data.universe import Listing, Universe
from quantlab.data import generate_synthetic_market


def _bar(date, close=10.0, **kw):
    base = dict(date=date, code="000001.SZ", open=close, high=close * 1.01,
                low=close * 0.99, close=close, amount=1e7, volume=1e6,
                mktcap=1e9)
    base.update(kw)
    return Bar(**base)


class TestClock(unittest.TestCase):
    def test_assert_visible_boundary(self):
        at = AsOf("2022-01-10")
        at.assert_visible("2022-01-10")          # 当日可见
        at.assert_visible("2021-12-31")
        with self.assertRaises(LookaheadError):
            at.assert_visible("2022-01-11")

    def test_error_message_is_actionable(self):
        with self.assertRaisesRegex(LookaheadError, "未来函数"):
            AsOf("2022-01-10").assert_visible("2022-03-01", "年报")

    def test_asof_helper_and_ordering(self):
        self.assertEqual(asof("2022-01-01"), AsOf("2022-01-01"))
        self.assertLess(AsOf("2020-01-01"), AsOf("2021-01-01"))


class TestDisclosureDeadline(unittest.TestCase):
    def test_four_standard_periods(self):
        self.assertEqual(disclosure_deadline("2022-03-31"), "2022-04-30")
        self.assertEqual(disclosure_deadline("2022-06-30"), "2022-08-31")
        self.assertEqual(disclosure_deadline("2022-09-30"), "2022-10-31")
        self.assertEqual(disclosure_deadline("2022-12-31"), "2023-04-30")

    def test_rejects_non_standard(self):
        with self.assertRaises(ValueError):
            disclosure_deadline("2022-05-15")


class TestPITFundamentals(unittest.TestCase):
    def setUp(self):
        self.f = PITFundamentals()
        c = "000001.SZ"
        self.f.add(Report(c, "2021-03-31", "2021-04-28", eps=0.20, bps=5.0))
        self.f.add(Report(c, "2021-12-31", "2022-04-20", eps=1.00, bps=5.0))
        self.f.add(Report(c, "2022-03-31", "2022-04-28", eps=0.30, bps=5.2))
        # 事后修订：公告日更晚、版本号更高
        self.f.add(Report(c, "2021-12-31", "2022-09-01", eps=0.55, bps=5.0,
                          version=2, restated=True))

    def test_announcement_date_not_period_gates_visibility(self):
        # 2022-04-25 时 2022Q1 尚未公告，最新可见仍是 2021 年报
        at = AsOf("2022-04-25")
        latest = self.f.field("000001.SZ", at, "eps")
        self.assertEqual(latest, 1.00)
        self.assertEqual(len(self.f.visible("000001.SZ", at)), 2)

    def test_restatement_invisible_before_its_announcement(self):
        eps_apr = self.f.latest_per_period("000001.SZ", AsOf("2022-05-01"))["2021-12-31"].eps
        eps_oct = self.f.latest_per_period("000001.SZ", AsOf("2022-10-01"))["2021-12-31"].eps
        self.assertEqual(eps_apr, 1.00)
        self.assertEqual(eps_oct, 0.55)
        self.assertTrue(self.f.visible("000001.SZ", AsOf("2022-10-01"))[-1].restated)

    def test_ttm_eps_uses_only_visible_reports(self):
        # 2022Q1 TTM = 0.30 + 2021年报 1.00 - 2021Q1 0.20 = 1.10
        self.assertAlmostEqual(
            self.f.ttm_eps("000001.SZ", AsOf("2022-05-01")), 1.10, places=8)
        # 2022-04-25 只能看到 2021 年报
        self.assertEqual(self.f.ttm_eps("000001.SZ", AsOf("2022-04-25")), 1.00)

    def test_add_rejects_announcement_before_period(self):
        with self.assertRaises(ValueError):
            self.f.add(Report("X", "2022-12-31", "2022-01-01", eps=1.0))


class TestUniverse(unittest.TestCase):
    def setUp(self):
        self.u = Universe()
        self.u.add(Listing("A", "2020-01-01"))
        self.u.add(Listing("B", "2020-01-01", delisted_on="2022-06-30",
                           st_windows=(("2021-01-01", "2021-03-31"),)))

    def test_delisted_stock_stays_before_delisting(self):
        self.assertIn("B", self.u.members(AsOf("2022-06-29")))
        self.assertNotIn("B", self.u.members(AsOf("2022-06-30")))
        self.assertTrue(self.u.is_delisted("B", AsOf("2022-06-30")))

    def test_st_window_is_not_treated_as_missing(self):
        self.assertTrue(self.u.is_st("B", AsOf("2021-02-01")))
        self.assertFalse(self.u.is_st("B", AsOf("2021-04-01")))
        self.assertFalse(self.u.is_st("A", AsOf("2021-02-01")))

    def test_survivorship_bias_is_measurable(self):
        self.assertEqual(self.u.survivors_only_count(AsOf("2022-01-01")), 1)
        self.assertEqual(len(self.u.members(AsOf("2022-01-01"))), 2)

    def test_invalid_listing_rejected(self):
        with self.assertRaises(ValueError):
            self.u.add(Listing("C", "2022-01-01", delisted_on="2021-01-01"))


class TestCalendar(unittest.TestCase):
    def setUp(self):
        self.dates = TradingCalendar.business_days("2022-01-03", 70)
        self.cal = TradingCalendar(self.dates)

    def test_every_rebalance_date_is_a_real_session(self):
        for freq in ("weekly", "monthly", "quarterly"):
            rb = self.cal.rebalance_dates("2022-01-03", "2022-12-31", freq)
            self.assertTrue(rb)
            self.assertTrue(all(self.cal.contains(d) for d in rb))
            self.assertEqual(len(rb), len(set(rb)))
        self.assertLess(len(self.cal.rebalance_dates("2022-01-03", "2022-12-31", "quarterly")),
                        len(self.cal.rebalance_dates("2022-01-03", "2022-12-31", "monthly")))

    def test_unknown_frequency_rejected(self):
        with self.assertRaises(ValueError):
            self.cal.rebalance_dates("2022-01-03", "2022-12-31", "every_20_rows")

    def test_next_session(self):
        self.assertEqual(self.cal.next_session(self.dates[0]), self.dates[1])
        self.assertIsNone(self.cal.next_session(self.dates[-1]))


class TestPanel(unittest.TestCase):
    def setUp(self):
        self.p = Panel()
        for d in ("2022-01-04", "2022-01-05", "2022-01-06"):
            self.p.add(_bar(d))

    def test_history_never_returns_future_bars(self):
        h = self.p.history("000001.SZ", AsOf("2022-01-05"), 10)
        self.assertEqual(tuple(b.date for b in h), ("2022-01-04", "2022-01-05"))
        self.assertTrue(all(b.date <= "2022-01-05" for b in h))

    def test_lookback_window(self):
        h = self.p.history("000001.SZ", AsOf("2022-01-06"), 2)
        self.assertEqual(len(h), 2)
        self.assertEqual(h[-1].date, "2022-01-06")

    def test_direct_future_read_raises(self):
        with self.assertRaises(LookaheadError):
            self.p.get("000001.SZ", "2022-01-06", AsOf("2022-01-05"))

    def test_bars_after_counts_only_future(self):
        self.assertEqual(self.p.bars_after("2022-01-05"), 1)
        self.assertEqual(self.p.bars_after("2099-01-01"), 0)

    def test_cache_invalidated_on_add(self):
        self.assertEqual(len(self.p.history("000001.SZ", AsOf("2022-01-06"), 10)), 3)
        self.p.add(_bar("2022-01-07"))
        self.assertEqual(len(self.p.history("000001.SZ", AsOf("2022-01-07"), 10)), 4)


class TestSyntheticMarket(unittest.TestCase):
    def test_injects_all_the_hard_cases(self):
        m = generate_synthetic_market(n_codes=40, n_days=260, seed=123)
        future = AsOf("9999-12-31")
        all_bars = [b for c in m.panel.codes()
                    for b in m.panel.history(c, future, 10 ** 9)]
        self.assertTrue(any(b.suspended for b in all_bars), "未注入停牌")
        self.assertTrue(any(b.limit_up for b in all_bars), "未注入一字涨停")
        self.assertTrue(any(b.limit_down for b in all_bars), "未注入一字跌停")
        self.assertTrue(any(m.universe.get(c).delisted_on for c in m.universe.all_codes()),
                        "未注入退市")
        self.assertTrue(any(m.universe.get(c).st_windows for c in m.universe.all_codes()),
                        "未注入 ST 窗口")
        self.assertTrue(any(r.restated for c in m.universe.all_codes()
                            for r in m.fundamentals.visible(c, future)),
                        "未注入财报修订")

    def test_deterministic(self):
        a = generate_synthetic_market(n_codes=8, n_days=80, seed=9)
        b = generate_synthetic_market(n_codes=8, n_days=80, seed=9)
        c = generate_synthetic_market(n_codes=8, n_days=80, seed=10)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), c.fingerprint())

    def test_truncate_physically_removes_future(self):
        m = generate_synthetic_market(n_codes=12, n_days=120, seed=5)
        cut = m.calendar.dates[70]
        t = m.truncate(cut)
        self.assertEqual(t.panel.bars_after(cut), 0)
        self.assertGreater(m.panel.bars_after(cut), 0)
        future = AsOf("9999-12-31")
        self.assertEqual(sum(1 for c in t.universe.all_codes()
                             for r in t.fundamentals.visible(c, future)
                             if r.announced_at > cut), 0)


if __name__ == "__main__":
    unittest.main()
