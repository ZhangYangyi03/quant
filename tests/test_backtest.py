"""端到端验收：可复现性、真实基准、审计留痕、账本健康。"""
import unittest
from dataclasses import replace

from quantlab.backtest.engine import (run_backtest, run_benchmark, default_signal,
                                      Result)
from quantlab.backtest.execution import REJECT
from quantlab.config import Config, BacktestConfig, CostConfig
from quantlab.data import generate_synthetic_market
from quantlab.reporting import SCHEMA, result_record


class BacktestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = generate_synthetic_market(n_codes=24, n_days=260, seed=2026)
        cls.cfg = Config(backtest=BacktestConfig(start="2020-01-01",
                                                 end="2020-12-31", top_k=8))
        cls.result = run_backtest(cls.market, cls.cfg)

    def test_equity_curve_covers_every_session(self):
        dates = self.market.calendar.between(self.cfg.backtest.start,
                                             self.cfg.backtest.end)
        self.assertEqual(len(self.result.equity_curve), len(dates))
        self.assertEqual(self.result.equity_curve[0][0], dates[0])
        self.assertTrue(all(v > 0 for _, v in self.result.equity_curve))
        self.assertEqual(len(self.result.daily_returns()),
                         len(self.result.equity_curve) - 1)

    def test_bit_identical_reproducibility(self):
        again = run_backtest(self.market, self.cfg)
        self.assertEqual(self.result.equity_curve, again.equity_curve)
        self.assertEqual(self.result.fees_paid, again.fees_paid)
        self.assertEqual(self.result.config_fingerprint, again.config_fingerprint)
        self.assertEqual(self.result.data_fingerprint, again.data_fingerprint)
        self.assertEqual(self.result.data_fingerprint, self.market.fingerprint())

    def test_config_fingerprint_tracks_every_parameter(self):
        self.assertNotEqual(
            self.cfg.fingerprint(),
            Config(backtest=replace(self.cfg.backtest, top_k=9)).fingerprint())
        self.assertNotEqual(
            self.cfg.fingerprint(),
            Config(cost=replace(self.cfg.cost, stamp_bps_after=4.0)).fingerprint())
        self.assertEqual(self.cfg.fingerprint(), self.cfg.fingerprint())

    def test_benchmark_is_a_rotating_basket_not_a_fixed_one(self):
        # 历史缺陷：eq_nofilter 既不排序也不抽样，16 年只成交 993 笔，
        # 实际是 fundamentals.json 键序里固定的前 k 只老股。
        bench = run_benchmark(self.market, self.cfg)
        self.assertEqual(bench.label, "benchmark_random_k_equal")
        self.assertGreater(len(bench.fills), 60)
        traded = {f.order.code for f in bench.fills if f.status != "REJECTED"}
        self.assertGreater(len(traded), self.cfg.backtest.top_k,
                           "基准必须逐期换仓，不能是固定篮子")
        self.assertEqual(bench.data_fingerprint, self.result.data_fingerprint,
                         "基准与策略必须跑在同一份数据上")
        self.assertNotEqual(bench.config_fingerprint, self.result.config_fingerprint)
        self.assertEqual(len(bench.equity_curve), len(self.result.equity_curve))

    def test_rejections_are_all_accounted_for(self):
        reasons = {getattr(REJECT, k) for k in dir(REJECT) if k.isupper()}
        for f in self.result.rejected:
            self.assertEqual(f.status, "REJECTED")
            self.assertIn(f.reason, reasons)
        self.assertEqual(len(self.result.rejected), sum(self.result.blocked.values()))

    def test_metrics_and_audit_record(self):
        m = self.result.metrics(self.cfg)
        for key in ("cagr_pct", "sharpe", "sortino", "max_drawdown_pct",
                    "ann_vol_pct", "total_multiple", "days"):
            self.assertIn(key, m)
        self.assertLessEqual(m["max_drawdown_pct"], 0.0)

        rec = result_record(self.result, self.cfg)
        self.assertEqual(rec["schema"], SCHEMA)
        self.assertEqual(rec["config_fingerprint"], self.result.config_fingerprint)
        self.assertEqual(rec["data_fingerprint"], self.result.data_fingerprint)
        self.assertEqual(rec["seed"], self.cfg.backtest.seed)
        self.assertEqual(rec["config"], self.cfg.as_dict())
        self.assertEqual(len(rec["equity_curve"]), len(self.result.equity_curve))
        self.assertEqual(rec["trades"]["n_orders"], len(self.result.fills))

    def test_signals_are_pure_views_of_the_cross_section(self):
        cs = self.market.view(self.market.calendar.dates[100])
        fn = default_signal("random", 7)
        self.assertEqual(fn(cs), fn(cs))                    # 确定性
        self.assertTrue(set(fn(cs)).issubset(set(cs.members)))
        with self.assertRaises(ValueError):
            default_signal("does_not_exist")(cs)

    def test_short_window_is_rejected(self):
        cfg = Config(backtest=BacktestConfig(start="2020-01-01", end="2020-01-31"))
        with self.assertRaises(ValueError):
            run_backtest(self.market, cfg)

    def test_cost_change_moves_the_equity_curve(self):
        expensive = Config(cost=CostConfig(commission_bps=80.0),
                           backtest=self.cfg.backtest)
        cheap = run_backtest(self.market, expensive)
        self.assertNotEqual(cheap.equity_curve[-1][1], self.result.equity_curve[-1][1])
        self.assertGreater(cheap.fees_paid, self.result.fees_paid)


if __name__ == "__main__":
    unittest.main()
