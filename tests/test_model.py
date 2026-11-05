"""模型层验收：成本口径、因子预处理、权重约束、风险分解。"""
import unittest

from quantlab.clock import AsOf
from quantlab.config import CostConfig
from quantlab.data import generate_synthetic_market
from quantlab.model.cost import compute_cost, impact_bps, max_notional
from quantlab.model.factors import winsorize, zscore, rank, neutralize, spearman_ic
from quantlab.model.portfolio import (build_weights, equal_weight,
                                      inverse_vol_weights, _enforce_cap)
from quantlab.model.risk import covariance, portfolio_risk, style_exposures


class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.cc = CostConfig()

    def test_stamp_duty_cutover_2023_08_28(self):
        before = compute_cost("SELL", 100_000, "2023-08-27", 1e7, self.cc)
        on = compute_cost("SELL", 100_000, "2023-08-28", 1e7, self.cc)
        self.assertAlmostEqual(before.stamp, 100.0)
        self.assertAlmostEqual(on.stamp, 50.0)
        self.assertEqual(compute_cost("BUY", 100_000, "2023-08-27", 1e7, self.cc).stamp, 0.0)

    def test_min_commission_and_transfer(self):
        small = compute_cost("BUY", 100.0, "2022-01-01", 1e7, self.cc)
        self.assertEqual(small.commission, 5.0)
        large = compute_cost("BUY", 100_000.0, "2022-01-01", 1e7, self.cc)
        self.assertAlmostEqual(large.commission, 25.0)
        self.assertAlmostEqual(large.transfer, 10.0)

    def test_impact_and_slippage_are_mutually_exclusive(self):
        imp = CostConfig(use_impact_model=True)
        slip = CostConfig(use_impact_model=False)
        a = compute_cost("BUY", 100_000, "2022-01-01", 1e7, imp)
        b = compute_cost("BUY", 100_000, "2022-01-01", 1e7, slip)
        self.assertGreater(a.impact, 0.0)
        self.assertEqual(a.slippage, 0.0)
        self.assertGreater(b.slippage, 0.0)
        self.assertEqual(b.impact, 0.0)

    def test_impact_is_sqrt_in_participation(self):
        self.assertEqual(impact_bps(0.0, self.cc), 0.0)
        p4 = impact_bps(0.04, self.cc)
        p1 = impact_bps(0.01, self.cc)
        self.assertAlmostEqual(p4 / p1, 2.0, places=6)

    def test_capacity_and_bad_inputs(self):
        self.assertAlmostEqual(max_notional(1e6, self.cc), 1e5)
        self.assertEqual(max_notional(0.0, self.cc), 0.0)
        with self.assertRaises(ValueError):
            compute_cost("HOLD", 1.0, "2022-01-01", 1e7, self.cc)


class TestFactors(unittest.TestCase):
    def test_winsorize_clips_outlier_and_preserves_none(self):
        xs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0, None]
        out = winsorize(xs, k=3.0)
        self.assertLess(out[7], 1000.0)
        self.assertIsNone(out[8])
        self.assertAlmostEqual(out[0], 1.0)

    def test_zscore_is_standardized(self):
        out = zscore([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = sum(out) / len(out)
        self.assertAlmostEqual(mu, 0.0, places=10)
        self.assertAlmostEqual((sum(x * x for x in out) / (len(out) - 1)) ** 0.5,
                               1.0, places=10)

    def test_rank_handles_ties(self):
        out = rank([1.0, 1.0, 2.0, 2.0])
        self.assertAlmostEqual(out[0], out[1])
        self.assertAlmostEqual(out[2], out[3])
        self.assertLess(out[0], out[2])

    def test_spearman_ic_monotone(self):
        x = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(spearman_ic(x, [2, 4, 6, 8, 10]), 1.0, places=10)
        self.assertAlmostEqual(spearman_ic(x, [10, 8, 6, 4, 2]), -1.0, places=10)
        self.assertIsNone(spearman_ic([1, 2], [1, 2]))       # 样本太小不算 IC

    def test_neutralize_removes_linear_exposure(self):
        xs = [float(i) for i in range(10)]
        y = [3.0 + 2.0 * x for x in xs]
        resid = neutralize(y, [[1.0] * len(xs), xs])
        self.assertLess(max(abs(r) for r in resid), 1e-6)


class TestPortfolio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = generate_synthetic_market(n_codes=30, n_days=90, seed=31)
        cls.cs = cls.market.view(cls.market.calendar.dates[40])

    def _scores(self):
        return {c: float(i) for i, c in enumerate(self.cs.members)}

    def test_weights_are_long_only_sum_one_and_capped(self):
        for method in ("equal", "inverse_vol", "min_variance"):
            w = build_weights(self._scores(), self.cs, top_k=12,
                              method=method, max_weight=0.10)
            self.assertTrue(w, f"{method} 返回空权重")
            self.assertLessEqual(len(w), 12)
            self.assertAlmostEqual(sum(w.values()), 1.0, places=8,
                                   msg=f"{method} 权重和不是 1")
            self.assertTrue(all(v >= -1e-12 for v in w.values()))
            self.assertLessEqual(max(w.values()), 0.10 + 1e-9)

    def test_min_variance_no_longer_leaks_cash(self):
        # 历史实现只做 min(v, cap) 不归一化，权重和会 <1，等于凭空留现金
        w = build_weights(self._scores(), self.cs, top_k=12, method="min_variance")
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)

    def test_enforce_cap_on_skewed_weights(self):
        w = _enforce_cap({"A": 0.9, "B": 0.05, "C": 0.05}, 0.35)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)
        self.assertLessEqual(max(w.values()), 0.35 + 1e-9)

    def test_equal_and_inverse_vol_helpers(self):
        self.assertAlmostEqual(sum(equal_weight(["A", "B"]).values()), 1.0)
        w = inverse_vol_weights(["A", "B"], {"A": 0.01, "B": 0.04})
        self.assertGreater(w["A"], w["B"])

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            build_weights(self._scores(), self.cs, 10, method="magic")


class TestRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = generate_synthetic_market(n_codes=20, n_days=90, seed=44)
        cls.cs = cls.market.view(cls.market.calendar.dates[60])

    def test_covariance_symmetric_and_positive_diagonal(self):
        codes = list(self.cs.members)[:8]
        cov = covariance(self.cs, codes)
        for a in codes:
            self.assertGreater(cov[a][a], 0.0)
            for b in codes:
                self.assertAlmostEqual(cov[a][b], cov[b][a], places=12)

    def test_portfolio_risk_decomposition_nonnegative(self):
        codes = list(self.cs.members)[:8]
        w = equal_weight(codes)
        r = portfolio_risk(w, covariance(self.cs, codes))
        self.assertGreaterEqual(r["variance"], 0.0)
        self.assertGreaterEqual(r["factor_var"], 0.0)
        self.assertAlmostEqual(r["volatility"] ** 2, r["variance"], places=12)

    def test_style_exposures_use_only_past_history(self):
        code = list(self.cs.members)[0]
        e = style_exposures(self.cs, code)
        self.assertEqual(set(e), {"size", "volatility", "momentum", "liquidity"})
        self.assertTrue(all(h.date <= self.cs.at.date
                            for h in self.cs.history[code]))


if __name__ == "__main__":
    unittest.main()
