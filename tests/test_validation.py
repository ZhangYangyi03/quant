"""验证层验收：统计检验、真 walk-forward、未来函数检验器本身的功效。"""
import math
import random
import unittest

from quantlab.config import Config, BacktestConfig
from quantlab.data import generate_synthetic_market
from quantlab.validate.stats import (performance, newey_west_t, deflated_sharpe,
                                     pbo, bootstrap_max_sharpe_p, annualized_sharpe)
from quantlab.validate.walkforward import (run_walkforward, detect_lookahead,
                                           WalkForwardLeakError)
from quantlab.validate.no_lookahead import verify_no_lookahead, make_cheating_signal


def _param_returns(seed=3, n=200, k=4):
    rng = random.Random(seed)
    return {f"P{i}": [rng.gauss(0.0003 + 0.0001 * i, 0.01) for _ in range(n)]
            for i in range(k)}


def _honest_choose(hist, future=None):
    """只看历史均值，完全忽略蜜罐参数。"""
    return max(hist, key=lambda k: sum(hist[k]) / max(len(hist[k]), 1))


class TestPerformance(unittest.TestCase):
    def test_multiple_and_drawdown(self):
        m = performance([("d0", 100.0), ("d1", 110.0), ("d2", 99.0)])
        self.assertAlmostEqual(m["total_multiple"], 0.99, places=6)
        self.assertLess(m["max_drawdown_pct"], 0.0)
        self.assertEqual(m["days"], 3)

    def test_too_short_curve(self):
        self.assertEqual(performance([("d0", 100.0)]), {"error": "too_short"})

    def test_annualized_sharpe_uses_250_days(self):
        rng = random.Random(1)
        xs = [rng.gauss(0.001, 0.01) for _ in range(500)]
        self.assertAlmostEqual(annualized_sharpe(xs),
                               sum(xs) / len(xs) /
                               (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) /
                                (len(xs) - 1)) ** 0.5 * math.sqrt(250), places=8)


class TestStatisticalTests(unittest.TestCase):
    def test_newey_west_t_detects_persistent_drift(self):
        t = newey_west_t([0.002] * 120)
        self.assertGreater(t, 2.0)
        rng = random.Random(11)
        t0 = newey_west_t([rng.gauss(0, 0.01) for _ in range(300)])
        self.assertLess(abs(t0), 4.0)

    def test_deflated_sharpe_single_trial_does_not_crash(self):
        v = deflated_sharpe(0.05, 1, 250)
        self.assertTrue(0.0 <= v <= 1.0)
        many = deflated_sharpe(0.05, 100, 250, variance_of_sr=0.01)
        self.assertTrue(0.0 <= many <= 1.0)

    def test_pbo_and_bootstrap_return_valid_probabilities(self):
        data = _param_returns()
        self.assertTrue(0.0 <= pbo(data, slices=8) <= 1.0)
        self.assertTrue(0.0 <= bootstrap_max_sharpe_p(data, draws=60) <= 1.0)
        self.assertEqual(pbo({"A": [0.01] * 8}), 0.5)   # 配置不足，中性返回


class TestWalkForward(unittest.TestCase):
    def test_only_history_is_used_and_oos_blocks_are_disjoint(self):
        res = run_walkforward(_param_returns(), _honest_choose, 60, 20)
        self.assertEqual(len(res), 7)
        for i, r in enumerate(res):
            self.assertEqual(r.history_len, 60 + i * 20)
            self.assertEqual(r.period, i)
            self.assertIn(r.chosen, {"P0", "P1", "P2", "P3"})
        self.assertTrue(all(r.history_len < 200 for r in res))

    def test_unknown_param_rejected(self):
        with self.assertRaises(ValueError):
            run_walkforward(_param_returns(), lambda h: "NOPE", 60, 20)

    def test_strict_mode_catches_history_mutation(self):
        def vandal(hist, future=None):
            for v in hist.values():
                v.append(1.0)
            return list(hist)[0]
        with self.assertRaises(WalkForwardLeakError):
            run_walkforward(_param_returns(), vandal, 60, 20, strict=True)


class TestLookaheadDetector(unittest.TestCase):
    def _dataset(self):
        return _param_returns(seed=5)

    def test_honest_chooser_is_clean(self):
        rep = detect_lookahead(_honest_choose, self._dataset(), 60, 20)
        self.assertFalse(rep["leak_detected"])
        self.assertEqual(rep["changed_ratio"], 0.0)
        self.assertGreater(rep["tests"], 0)

    def test_honeypot_cheater_is_caught(self):
        def cheat(hist, future=None):
            if future:
                return max(future, key=lambda k: sum(future[k][-20:]))
            return _honest_choose(hist)
        rep = detect_lookahead(cheat, self._dataset(), 60, 20)
        self.assertTrue(rep["leak_detected"])
        self.assertGreater(rep["changed_ratio"], 0.0)

    def test_closure_cheater_is_caught_and_data_is_restored(self):
        data = self._dataset()
        before = {k: list(v) for k, v in data.items()}
        box = [data]

        def cheat(hist, future=None):
            return max(box[0], key=lambda k: sum(box[0][k][-20:]))

        rep = detect_lookahead(cheat, data, 60, 20)
        self.assertTrue(rep["leak_detected"])
        self.assertEqual(data, before, "检验过程不得污染调用方数据")

    def test_short_history_is_neutral(self):
        rep = detect_lookahead(_honest_choose, {"A": [0.01] * 10}, 60, 20)
        self.assertEqual(rep["tests"], 0)
        self.assertFalse(rep["leak_detected"])


class TestNoLookaheadAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market = generate_synthetic_market(n_codes=20, n_days=200, seed=77)
        cls.cfg = Config(backtest=BacktestConfig(start="2020-01-01",
                                                 end="2020-12-31", top_k=6))

    def test_clean_strategy_passes_all_three_lines_of_evidence(self):
        rep = verify_no_lookahead(self.market, self.cfg)
        self.assertTrue(rep["cross_section_pit"])
        self.assertTrue(rep["data_actually_removed"])
        self.assertEqual(rep["data_removed"]["truncated_bars_after_cut"], 0)
        self.assertEqual(rep["data_removed"]["truncated_reports_after_cut"], 0)
        self.assertTrue(rep["strategy_prefix"]["aligned"])
        self.assertEqual(rep["strategy_prefix"]["max_abs_equity_diff"], 0.0)
        self.assertTrue(rep["equity_invariant"])
        self.assertTrue(rep["honeypot_caught"])
        self.assertTrue(rep["passed"])

    def test_injected_leak_is_actually_rejected(self):
        """检验器不能只会对干净策略点头 —— 喂一个作弊信号必须判失败。"""
        dates = list(self.market.calendar.between(self.cfg.backtest.start,
                                                  self.cfg.backtest.end))
        cut = dates[int(len(dates) * 0.6)]
        cheat = make_cheating_signal(cut)
        rep = verify_no_lookahead(self.market, self.cfg, signal_fn=cheat,
                                  cut=cut, deep_power_check=False)
        self.assertFalse(rep["equity_invariant"])
        self.assertFalse(rep["passed"])

    def test_cut_must_be_a_real_session(self):
        with self.assertRaises(ValueError):
            verify_no_lookahead(self.market, self.cfg, cut="2020-06-06")  # 周六


if __name__ == "__main__":
    unittest.main()
