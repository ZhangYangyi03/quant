# -*- coding: utf-8 -*-
"""策略层新增功能的回归测试：真实公告日、ST 判废、high/low、统计检验接线。"""
import os, sys, json, unittest, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import strategy_layer as SL


class TestAnnDates(unittest.TestCase):
    def test_report_ann_dates_clean(self):
        d = json.load(open(os.path.join(HERE, "report_ann_dates.json"), encoding="utf-8"))
        self.assertGreater(len(d), 5000)
        # 清洗后不允许出现「2010 年报 2012 年公告」这类明显错位
        for c, per in list(d.items())[:200]:
            for period, ann in per.items():
                self.assertLess(int(ann[:4]) - int(period[:4]), 2, (c, period, ann))

    def test_st_history_is_rejected(self):
        v = json.load(open(os.path.join(HERE, "st_history_verdict.json"), encoding="utf-8"))
        self.assertIn("UNUSABLE", v["verdict"])
        self.assertIn("回填", v["reason"])

    def test_visible_date_uses_real_ann(self):
        rp = SL.RealPanel()
        d = rp.ann("000001", "2010-03-31")
        self.assertEqual(d, "2010-04-29")
        self.assertEqual(SL.visible_date(rp, "000001", "2010-03-31"), "2010-04-29")
        # 缺失时退回法定截止日
        self.assertTrue(SL.visible_date(rp, "NOPE_CODE", "2020-03-31"))
        rp.drop_ann()


class TestHighLow(unittest.TestCase):
    def test_hl_exec_shape(self):
        import numpy as np
        z = np.load(os.path.join(HERE, "hl_exec.npz"))
        self.assertEqual(len(z["dates"]), 166)
        self.assertEqual(z["high"].shape, z["low"].shape)
        self.assertGreater(np.isfinite(z["high"]).sum(), 500000)
        self.assertTrue(np.all(z["high"][np.isfinite(z["high"])] >=
                               z["low"][np.isfinite(z["high"])] - 1e-6))

    def test_flags_at_uses_high_low(self):
        rp = SL.RealPanel()
        M = __import__("fraw").meta_unique()
        JJ = [int(x) for x in M["dates"]]
        S, E = JJ[0], JJ[0] + 1
        rp.clean = SL._clean_from_bad(rp)
        fl = SL.flags_at(rp, S, E, [0, 1, 2])
        self.assertIn("one_word_board", fl[0])
        self.assertIn("decision_price", fl[0])
        rp.drop_ann()


class TestUnifiedResult(unittest.TestCase):
    def test_validation_and_resolved(self):
        j = json.load(open(os.path.join(HERE, "unified_result.json"), encoding="utf-8"))
        self.assertIn("validation", j)
        self.assertIn("pbo", j["validation"])
        self.assertIn("walkforward", j["validation"])
        self.assertGreaterEqual(len(j["validation"]["per_config"]), 7)
        self.assertGreaterEqual(len(j["panel"]["resolved"]), 4)
        # EP 用 BM 构造的 HML 后，价值暴露应当显著为正
        self.assertGreater(j["attribution"]["ep_top50"]["beta_hml"], 0.3)
        # 严格 / 宽松打分 / 宽松池三者不得再出现口径混用造成的假差距
        a = j["results"]["size_amt20_top50"]["stats"]["cagr_pct"]
        b = j["results"]["loose_size_amt20_top50"]["stats"]["cagr_pct"]
        c = j["results"]["pool_loose_size_amt20_top50"]["stats"]["cagr_pct"]
        self.assertLess(abs(a - b), 1.0)
        self.assertLess(abs(a - c), 1.0)


if __name__ == "__main__":
    unittest.main()
