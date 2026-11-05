# -*- coding: utf-8 -*-
"""survivorship_probe.py —— 量化「永久剔除 corrupt 股票」带来的偏差。

统一口径：严格与宽松使用**完全相同的** size+amt20 信号、成本、风控、执行日，
唯一差别是 rp.clean（是否把出现过前复权数据损坏的股票永久移出股票池）。
"""
import os, sys, json, datetime
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strategy_layer as SL


def main():
    rp, M, JJ, EX, NS, icj, cal = SL.load_context()
    clean = SL._clean_from_bad(rp)
    rp.clean = clean
    sig = SL.build_signals(rp, M, JJ, NS, icj)["size_amt20"]
    rp.drop_ann()
    cfg = __import__("quantlab.config", fromlist=["Config"]).Config()
    industries = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    strict, _ = SL.run_config("strict", sig, rp, M, JJ, EX, NS, cfg,
                              aum=1e6, top_k=50, industries=industries, risk_on=True)
    rp.clean = np.ones((rp.ND, rp.NC), dtype=bool)
    loose, _ = SL.run_config("loose", sig, rp, M, JJ, EX, NS, cfg,
                             aum=1e6, top_k=50, industries=industries, risk_on=True)
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "question": "永久剔除 corrupt 股票是否制造幸存者偏差？",
           "method": "严格与宽松共用同一 size+amt20 信号；仅 rp.clean 不同",
           "strict": {k: v for k, v in strict.items() if k != "curve"},
           "loose": {k: v for k, v in loose.items() if k != "curve"},
           "cagr_gap_pp": round(loose["stats"]["cagr_pct"] - strict["stats"]["cagr_pct"], 2)}
    json.dump(out, open(os.path.join(HERE, "survivorship_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("strict CAGR %.2f%%  loose CAGR %.2f%%  gap %.2f pp" % (
        strict["stats"]["cagr_pct"], loose["stats"]["cagr_pct"], out["cagr_gap_pp"]), flush=True)


if __name__ == "__main__":
    main()
