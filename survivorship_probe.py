# -*- coding: utf-8 -*-
"""survivorship_probe.py —— 量化「永久剔除 corrupt 股票」带来的幸存者偏差。

data_quality.py 把出现过前复权数据损坏的股票从池子里永久剔除（clean 只用过去），
这会引入幸存者偏差：被剔掉的多是长期停牌/重组/微盘股，往往不是随机样本。
本探针用「不剔除 corrupt、只在当日 corrupt 时跳过计价」的宽松口径，
与统一管线的严格口径对比，给出这个偏差的方向与大小。

用法：python survivorship_probe.py （输出 survivorship_result.json）
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
    clean = np.empty((rp.ND, rp.NC), dtype=bool)
    running = np.zeros(rp.NC, dtype=bool)
    for j in range(rp.ND):
        running |= rp.bad[j].astype(bool)
        clean[j] = ~running
    rp.clean = clean
    sig = SL.build_signals(rp, M, JJ, NS, icj)
    cfg = __import__("quantlab.config", fromlist=["Config"]).Config()
    industries = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    # 严格口径（与 unified_result 一致）
    strict, _ = SL.run_config("strict", sig["size_amt20"], rp, M, JJ, EX, NS, cfg,
                              aum=1e6, top_k=50, industries=industries, risk_on=True)
    # 宽松口径：把 clean 全开，仅保留 halt 与当日 corrupt 的计价保护
    rp.clean = np.ones((rp.ND, rp.NC), dtype=bool)
    loose_sig = SL.build_signals(rp, M, JJ, NS, icj)
    loose, _ = SL.run_config("loose", loose_sig["size_amt20"], rp, M, JJ, EX, NS, cfg,
                             aum=1e6, top_k=50, industries=industries, risk_on=True)
    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "question": "永久剔除 corrupt 股票是否制造幸存者偏差？",
           "strict": {k: v for k, v in strict.items() if k != "curve"},
           "loose": {k: v for k, v in loose.items() if k != "curve"},
           "cagr_gap_pp": round(loose["stats"]["cagr_pct"] - strict["stats"]["cagr_pct"], 2)}
    json.dump(out, open(os.path.join(HERE, "survivorship_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("strict CAGR %.2f%%  loose CAGR %.2f%%  gap %.2f pp" % (
        strict["stats"]["cagr_pct"], loose["stats"]["cagr_pct"], out["cagr_gap_pp"]), flush=True)
    print("strict dead", strict["delisted_mark_events"], "loose dead", loose["delisted_mark_events"], flush=True)


if __name__ == "__main__":
    main()
