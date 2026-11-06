# -*- coding: utf-8 -*-
"""verdict_all.py —— 一键输出两条赛道的最终判定（个股 / ETF）。"""
import os, json, sys
Q = os.path.dirname(os.path.abspath(__file__))

def main():
    a = json.load(open(os.path.join(Q, "audit_verdict.json"), encoding="utf-8"))
    e = json.load(open(os.path.join(Q, "etf_verdict.json"), encoding="utf-8"))
    out = {
        "stock_cross_section_alpha": {
            "verdict": a["verdict"], "pass": a["pass"],
            "positive_years": a["positive_years"], "total_years": a["total_years"],
            "compound_excluding_best": a["compound_excluding_best"],
        },
        "etf_lowvol_rotation_alpha": {
            "verdict": e["verdict"], "pass": e["pass"],
            "positive_years": e["positive_years"], "total_years": e["total_years"],
            "compound_excluding_best": e["compound_excluding_best"],
        },
        "note": ("同一套 audit_backtest.py 判据。个股被涨停/停牌不可买判负；"
                 "ETF 无一字板，lowvol60 通过全部审计。见 ETF_ALPHA.md / EXEC_CONSTRAINTS.md"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(Q, "verdict_all.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
