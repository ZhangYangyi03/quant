# -*- coding: utf-8 -*-
"""audit_all.py —— 一键对逐年动态全市场结果跑三层审计 + 跨年度判定。"""
import os, json, sys, glob
Q = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, Q)
from audit_backtest import multi_year_verdict

def main():
    reps = []
    for fp in sorted(glob.glob(os.path.join(Q, "audit_20*.json"))):
        y = int(os.path.basename(fp)[6:10])
        reps.append((y, json.load(open(fp, encoding="utf-8"))))
    if not reps:
        print("没有 audit_*.json；先逐年跑 qlib_bridge/audit_year.py")
        return
    v = multi_year_verdict(reps)
    print(json.dumps(v, ensure_ascii=False, indent=1))
    json.dump(v, open(os.path.join(Q, "audit_verdict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
