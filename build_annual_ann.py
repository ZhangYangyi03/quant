# -*- coding: utf-8 -*-
"""build_annual_ann.py —— 从 fund_parts/ 重建年报首次披露日。

背景：report_ann_dates.json 里年报只有 288 条（季报各 6 万条），
原因是早期清洗把「报告期 + 0~法定上限天」之外的 NOTICE_DATE 全当错位剔除，
而东财的年报分片里同一 (code, period) 常有多条记录（原始披露 + 后续更正稿），
更正稿的 NOTICE_DATE 可能晚一两年。取最早那条才是首次披露日。

输出 annual_ann_dates.json：{code: {"YYYY-12-31": "YYYY-MM-DD"}}，
并提供 merge 到 report_ann_dates.json 的选项。
"""
import os, json, glob, collections
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "fund_parts")

def _d(s):
    if not s: return None
    s = str(s)[:10]
    try: return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError: return None

def build():
    """年报分片的坑：每个 (报告期) 分片里混着两类行——
      A) REPORT_DATE 有值、NOTICE_DATE 晚 1~2 年、DATA_STATE=2：更正/追溯稿，不是首次披露；
      B) REPORT_DATE 为空、NOTICE_DATE 在报告期后 0~180 天：真实首次披露（多为业绩快报/年报）。
    报告期从文件名取。只取 B 类；A 类一律丢弃。
    """
    cand = collections.defaultdict(list)
    files = glob.glob(os.path.join(PARTS, "*20??-12-31*.json"))
    for fp in files:
        base = os.path.basename(fp)
        import re
        m = re.search(r"(20\d\d-12-31)", base)
        if not m:
            continue
        period = m.group(1)
        rd_expect = datetime.strptime(period, "%Y-%m-%d").date()
        try: rows = json.load(open(fp, encoding="utf-8"))
        except Exception: continue
        if not isinstance(rows, list): continue
        for r in rows:
            code = str(r.get("SECURITY_CODE", "")).strip()
            nd = _d(r.get("NOTICE_DATE"))
            rd = _d(r.get("REPORT_DATE"))
            if not code or not nd: continue
            if rd is not None:        # A 类：更正/追溯稿，跳过
                continue
            delta = (nd - rd_expect).days
            if 0 <= delta <= 180:     # 年报首次披露窗口
                cand[(code, period)].append(nd)
    out = collections.defaultdict(dict)
    for (code, period), ds in cand.items():
        out[code][period] = min(ds).isoformat()
    return dict(out), len(files), len(cand)

if __name__ == "__main__":
    res, nfiles, npairs = build()
    json.dump(res, open(os.path.join(HERE, "annual_ann_dates.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=0, sort_keys=True)
    n = sum(len(v) for v in res.values())
    print("[annual] files", nfiles, "pairs", npairs, "codes", len(res), "entries", n)
    # 与现有 report_ann_dates.json 对比
    old = json.load(open(os.path.join(HERE, "report_ann_dates.json"), encoding="utf-8"))
    oldn = sum(1 for c, m in old.items() for p in m if p.endswith("-12-31"))
    print("[annual] old annual entries", oldn, "-> new", n)
    # 抽查
    for c in ("000001", "000002", "600000"):
        print("[annual]", c, res.get(c, {}))
