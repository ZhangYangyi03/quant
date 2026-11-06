# -*- coding: utf-8 -*-
"""fetch_margin_monthly.py —— 两融月度末截面（只抓回测调仓日，不抓逐日全量）。
理由：逐日全量是千万级，而策略月度调仓只需要 170 个截面。
"""
import os, json, time, urllib.parse, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "alt_data")
HDR = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"
COLS = "SCODE,DATE,RZYE,RZMRE,RZYEZB,RZRQYE,SZ,RQYE"
REP = "RPTA_WEB_RZRQ_GGMX"

def fetch(day, page, size=500, retry=3):
    url = DC + "?" + urllib.parse.urlencode({
        "reportName": REP, "columns": COLS, "filter": "(DATE='%s')" % day,
        "pageNumber": page, "pageSize": size, "sortColumns": "SCODE", "sortTypes": "1"})
    for k in range(retry):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
            res = d.get("result") or {}
            return int(res.get("count") or 0), res.get("data") or []
        except Exception:
            if k == retry - 1: return 0, []
            time.sleep(0.8 + k)

if __name__ == "__main__":
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    dates = meta["dates"]
    ends = {}
    for d in dates:
        if d >= "2012-01-01": ends[d[:7]] = d
    days = list(ends.values())
    path = os.path.join(OUT, "margin_monthly.jsonl")
    done = set()
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            try: done.add(json.loads(ln)["_day"])
            except Exception: pass
    t0 = time.time(); total = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for i, day in enumerate(days):
            if day in done: continue
            c, first = fetch(day, 1)
            npages = max(1, (c + 499) // 500)
            n = 0
            for pg in range(1, npages + 1):
                _, data = (c, first) if pg == 1 else fetch(day, pg)
                for r in data:
                    r["_day"] = day
                    f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
            total += n
            if i % 20 == 0 or i == len(days) - 1:
                print("[margin] %s %d/%d day=%s count=%d rows=%d %.0fs" %
                      (day, i + 1, len(days), day, c, n, time.time() - t0), flush=True)
            f.flush()
    print("[done] margin monthly rows", total, "%.0fs" % (time.time() - t0), flush=True)
