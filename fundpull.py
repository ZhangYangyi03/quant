# -*- coding: utf-8 -*-
"""fundpull.py -- 从东方财富 datacenter 拉基本面/评级/龙虎榜，落盘到 fund_parts/。

用法：python fundpull.py [预算秒数]
按需续传：已存在的文件跳过。
"""
import json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fund_parts")
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(HERE, "fundpull.log")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    print(msg, flush=True)


def em(report, filt="", page=1, size=500, cols="ALL"):
    url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=%s&columns=%s&filter=%s&pageNumber=%d&pageSize=%d"
           % (urllib.parse.quote(report), cols, urllib.parse.quote(filt), page, size))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
    for k in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=20).read().decode("utf-8"))
            d = r.get("result") or {}
            return d.get("pages") or 0, d.get("data") or []
        except Exception:
            if k == 2:
                return None, []
            time.sleep(1.5)


TABLES = [("lico", "RPT_LICO_FN_CPD", "REPORTDATE"),
          ("balance", "RPT_DMSK_FN_BALANCE", "REPORT_DATE"),
          ("income", "RPT_DMSK_FN_INCOME", "REPORT_DATE"),
          ("cash", "RPT_DMSK_FN_CASHFLOW", "REPORT_DATE")]


def quarters(y0=2010, y1=2026):
    for y in range(y0, y1 + 1):
        for md in ("-03-31", "-06-30", "-09-30", "-12-31"):
            yield "%d%s" % (y, md)


def pull_table(tag, rep, key, d):
    p1, _ = em(rep, "(%s='%s')" % (key, d), 1, 500)
    if p1 is None:
        log("MISS %s %s header" % (tag, d))
        return 0
    n = 0
    for page in range(1, p1 + 1):
        fn = os.path.join(OUT, "%s_%s_p%02d.json" % (tag, d, page))
        if os.path.exists(fn):
            n += 1
            continue
        _, rows = em(rep, "(%s='%s')" % (key, d), page, 500)
        if not rows:
            continue
        json.dump(rows, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
        n += 1
    log("%s %s %d/%d pages" % (tag, d, n, p1))
    return n


def pull_ratings():
    """机构评级：每个报告期的买入/增持/中性/减持/卖出家数 + 未来两年预测。"""
    fn = os.path.join(OUT, "rating.json")
    if os.path.exists(fn):
        return 1
    p1, rows = em("RPT_WEB_RESPREDICT", "", 1, 500)
    if p1 is None:
        log("MISS rating")
        return 0
    all_rows = []
    for page in range(1, p1 + 1):
        _, r = em("RPT_WEB_RESPREDICT", "", page, 500)
        all_rows.extend(r)
    json.dump(all_rows, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
    log("rating %d rows" % len(all_rows))
    return 1


def pull_billboard(d0="2025-01-01"):
    """龙虎榜：按交易日拉。"""
    p1, rows = em("RPT_DAILYBILLBOARD_DETAILSNEW", "", 1, 500)
    if p1 is None:
        log("MISS billboard")
        return 0
    all_rows = []
    for page in range(1, min(p1, 40) + 1):
        _, r = em("RPT_DAILYBILLBOARD_DETAILSNEW", "", page, 500)
        all_rows.extend(r)
    fn = os.path.join(OUT, "billboard.json")
    json.dump(all_rows, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
    log("billboard %d rows" % len(all_rows))
    return 1


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 600
    t0 = time.time()
    log("start budget=%.0fs" % budget)
    for tag, rep, key in TABLES:
        for d in quarters():
            if time.time() - t0 > budget:
                log("budget reached")
                return
            pull_table(tag, rep, key, d)
    if time.time() - t0 < budget:
        pull_ratings()
    if time.time() - t0 < budget:
        pull_billboard()
    log("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
