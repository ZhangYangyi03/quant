# -*- coding: utf-8 -*-
"""fetch_alt_data.py —— 从东财 datacenter 免费公开接口拉非价量面板。
无需注册 / 无需 token / 无需第三方包，纯 urllib。
输出 alt_data/<name>.jsonl，每行一条原始记录（可重跑、可断点续传）。
"""
import os, json, time, urllib.parse, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "alt_data")
os.makedirs(OUT, exist_ok=True)
HDR = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
DC = "https://datacenter-web.eastmoney.com/api/data/v1/get"

DATASETS = {
    # 名称: (reportName, 过滤模板, 排序字段, 起始年, 结束年, columns)
    "lhb": ("RPT_DAILYBILLBOARD_DETAILSNEW",
            "(TRADE_DATE>='%s-01-01')(TRADE_DATE<'%s-01-01')", "TRADE_DATE", 2010, 2027,
            "SECURITY_CODE,SECUCODE,TRADE_DATE,BILLBOARD_NET_AMT,BILLBOARD_BUY_AMT,"
            "BILLBOARD_SELL_AMT,DEAL_NET_RATIO,ACCUM_AMOUNT,FREE_MARKET_CAP,CHANGE_RATE,"
            "TURNOVERRATE,EXPLANATION,SECURITY_NAME_ABBR,TRADE_ID"),
    "holdernum": ("RPT_HOLDERNUM_DET",
            "(END_DATE>='%s-01-01')(END_DATE<'%s-01-01')", "END_DATE", 2010, 2027,
            "SECURITY_CODE,SECUCODE,END_DATE,HOLD_NOTICE_DATE,HOLDER_NUM,PRE_HOLDER_NUM,"
            "HOLDER_NUM_CHANGE,HOLDER_NUM_RATIO,AVG_MARKET_CAP,AVG_HOLD_NUM,"
            "TOTAL_MARKET_CAP,TOTAL_A_SHARES,CLOSE_PRICE"),
    "orghold": ("RPT_F10_MAIN_ORGHOLD",
            "(REPORT_DATE>='%s-01-01')(REPORT_DATE<'%s-01-01')", "REPORT_DATE", 2010, 2027,
            "SECURITY_CODE,SECUCODE,REPORT_DATE,NOTICE_DATE,TOTAL_ORG_NUM,TOTAL_ORG_NUM_CHANGE,"
            "TOTAL_SHARES_RATIO,CHANGE_RATIO,CHANGE_RATIO_CHANGE,IS_INCREASE,LABEL_NAME"),
    "margin": ("RPTA_WEB_RZRQ_GGMX",
            "(DATE>='%s-01-01')(DATE<'%s-01-01')", "DATE", 2010, 2027,
            "SCODE,DATE,RZYE,RZMRE,RZYEZB,RZRQYE,SZ,RQYE"),
}

def fetch_page(rep, filt, cols, n, sortcol, size=500, retry=3):
    url = DC + "?" + urllib.parse.urlencode({
        "reportName": rep, "columns": cols, "filter": filt,
        "pageNumber": n, "pageSize": size, "sortColumns": sortcol, "sortTypes": "1"})
    for k in range(retry):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=25) as r:
                d = json.loads(r.read().decode("utf-8"))
            res = d.get("result") or {}
            return int(res.get("count") or 0), res.get("data") or []
        except Exception as e:
            if k == retry - 1: raise
            time.sleep(1.0 + k)

def run(name, years=None):
    rep, tpl, sortcol, y0, y1, cols = DATASETS[name]
    path = os.path.join(OUT, name + ".jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try: done.add(json.loads(line).get("_year"))
                except Exception: pass
    t0 = time.time(); total = 0
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for y in (years or range(y0, y1)):
            if y in done:
                print("[skip]", name, y, flush=True); continue
            filt = tpl % (y, y + 1)
            c, first = fetch_page(rep, filt, cols, 1, sortcol)
            npages = (c + 499) // 500
            nrow = 0
            for n in range(1, npages + 1):
                _, data = fetch_page(rep, filt, cols, n, sortcol) if n > 1 else (c, first)
                for r in data:
                    r["_year"] = y
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    nrow += 1
            total += nrow
            print("[%s] %d count=%d pages=%d rows=%d %.0fs" %
                  (name, y, c, npages, nrow, time.time() - t0), flush=True)
            f.flush()
    print("[done]", name, "rows", total, "%.0fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    years = [int(a) for a in args if a.isdigit()]
    names = [a for a in args if not a.isdigit()] or ["lhb"]
    for nm in names: run(nm, years or None)
