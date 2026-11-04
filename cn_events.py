"""A 股当日事件面：把 entity-graph/cnmarket 的公开数据，变成三个横截面信号。

为什么要这一步
--------------
quant 现在所有因子都来自日线和财报（EP/BM/动量/低波…），它们回答的是"这家公司值多少"。
市场里还有一类信息回答的是"今天谁在动它"，公开、免费、当天就有：
    资金     主力净流入 f62 / 净占比 f184
    席位     龙虎榜净买额、机构专用席位上没上
    融资     融资余额日变化率（杠杆资金在加仓还是在撤）
这三个此前完全没有进过信号。cnmarket.py 已经证明拿得到，这里把它们做成
能和现有回测同一个口径（code -> 当日数值）的横截面。

数据纪律
--------
* 只放**当日**能拿到的值，不带未来：龙虎榜是收盘后公布，所以它比资金流晚一天
  （build 时 shift 一天，见 LAG_DAYS）。
* 任何一项拿不到就返回 None，绝不填 0 —— 0 会被当成"没变化"混进排名。
* 每个文件写出 source_url 与 fetched_at，事后能追到是哪个接口哪一秒给的。

用法
----
    python cn_events.py --out events/cn_flow_20261104.json
    python cn_events.py --build-panel          # 扫 events/ 落成 panel 结构
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EG = os.path.join(os.path.dirname(HERE), "entity-graph")
if EG not in sys.path:
    sys.path.insert(0, EG)

LAG_DAYS = {"main_net_pct": 0, "main_net_in": 0, "lhb_net": 1, "lhb_inst": 1,
            "rz_chg": 0, "short_chg": 0}


def _cnmarket():
    from entitygraph import cnmarket          # noqa: PLC0415
    return cnmarket


EM_ULIST = "https://push2.eastmoney.com/api/qt/ulist.np/get"


UNIVERSE_CAP = int(os.environ.get("CN_EVENTS_UNIVERSE", "2000"))


def _universe() -> list[str]:
    """要拿哪些票的数据：优先用 quant 自己的股票池，拿不到才退回样本 20 只。"""
    import glob
    for pat in (os.path.join(HERE, "stockdata", "_universe.json"),
                os.path.join(HERE, "universe.json"),
                os.path.join(HERE, "data", "_universe.json")):
        hits = glob.glob(pat)
        if hits:
            try:
                d = json.load(open(hits[0], encoding="utf-8"))
                # _universe.json 的顶层直接是 {code: {...}}；先认字典，再认 list / 包着的 list
                if isinstance(d, dict) and "codes" not in d and "universe" not in d:
                    codes = list(d.keys())
                elif isinstance(d, list):
                    codes = d
                else:
                    codes = d.get("codes") or d.get("universe") or []
                if isinstance(codes, dict):
                    codes = list(codes.keys())
                codes = [str(c)[:6] for c in codes if c]
                if len(codes) > 100:
                    return codes[:UNIVERSE_CAP]
            except Exception:                    # noqa: BLE001
                pass
    return ["600000", "000001", "300750", "600519", "000858", "601318", "002594",
            "300059", "688981", "601899", "601288", "601398", "601857", "000333",
            "600036", "601166", "002415", "300124", "688111", "601988"]


def _secid(code: str) -> str:
    return ("1." if str(code)[:1] in ("6", "5", "9") else "0.") + str(code)


def snapshot_flow(codes: list[str] | None = None, batch: int = 500) -> dict:
    """主力资金 + 快照（市值/行业/PE），一批 100 只。

    为什么不用 clist 翻页：东财 clist 每页最多 100 条，且翻页会漂（实测 page1 与 page2
    的代码区间重叠），29 页才凑出 2883 个唯一代码。ulist.np 一次传 100 个 secid，
    返回的 total 与请求数一致 —— 不漂。
    """
    import time as _t
    import urllib.request
    codes = codes or _universe()
    out, src = {}, []
    hdr = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/",
           "Connection": "close"}
    fields = "f12,f14,f2,f3,f20,f21,f9,f23,f100,f62,f184"
    for i in range(0, len(codes), batch):
        chunk = codes[i:i + batch]
        url = EM_ULIST + "?fltt=2&secids=" + ",".join(_secid(c) for c in chunk) + "&fields=" + fields
        doc = None
        for attempt in range(3):
            try:
                r = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=25)
                doc = json.loads(r.read())
                r.close()
                break
            except Exception:                    # noqa: BLE001
                _t.sleep(1.5 + attempt)
        if not doc or not doc.get("data"):
            src.append(f"ulist batch {i} FAILED")
            continue
        src.append(url)
        diff = doc["data"].get("diff") or []
        items = diff if isinstance(diff, list) else list(diff.values())
        for r in items:
            c = r.get("f12")
            if not c:
                continue
            out[c] = {"main_net_in": r.get("f62"), "main_net_pct": r.get("f184"),
                      "float_mktcap": r.get("f21"), "total_mktcap": r.get("f20"),
                      "industry": r.get("f100"), "pe_ttm": r.get("f9"),
                      "pb": r.get("f23"), "close": r.get("f2"), "chg_pct": r.get("f3")}
    return {"data": out, "sources": src}


def lhb_flow(max_pages: int = 12) -> dict:
    """龙虎榜：净买额（万元）与是否出现"机构专用"席位。

    pageSize=4000 会被静默截断成 500（实测）。取到最新交易日的全部记录需要翻页，
    翻到出现更早日期为止 —— 少翻一天就是漏掉几十只票的净买额，且不会报错。
    """
    cm = _cnmarket()
    rows, srcs = [], []
    for pn in range(1, max_pages + 1):
        try:
            page, url = cm._em_dc("RPT_DAILYBILLBOARD_DETAILSNEW", pageSize=500, pageNumber=pn,
                                  sortColumns="TRADE_DATE", sortTypes="-1")
        except Exception as exc:                 # noqa: BLE001
            srcs.append(f"lhb page {pn} FAILED {type(exc).__name__}")
            break
        if not page:
            break
        rows.extend(page)
        srcs.append(url)
        dates = {str(r.get("TRADE_DATE"))[:10] for r in page}
        if max(dates) != min(dates) and pn > 1:
            break
    if not rows:
        return {"data": {}, "sources": srcs or ["lhb FAILED"], "date": None}
    newest = max(str(r.get("TRADE_DATE"))[:10] for r in rows)
    rows = [r for r in rows if str(r.get("TRADE_DATE"))[:10] == newest]
    out, day = {}, newest
    for r in rows:
        code = r.get("SECURITY_CODE")
        if not code:
            continue
        day = str(r.get("TRADE_DATE"))[:10]
        seats = f"{r.get('BUY_SEAT') or ''},{r.get('SELL_SEAT') or ''}"
        rec = out.setdefault(code, {"lhb_net": 0.0, "lhb_inst": 0})
        net = r.get("BILLBOARD_NET_AMT")
        if net is not None:
            rec["lhb_net"] += float(net) / 1e4          # 元 -> 万元
        if "机构专用" in seats:
            rec["lhb_inst"] = 1
    return {"data": out, "sources": [url], "date": day}


EM_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def margin_by_code(codes: list[str], day: str | None = None, chunk: int = 40) -> dict:
    """两融按股票批量查，只要最新一天。

    实测 2026-11-04：filter=(SCODE in (...)) 有问必答，但 500 条被**每只股票的历史日期**填满
    （一只票 250 天），排在前面的股票把后面的挤掉 —— sortColumns 对 group by 后的日期不生效。
    所以必须 filter 里同时锁日期，只取那一天。
    """
    import urllib.parse
    import urllib.request
    hdr = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/",
           "Content-Type": "application/x-www-form-urlencoded", "Connection": "close"}
    day = day or time.strftime("%Y-%m-%d")
    out, srcs = {}, []
    for i in range(0, len(codes), chunk):
        part = codes[i:i + chunk]
        flt = ('(SCODE in (' + ",".join(f'"{c}"' for c in part) + f'))(DATE=\'{day}\')')
        body = urllib.parse.urlencode({
            "reportName": "RPTA_WEB_RZRQ_GGMX", "columns": "ALL", "pageSize": 500,
            "pageNumber": 1, "filter": flt}).encode()
        try:
            r = urllib.request.urlopen(urllib.request.Request(EM_DATACENTER, data=body, headers=hdr),
                                       timeout=25)
            doc = json.loads(r.read())
            r.close()
        except Exception as exc:                 # noqa: BLE001
            srcs.append(f"margin chunk {i} FAILED {type(exc).__name__}")
            continue
        data = ((doc.get("result") or {}).get("data")) or []
        srcs.append(f"{EM_DATACENTER}?reportName=RPTA_WEB_RZRQ_GGMX&filter={flt}")
        for r in data:
            c = r.get("SCODE")
            if c and c not in out:
                out[c] = r
    return {"data": out, "sources": srcs, "date": day}


def margin_flow(mg: dict | None = None) -> dict:
    """两融截面。数据来自 margin_by_code（已按最新交易日取好），这里只做字段归一。

    保留这个函数是为了给 build 一个统一的返回形状；真正的取数在 margin_by_code。
    """
    if not mg:
        return {"data": {}, "sources": ["margin: no data"], "date": None}
    out = {}
    for code, r in mg.get("data", {}).items():
        out[code] = {"rz": r.get("RZYE"), "rq_vol": r.get("RQYL"), "rq_amt": r.get("RQYE"),
                     "rz_chg_pct": r.get("FIN_BALANCE_GR"),      # 融资余额日变化率（%，东财原字段）
                     "rz_net": r.get("RZJME"),                   # 融资净买入（元）
                     "rz_net_5d": r.get("RZJME5D"), "rz_net_10d": r.get("RZJME10D"),
                     "rq_net": r.get("RQJMG"),                   # 融券净卖出量（股）
                     "rz_balance_pct": r.get("RZYEZB")}          # 融资余额占流通市值 %
    return {"data": out, "sources": mg.get("sources") or [], "date": mg.get("date")}


def build(out_dir: str) -> str:
    """写一份当日截面。文件名带日期，历史靠每天攒一份。"""
    day = time.strftime("%Y%m%d")
    flow = snapshot_flow()
    lhb = lhb_flow()
    universe = _universe()
    mg = margin_by_code(universe)
    if not mg["data"]:                            # 非交易日：回退到最近一次有数据的日子
        for back in range(1, 8):
            day_try = time.strftime("%Y-%m-%d", time.localtime(time.time() - back * 86400))
            mg = margin_by_code(universe, day=day_try)
            if mg["data"]:
                break
    mg_raw = mg
    mg = margin_flow(mg_raw)
    prev = _previous_file(out_dir)
    prev_mg = {}
    if prev:
        try:
            prev_mg = (json.load(open(prev, encoding="utf-8")) or {}).get("margin_raw") or {}
        except Exception:                        # noqa: BLE001
            prev_mg = {}

    codes = set(flow["data"]) | set(lhb["data"]) | set(mg["data"])
    rows = {}
    for c in codes:
        f = flow["data"].get(c, {})
        l = lhb["data"].get(c, {})
        m = mg["data"].get(c, {})
        # 东财两融表自带 FIN_BALANCE_GR（融资余额日变化率，%）与 RZJME（融资净买入），
        # 不再自己维护"前一天的数"——那个方案第一跑就永远是 None。
        rz_chg = m.get("rz_chg_pct")
        short_chg = m.get("rq_net")
        rows[c] = {
            "main_net_pct": f.get("main_net_pct"), "main_net_in": f.get("main_net_in"),
            "lhb_net": l.get("lhb_net"), "lhb_inst": l.get("lhb_inst"),
            "rz_chg": rz_chg, "short_chg": short_chg,
            "rz_net": m.get("rz_net"), "rz_net_5d": m.get("rz_net_5d"),
            "rz_net_10d": m.get("rz_net_10d"), "rz_balance_pct": m.get("rz_balance_pct"),
            "float_mktcap": f.get("float_mktcap"), "industry": f.get("industry"),
            "pe_ttm": f.get("pe_ttm"), "pb": f.get("pb"), "close": f.get("close"),
        }
    doc = {
        "as_of": day,
        "lhb_date": lhb.get("date"), "margin_date": mg.get("date"),
        "lag_days": LAG_DAYS,
        "rows": rows,
        "sources": flow["sources"] + lhb["sources"] + mg["sources"],
        "margin_raw": mg["data"],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prev_file": os.path.basename(prev) if prev else None,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cn_flow_{day}.json")
    json.dump(doc, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    n_flow = sum(1 for v in rows.values() if v["main_net_pct"] is not None)
    n_lhb = sum(1 for v in rows.values() if v["lhb_net"] is not None)
    n_rz = sum(1 for v in rows.values() if v["rz_chg"] is not None)
    print(f"[cn_events] {path}\n  universe={len(universe)} codes={len(rows)} 资金流={n_flow} 龙虎榜={n_lhb} 两融变化={n_rz}"
          f"  龙虎榜日期={lhb.get('date')}  两融日期={mg.get('date')}")
    return path


def _previous_file(out_dir: str):
    files = sorted(glob.glob(os.path.join(out_dir, "cn_flow_*.json")))
    return files[-1] if files else None




# ----------------------------------------------------------- 信号 vs 未来收益

def spearman(a, b):
    """秩相关。要回答的是'排名有没有关系'，不是'数值有没有线性关系'。"""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n < 10:
        return None, n

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    if not da or not db:
        return None, n
    return num / (da * db), n


def _load_close(code: str):
    """读 quant 的列式 store（store/<code>.bin + .idx）。第 2 列是 close，见 build_store.py。"""
    import struct
    b = os.path.join(HERE, "store", code + ".bin")
    i = os.path.join(HERE, "store", code + ".idx")
    if not (os.path.exists(b) and os.path.exists(i)):
        return None
    raw = open(i, "rb").read()
    n = struct.unpack("<I", raw[:4])[0]
    dates = [raw[4 + k * 10: 14 + k * 10].decode("ascii") for k in range(n)]
    body = open(b, "rb").read()[8:]
    closes = [struct.unpack_from("<7d", body, k * 56)[1] for k in range(n)]
    return dates, closes


def check(fwd_days: int = 5, day: str | None = None) -> dict:
    """今天的截面信号 vs 未来 N 日收益的 rank IC。

    这是第一步检验：**今天的资金流/两融/龙虎榜，和之后几天的涨跌有没有关系**。
    没关系的话，把它做成因子的活就不用干了 —— 先花十分钟回答这个问题。
    """
    files = sorted(glob.glob(os.path.join(HERE, "events", "cn_flow_*.json")))
    if not files:
        raise SystemExit("no cn_flow_*.json; run `python cn_events.py` first")
    doc = json.load(open(files[-1], encoding="utf-8"))
    as_of = doc["as_of"]
    res = {"as_of": as_of, "horizon_days": fwd_days, "store": os.path.isdir(os.path.join(HERE, "store")),
           "note": "", "ic": {}}
    if not res["store"]:
        res["note"] = "没有 store/ 目录（先跑 build_store.py）"
        return res
    # store 的最后交易日是整个市场共有的：先看一眼，比信号日旧就没得检验
    probe = next((f[:-4] for f in os.listdir(os.path.join(HERE, "store")) if f.endswith(".bin")), None)
    sig_day = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
    if probe:
        got0 = _load_close(probe)
        res["store_last_date"] = got0[0][-1] if got0 else None
    if not res.get("store_last_date") or res["store_last_date"] <= sig_day:
        res["note"] = (f"store 最后一个交易日 {res.get('store_last_date')} 不晚于信号日 {sig_day}："
                       "还没有'未来'可检验。等下一个交易日的日线落库后再跑 --check。")
        return res
    fields = ("main_net_pct", "lhb_net", "rz_chg", "rz_net_10d", "rz_balance_pct")
    xs = {f: [] for f in fields}
    rets = []
    n_px = 0
    for c, v in (doc.get("rows") or {}).items():
        got = _load_close(c)
        if not got:
            continue
        dates, closes = got
        last_date = dates[-1]
        if len(closes) <= fwd_days:
            continue
        # 必须严格晚于信号日：store 的最后一个交易日若等于信号日，取到的是"过去"的收益。
        sig_day = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
        if dates[-1] <= sig_day:
            last_date = dates[-1]
            continue
        d0, d1 = closes[-1 - fwd_days], closes[-1]
        if not d0:
            continue
        n_px += 1
        rets.append(d1 / d0 - 1.0)
        for f in fields:
            xs[f].append(v.get(f))
    res["stocks_with_price"] = n_px
    res["store_last_date"] = last_date if n_px else None
    res["same_day_warning"] = (
        None if not n_px else
        ("信号来自 " + as_of + "，但 store 最后一个交易日是 " + str(res["store_last_date"]) +
         "；两者相同的话，这个 IC 里的'未来收益'其实是过去 " + str(fwd_days) + " 天的收益，属于同日前视，不算检验。"))

    for f in fields:
        ic, n = spearman(xs[f], rets)
        res["ic"][f] = {"rank_ic": round(ic, 4) if ic is not None else None, "n": n}
    return res



def replay(day: str, fwd_days: int = 5, ftype: str = "RPTA_WEB_RZRQ", page_size: int = 500,
           max_pages: int = 40) -> dict:
    """用一个**过去的交易日 D** 的历史两融截面，去对 D+1..D+fwd 的真实收益。

    意义：今天的数据还不构成检验（没有未来），但两融的公开历史可以按日回放。
    这是这些信号第一次真正被放在时间外测。
    """
    import struct
    codes = {}
    flt = f"(DATE='{day}')"
    rows_seen = 0
    for pn in range(1, max_pages + 1):
        try:
            page, url = _cnmarket()._em_dc("RPTA_WEB_RZRQ_GGMX", pageSize=page_size,
                                           pageNumber=pn, filter=flt)
        except Exception as exc:                 # noqa: BLE001
            return {"error": f"page {pn}: {type(exc).__name__}"}
        if not page:
            break
        rows_seen += len(page)
        for r in page:
            c = r.get("SCODE")
            if c:
                codes[c] = r
        if len(page) < page_size:
            break
    out = {"day": day, "rows": rows_seen, "codes": len(codes), "ic": {}}
    if not codes:
        out["note"] = "那一天没有两融数据（非交易日或接口不给历史）"
        return out
    xs = {f: [] for f in ("rz_chg", "rz_net", "rz_net_5d", "rz_net_10d", "rz_balance_pct")}
    rets = []
    n_px = 0
    for c, r in codes.items():
        got = _load_close(c)
        if not got:
            continue
        dates, closes = got
        if day not in dates:
            continue
        k = dates.index(day)
        if k + fwd_days >= len(dates):
            continue
        d0, d1 = closes[k], closes[k + fwd_days]
        if not d0:
            continue
        n_px += 1
        rets.append(d1 / d0 - 1.0)
        xs["rz_chg"].append(r.get("FIN_BALANCE_GR"))
        xs["rz_net"].append(r.get("RZJME"))
        xs["rz_net_5d"].append(r.get("RZJME5D"))
        xs["rz_net_10d"].append(r.get("RZJME10D"))
        xs["rz_balance_pct"].append(r.get("RZYEZB"))
    out["stocks_with_price"] = n_px
    for f, v in xs.items():
        ic, n = spearman(v, rets)
        out["ic"][f] = {"rank_ic": round(ic, 4) if ic is not None else None, "n": n}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "events"))
    ap.add_argument("--build-panel", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fwd", type=int, default=5)
    ap.add_argument("--replay", default=None, help="用某个过去交易日回放两融截面（YYYY-MM-DD）")
    a = ap.parse_args(argv)
    if a.replay:
        print(json.dumps(replay(a.replay, a.fwd), ensure_ascii=False, indent=1))
        return None
    if a.check:
        print(json.dumps(check(a.fwd), ensure_ascii=False, indent=1))
        return None
    if a.build_panel:
        panel = {}
        for p in sorted(glob.glob(os.path.join(a.out, "cn_flow_*.json"))):
            d = json.load(open(p, encoding="utf-8"))
            day = d["as_of"]
            for c, v in (d.get("rows") or {}).items():
                panel.setdefault(c, {})[day] = v
        dst = os.path.join(a.out, "cn_panel.json")
        json.dump(panel, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[cn_events] panel -> {dst}  codes={len(panel)}  days="
              f"{len({d for v in panel.values() for d in v})}")
        return dst
    return build(a.out)


if __name__ == "__main__":
    main()
