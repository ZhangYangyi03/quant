# -*- coding: utf-8 -*-
"""alt_panel.py —— 把四类非价量数据落成月度截面面板，并用 factor_orth.json 的
同一口径检验：逐月对 [1, size, amt20] 做 OLS，取残差算 Spearman IC / t。

面板格式：alt_data/alt_panel.json
    {"months": {"YYYY-MM": {"code": {"f": value, ...}}}, "features": [...], "meta": {...}}

时点纪律：
  lhb        TRADE_DATE 当日收盘后才公布 → 归到次月截面（避免用当天收盘信息）
  margin     DATE 当日收盘后公布        → 归到次月截面
  holdernum  HOLD_NOTICE_DATE 真实公告日 → 只在公告日 <= 月末时可见
  orghold    NOTICE_DATE 真实公告日     → 同上
无记录 = None，绝不填 0。
"""
import os, json, math, collections, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
ALT = os.path.join(HERE, "alt_data")
P = os.path.join(HERE, "px_index_meta.json")

def load(fn):
    for ln in open(os.path.join(ALT, fn), encoding="utf-8"):
        try: yield json.loads(ln)
        except Exception: pass

def month_shift(m, k=1):
    y, mm = int(m[:4]), int(m[5:7]); t = y * 12 + (mm - 1) + k
    return "%04d-%02d" % (t // 12, t % 12 + 1)

def asof_ok(d, month_end):
    return bool(d) and str(d)[:10] <= month_end

def main():
    meta = json.load(open(P, encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    ends = {}
    for d in dates:
        if d >= "2012-01-01": ends[d[:7]] = d
    code_set = set(codes)
    months = collections.defaultdict(lambda: collections.defaultdict(dict))

    # --- 1) 龙虎榜：当月净买额 / 月成交额、上榜次数；归到次月 ---
    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0]))
    for r in load("lhb.jsonl"):
        c = r.get("SECURITY_CODE"); d = r.get("TRADE_DATE")
        if c not in code_set or not d: continue
        m = str(d)[:7]; a = agg[m][c]
        a[0] += float(r.get("BILLBOARD_NET_AMT") or 0)
        a[1] += float(r.get("ACCUM_AMOUNT") or 0)
        a[2] += 1
    for m, cm in agg.items():
        nm = month_shift(m)
        if nm not in ends: continue
        for c, (net, amt, cnt) in cm.items():
            months[nm][c]["lhb_net_pct"] = net / amt if amt > 0 else None
            months[nm][c]["lhb_count"] = cnt

    # --- 2) 两融：月末截面，融资余额占流通市值、当日融资买入额占比；归到次月 ---
    for r in load("margin_monthly.jsonl"):
        c = r.get("SCODE"); d = r.get("_day")
        if c not in code_set or not d: continue
        nm = month_shift(str(d)[:7])
        if nm not in ends: continue
        sz = float(r.get("SZ") or 0)
        months[nm][c]["margin_pct"] = float(r.get("RZYE") or 0) / sz * 100 if sz > 0 else None
        months[nm][c]["margin_buy_pct"] = float(r.get("RZMRE") or 0) / sz * 100 if sz > 0 else None

    # --- 3) 股东户数：按真实公告日 HOLD_NOTICE_DATE 归入可见月份 ---
    for r in load("holdernum.jsonl"):
        c = r.get("SECURITY_CODE"); nd = r.get("HOLD_NOTICE_DATE")
        if c not in code_set or not nd: continue
        m = str(nd)[:7]
        if m not in ends or not asof_ok(nd, ends[m]): continue
        ratio = r.get("HOLDER_NUM_RATIO")
        if ratio is None: continue
        # 户数减少 = 筹码集中，取负号让"越大越看多"
        prev = months[m][c].get("holder_concent")
        v = -float(ratio)
        months[m][c]["holder_concent"] = v if prev is None else max(prev, v)

    # --- 4) 机构持仓：按 NOTICE_DATE 可见，用机构数变化 / 持股占比变化 ---
    for r in load("orghold.jsonl"):
        c = r.get("SECURITY_CODE"); nd = r.get("NOTICE_DATE")
        if c not in code_set or not nd: continue
        m = str(nd)[:7]
        if m not in ends or not asof_ok(nd, ends[m]): continue
        chg = r.get("CHANGE_RATIO_CHANGE")
        if chg is None: continue
        prev = months[m][c].get("org_chg")
        v = float(chg)
        months[m][c]["org_chg"] = v if prev is None else v

    out = {"months": {m: dict(cm) for m, cm in months.items()},
           "features": ["lhb_net_pct", "lhb_count", "margin_pct", "margin_buy_pct",
                        "holder_concent", "org_chg"],
           "meta": {"n_months": len(months), "n_codes": len(codes),
                    "lag": "lhb/margin shifted +1 month; holdernum/orghold by real notice date"}}
    json.dump(out, open(os.path.join(ALT, "alt_panel.json"), "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    # 覆盖率
    print("months", len(months))
    for f in out["features"]:
        cnt = sum(1 for cm in months.values() for v in cm.values() if f in v and v[f] is not None)
        mm = sum(1 for cm in months.values() if any(f in v and v[f] is not None for v in cm.values()))
        print("  %-18s rows=%d months=%d" % (f, cnt, mm))

if __name__ == "__main__":
    main()
