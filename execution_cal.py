# -*- coding: utf-8 -*-
"""execution_cal.py —— 用真实 5 分钟线校准执行层参数（不用拍脑袋的 eta）

华尔街执行层的三件东西，之前 quant 全缺：
  1. 日内波动 sigma_bar   5 分钟对数收益的标准差（Almgren-Chriss 的 sigma）
  2. 成交量 U 型分布      日内 48 个 5 分钟桶各占全天成交的比例（VWAP/切分用）
  3. 参与率与冲击         用真实 ADV + 5 分钟价差代理，估 square-root 冲击的 eta 量级

数据源：东方财富 push2his klt=5（免费、无需 key、可回溯约 1000 根 = 21 个交易日）。
样本：26 只（大/中/小市值 + 各行业 + 沪深主板/创业板/科创板），每只 1000 根。
所有数字写进 execution_cal.json，带 source_url 与样本数，可复算。

诚实边界：免费源没有逐笔委托簿，因此这里不能估"真市场冲击"（那要成交回报）。
能估的是：用 5 分钟 bar 的振幅/成交量构造一个可复算的日内成本上界，
并把这个 eta 的合理区间代回 tca.py 做敏感性 —— 结论对 eta 的脆弱性才算被量出来。
"""
import os, json, urllib.request, urllib.error, time, math, datetime, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# (secid, 名称, 分组)。覆盖大盘/中盘/小盘与主板/创业/科创。
SAMPLE = [
    ("1.600000", "浦发银行", "bank_large"), ("1.601398", "工商银行", "bank_large"),
    ("1.600519", "贵州茅台", "consumer_large"), ("0.000858", "五粮液", "consumer_large"),
    ("1.601318", "中国平安", "insurance_large"), ("1.600036", "招商银行", "bank_large"),
    ("0.000333", "美的集团", "consumer_large"), ("0.000651", "格力电器", "consumer_large"),
    ("1.600276", "恒瑞医药", "pharma_large"), ("1.601899", "紫金矿业", "materials_large"),
    ("1.600030", "中信证券", "broker_large"), ("1.601012", "隆基绿能", "solar_large"),
    ("0.002594", "比亚迪", "auto_large"), ("0.300750", "宁德时代", "battery_large"),
    ("1.688981", "中芯国际", "semi_star"), ("1.688111", "金山办公", "software_star"),
    ("0.300059", "东方财富", "broker_gem"), ("0.300015", "爱尔眼科", "pharma_gem"),
    ("0.002415", "海康威视", "electronics_mid"), ("0.002230", "科大讯飞", "software_mid"),
    ("0.000725", "京东方A", "electronics_mid"), ("1.600703", "三安光电", "semi_mid"),
    ("0.002032", "苏泊尔", "consumer_small"), ("0.000538", "云南白药", "pharma_small"),
    ("1.603288", "海天味业", "consumer_mid"), ("0.300003", "乐普医疗", "pharma_gem"),
]


def fetch(secid, klt=5, lmt=1000):
    url = (API + "?secid=%s&klt=%d&fqt=1&lmt=%d&end=20500101&"
           "fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
           % (secid, klt, lmt))
    req = urllib.request.Request(url, headers={
        "Referer": "https://quote.eastmoney.com/", "User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=12) as r:
                j = json.loads(r.read().decode("utf-8", "replace"))
            d = j.get("data")
            if not d or not d.get("klines"):
                raise ValueError("empty klines")
            return d
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(0.6 * (attempt + 1))


def parse(klines):
    """fields2: 时间,开,收,高,低,成交量(手),成交额(元),振幅%"""
    out = []
    for s in klines:
        a = s.split(",")
        try:
            out.append({"t": a[0], "o": float(a[1]), "c": float(a[2]),
                        "h": float(a[3]), "l": float(a[4]),
                        "vol": float(a[5]), "amt": float(a[6]),
                        "amp": float(a[7])})
        except (ValueError, IndexError):
            continue
    return out


def main():
    t0 = time.time()
    allbars, per_stock, failed = [], [], []
    for secid, name, grp in SAMPLE:
        try:
            d = fetch(secid)
            bars = parse(d["klines"])
            if len(bars) < 200:
                failed.append((secid, name, "only %d bars" % len(bars)))
                continue
            for _b in bars:
                _b["secid"] = secid
            allbars.extend(bars)
            # 每只的日均成交额（按 bar 数/48 折成天数）
            days = len(bars) / 48.0
            adv = sum(b["amt"] for b in bars) / days
            amt_bar = [b["amt"] for b in bars]
            hi_lo = [(b["h"] - b["l"]) / b["c"] for b in bars if b["c"] > 0]
            per_stock.append({"secid": secid, "name": name, "group": grp,
                              "bars": len(bars), "days": round(days, 1),
                              "adv": round(adv, 0),
                              "median_bar_amt": round(statistics.median(amt_bar), 0),
                              "hl_median_bps": round(statistics.median(hi_lo) * 1e4, 2),
                              "first": bars[0]["t"], "last": bars[-1]["t"]})
            print("[%5.1fs] %-12s %4d bars  ADV %.2f亿  中位振幅 %.1fbps" % (
                time.time() - t0, name, len(bars), adv / 1e8,
                statistics.median(hi_lo) * 1e4), flush=True)
        except Exception as e:
            failed.append((secid, name, "%s: %s" % (type(e).__name__, str(e)[:80])))
            print("[%5.1fs] %-12s FAIL %s" % (time.time() - t0, name, str(e)[:80]), flush=True)
    if not allbars:
        print("全部取数失败，退出"); return 1

    # --- 1. 日内波动：同一天内的 5 分钟对数收益，跨日跳空不算 ---
    rets = []
    by_day = {}
    # 关键：必须按 (股票, 日期) 分组。按日期分组会把 26 只不同股票相邻 bar
    # 串成一条"收益"，算出来的 sigma 是 152 倍日波动，毫无意义。
    for b in allbars:
        by_day.setdefault((b["secid"], b["t"][:10]), []).append(b)
    for key, bs in by_day.items():
        bs.sort(key=lambda x: x["t"])
        for a, b in zip(bs, bs[1:]):
            if a["c"] > 0 and b["c"] > 0:
                rets.append(math.log(b["c"] / a["c"]))
    sigma_bar = statistics.pstdev(rets) if len(rets) > 10 else float("nan")
    sigma_day = sigma_bar * math.sqrt(48)

    # --- 2. 成交量 U 型：每个 (股票,日) 内部归一化，再对时刻取均值 ---
    buckets = {}
    full_days = {k: v for k, v in by_day.items() if len(v) >= 44}
    for key, bs in full_days.items():
        tot = sum(x["amt"] for x in bs)
        if tot <= 0: continue
        for x in bs:
            hm = x["t"][11:16]
            buckets.setdefault(hm, []).append(x["amt"] / tot)
    u_curve = {hm: round(statistics.mean(v), 6) for hm, v in sorted(buckets.items())}
    # 开盘 30 分钟 / 尾盘 30 分钟 占比
    open_sh = sum(v for hm, v in u_curve.items() if "09:30" <= hm < "10:00")
    close_sh = sum(v for hm, v in u_curve.items() if "14:30" <= hm < "15:00")

    # --- 3. 冲击系数 eta 的可复算区间 ---
    # 用"参与率 p = 单笔金额 / 当日 ADV"与"该 bar 的振幅"做横截面代理：
    # Almgren-Chriss 的临时冲击 ~ eta * sigma_bar * sqrt(p)；
    # 若把 bar 的半振幅 (h-l)/2c 视作可承受的单边成本上界，则
    #   eta_eff = (半振幅) / (sigma_bar * sqrt(p))
    # 对一组现实参与率 p 求 eta_eff 的分位数 —— 这就是"eta 该落在哪"的实测锚。
    hl_all = [(b["h"] - b["l"]) / (2 * b["c"]) for b in allbars if b["c"] > 0]
    hl_med = statistics.median(hl_all)
    hl_p25 = sorted(hl_all)[len(hl_all) // 4]
    hl_p75 = sorted(hl_all)[3 * len(hl_all) // 4]
    eta_anchor = {}
    for p in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        denom = sigma_bar * math.sqrt(p)
        eta_anchor[str(p)] = {"eta_p25": round(hl_p25 / denom, 3),
                              "eta_median": round(hl_med / denom, 3),
                              "eta_p75": round(hl_p75 / denom, 3)}
    # 再把锚代回 tca：tca.py 用的是 impact * sqrt(stake/adv)，系数 eta 直接可比
    # 关键结论：tca 原设 0.1 在什么分位
    p_ref = 0.02
    denom = sigma_bar * math.sqrt(p_ref)
    eta_at_2pct = {"p25": round(hl_p25 / denom, 3), "median": round(hl_med / denom, 3),
                   "p75": round(hl_p75 / denom, 3)}

    # --- 4. 把"半振幅上界"换算成 tca.py 的 impact 系数 ---
    # tca 的模型：cost_rate = impact * sqrt(stake/adv) = impact * sqrt(p)
    # 本文件的锚：hl_half = eta_eff * sigma_bar * sqrt(p)  =>  eta_eff = hl_half/(sigma_bar*sqrt(p))
    # 于是 impact = eta_eff * sigma_bar = hl_half / sqrt(p)。
    # 注意 impact 与 p 无关（因为 tca 的 sqrt(p) 已经把参与率吃掉了），
    # 取 p=2% 处校准：impact = hl_half / sqrt(0.02)。
    p_ref = 0.02
    conv = {"formula": "impact = hl_half / sqrt(p_ref)",
            "p_ref": p_ref,
            "at_p_2pct_p25": round(hl_p25 / math.sqrt(p_ref), 5),
            "at_p_2pct_median": round(hl_med / math.sqrt(p_ref), 5),
            "at_p_2pct_p75": round(hl_p75 / math.sqrt(p_ref), 5),
            "unit": "tca.py 的 impact 系数（无量纲）",
            "why_p_2pct": "50只等权、单票AUM 100万、ADV中位约20亿时参与率约0.01%；"
                          "2% 是策略规模放大到 2000万~1亿时的现实参与率，取它做压力锚"}
    out = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "impact_conversion": conv,
        "source_url": API, "klt": 5, "lmt": 1000,
        "sample": {"n_stocks": len(per_stock), "n_bars": len(allbars),
                   "n_days_per_stock": round(len(allbars) / max(len(per_stock), 1) / 48, 1),
                   "n_days_total": len(by_day), "n_full_days": len(full_days),
                   "failed": failed,
                   "window": [min(b["t"] for b in allbars), max(b["t"] for b in allbars)]},
        "intraday": {"n_5min_returns": len(rets),
                     "sigma_bar": round(sigma_bar, 8),
                     "sigma_bar_bps": round(sigma_bar * 1e4, 2),
                     "sigma_day_ann_pct": round(sigma_day * math.sqrt(252) * 100, 2),
                     "hl_half_median_bps": round(hl_med * 1e4, 2),
                     "hl_half_p25_bps": round(hl_p25 * 1e4, 2),
                     "hl_half_p75_bps": round(hl_p75 * 1e4, 2)},
        "u_curve": u_curve,
        "u_summary": {"open_30min_share": round(open_sh, 4),
                       "close_30min_share": round(close_sh, 4),
                       "n_buckets": len(u_curve)},
        "eta_anchor_by_participation": eta_anchor,
        "eta_at_2pct_participation": eta_at_2pct,
        "tca_original_eta": 0.1,
        "honest_limit": ("免费 5 分钟线没有委托簿/成交回报，半振幅只是可承受成本上界，"
                         "不是真实市场冲击。真实 TCA 需要券商成交回报逐笔对齐；"
                         "这里给出的是 eta 的量级锚与它对结论的敏感性区间。"),
        "per_stock": per_stock,
    }
    json.dump(out, open(os.path.join(HERE, "execution_cal.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n样本 %d 只 / %d 根 bar / %d 天" % (len(per_stock), len(allbars), len(by_day)))
    print("sigma_5min %.2f bps  -> 日 %.2f%%  年化 %.1f%%" % (
        sigma_bar * 1e4, sigma_day * 100, sigma_day * math.sqrt(252) * 100))
    print("开盘30min占比 %.1f%%  尾盘30min占比 %.1f%%" % (open_sh * 100, close_sh * 100))
    print("参与率2%% 处 eta 锚: p25 %.3f / median %.3f / p75 %.3f  (tca 原设 0.1)" % (
        eta_at_2pct["p25"], eta_at_2pct["median"], eta_at_2pct["p75"]))
    print("已写 execution_cal.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
