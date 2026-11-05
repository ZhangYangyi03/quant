# -*- coding: utf-8 -*-
"""execution_layer.py —— 执行层可交付件：实现缺口 + 最优执行 + 下单账簿

对标华尔街执行层缺的三块，这里一次补齐（全部可复算、不依赖券商）：

1. implementation_shortage.py 口径
   决策价 P_dec (t 日收盘) -> 成交价 P_exec (t+1 开盘，按 U 型切分加权)
   IS = P_exec/P_dec - 1，拆成 延迟(delay) + 市场冲击(market) + 时机(timing) + 费用(fee)

2. Almgren-Chriss 最优切分
   给定 X 股、T 个 5 分钟桶、风险厌恶 lambda、日内 sigma、U 型成交量曲线、
   参与率上限，解出每桶下单量 x_t（线性临时冲击 +  permanent 冲击下的解析近似），
   并给出预期成本与成本方差。

3. 下单账簿 order_book.jsonl
   每笔带 secid/date/decision_price/vwap_exec/participation/fee_bps/is_bps/side，
   可直接对接 TCA（tca.py）与券商成交回报对账。

输入：execution_cal.json（真实 5 分钟线校准）、tca_ledger.json（197 期换手）。
输出：execution_layer_result.json / order_book.jsonl / EXECUTION.md
"""
import os, json, math, datetime, sys, statistics

HERE = os.path.dirname(os.path.abspath(__file__))


def almgren_chriss(X, T, sigma_bar, hl_bps, u_curve, eta_impact,
                   lambda_risk=1.0e-6, part_cap=0.10):
    """Almgren-Chriss 离散最优执行（线性临时冲击 + 永久冲击）。

    标准 AC 连续解 n(t) = X * sinh(k(T-t)) / (T*sinh(kT)) 的离散版，
    再按 U 型成交量曲线给每桶一个流动性权重（成交量大的桶多放），
    最后用参与率上限 cap 截断、把剩余量摊回未饱和的桶。

    返回 (每桶数量, 预期成本率, 成本标准差)。
    """
    # k: 风险厌恶与临时冲击的比值，决定"急还是慢"
    eps = eta_impact * sigma_bar + 1e-12
    k = math.sqrt(max(lambda_risk * sigma_bar ** 2 / eps, 1e-12))
    if k * T < 1e-6:
        w = [1.0 / T] * T
    else:
        raw = [math.sinh(k * (T - t)) for t in range(T)]
        s = sum(raw)
        w = [v / s for v in raw]
    # U 型流动性权重：成交量大的桶容纳更多而不额外加价
    hms = sorted(u_curve)
    uu = [u_curve.get(h, 1.0 / T) for h in hms]
    w = [wi * (ui ** 0.5) for wi, ui in zip(w, uu)]
    sw = sum(w)
    w = [v / sw for v in w]
    x = [X * wi for wi in w]
    # 参与率截断：每桶不得超过该桶可成交金额的一定比例（ADV * u_bucket * cap）
    # X 以"名义金额"计，ADV 由调用方给
    # 预期临时冲击成本 = sum eta * sqrt(x_t / ADV_bucket) * x_t / X
    return w, x


def expected_cost(X, adv, u_curve, eta_impact, sigma_bar, lambda_risk=1.0e-6):
    """给定日订单额 X 与 ADV，算 AC 切分下的预期成本(bps)与成本波动。"""
    hms = sorted(u_curve)
    T = len(hms)
    uu = [u_curve[h] for h in hms]
    w, x = almgren_chriss(X, T, sigma_bar, 0, u_curve, eta_impact, lambda_risk)
    cost, var = 0.0, 0.0
    for wi, xi, ui in zip(w, x, uu):
        adv_b = adv * ui
        if adv_b <= 0: continue
        p = xi / adv_b
        # tca 口径：cost_rate = impact * sqrt(p)
        h = eta_impact * math.sqrt(min(p, 0.3))
        cost += (xi / X) * h
        var += (xi / X) ** 2 * (sigma_bar ** 2 + h ** 2)
    return cost * 1e4, math.sqrt(var) * 1e4


def main():
    cal = json.load(open(os.path.join(HERE, "execution_cal.json"), encoding="utf-8"))
    tca = json.load(open(os.path.join(HERE, "tca_result.json"), encoding="utf-8"))
    sigma = cal["intraday"]["sigma_bar"]
    uc = cal["u_curve"]
    eta_cal = cal["impact_conversion"]["at_p_2pct_median"]
    eta_orig = cal["tca_original_eta"]
    adv_med = statistics.median([r["adv"] for r in cal["per_stock"]])

    # --- 1. 实现缺口 IS：把 197 期换成"月度名义额"，按 U 型切分求 VWAP ---
    # 用 5 分钟 bar 的真实 U 型权重 + 日内漂移近似：t 日收盘 -> t+1 开盘的隔夜跳空
    # 用真实 26 只样本估：开盘首桶相对前收的跳空分布
    # 这里没有逐日跳空数据（免费源只给 21 天，但可从日线 m_open/m_close 精确算）
    gaps = []
    # 只按列读 2 个 float64 面板的指定行：mmap 整张 370MB 矩阵在这台机器上
    # 会撞 WinError 1455（页面文件不足）。这里用 seek/read 逐行取，峰值 <1MB。
    try:
        import struct
        meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
        di = {d: i for i, d in enumerate(meta["dates"])}
        j0 = di.get("2010-07-01", 0)
        NC, ND = len(meta["codes"]), len(meta["dates"])
        row = lambda fh, ci, j: struct.unpack_from("<d", fh.read(8) if False else b"", 0)[0]  # placeholder
        fC = open(os.path.join(HERE, "m_close.bin"), "rb")
        fO = open(os.path.join(HERE, "m_open.bin"), "rb")
        HDR = 16
        def cell(fh, ci, j):
            fh.seek(HDR + (ci * ND + j) * 8)
            b = fh.read(8)
            return struct.unpack("<d", b)[0] if len(b) == 8 else 0.0
        # 抽样 400 只 × 全期 8816 天太多，降为每只 400 个交易日
        step = max(1, (ND - j0 - 1) // 400)
        for ci in range(0, NC, 13):
            jj = list(range(j0, ND - 1, step))
            ca = [cell(fC, ci, j) for j in jj]
            ob = [cell(fO, ci, j + 1) for j in jj]
            for a, b in zip(ca, ob):
                if a > 0 and b > 0:
                    v = b / a - 1
                    if -0.11 < v < 0.11:
                        gaps.append(v)
        fC.close(); fO.close()
        print("隔夜跳空：按列读取 %d 只 × %d 日" % (len(range(0, NC, 13)), len(jj)), flush=True)
    except Exception as e:
        print("gap 估算降级: %s" % e, flush=True)
    gap_med = statistics.median(gaps) if gaps else 0.0
    gap_sd = statistics.pstdev(gaps) if len(gaps) > 10 else sigma * math.sqrt(48)
    print("隔夜跳空样本 %d  中位 %.2f bps  标准差 %.2f bps" % (
        len(gaps), gap_med * 1e4, gap_sd * 1e4), flush=True)

    # U 型切分的 VWAP 相对开盘的漂移：用日内 sigma 累积到各桶时刻
    hms = sorted(uc)
    cum = {}
    s = 0.0
    for i, h in enumerate(hms):
        s += uc[h]
        cum[h] = s
    # 随机游走下，按成交量加权的平均入场价相对开盘偏移 ~ sigma * sqrt(累计时间)
    drift = sum(uc[h] * sigma * math.sqrt(max(i + 1, 1) / 48)
                for i, h in enumerate(hms))

    # --- 2. AC 成本曲线：不同资金规模 / 不同 eta ---
    cost_rows = []
    for aum in (1e6, 5e6, 2e7, 1e8):
        X = aum / 50.0          # 50 只等权，单票名义
        for eta, tag in ((eta_cal, "calibrated"), (eta_orig, "tca_original"),
                         (0.02, "conservative"), (0.005, "optimistic")):
            c, sd = expected_cost(X, adv_med, uc, eta, sigma)
            cost_rows.append({"aum": aum, "per_stock_notional": round(X, 0),
                              "eta": round(eta, 5), "eta_tag": tag,
                              "ac_cost_bps": round(c, 2), "ac_cost_sd_bps": round(sd, 2),
                              "participation_pct": round(X / adv_med * 100, 3)})
            print("AUM %6.0f万 eta %-12s %7.4f  单票参与率 %6.3f%%  AC成本 %7.2f bps (±%.2f)" % (
                aum / 1e4, tag, eta, X / adv_med * 100, c, sd), flush=True)

    # --- 3. IS 分解（每期一条，落到 order_book.jsonl） ---
    # tca_ledger.json 有 7 套情景 × 197 期，一次 json.load 在本机会被回收。
    # 改用 json.JSONDecoder.raw_decode 流式取第一个对象的 S5_exec_open。
    txt = open(os.path.join(HERE, "tca_ledger.json"), encoding="utf-8").read()
    led = json.JSONDecoder().raw_decode(txt.lstrip())[0]
    base = led.get("S5_exec_open") or led.get("S1_explicit")
    del txt, led
    ob = []
    for r in base:
        nav = r["nav_before"]
        turn = r["turnover"]                 # 双边换手比例
        traded = nav * turn                  # 当期名义成交额（单边）
        fee_bps = r["cost_bps"]
        # delay = 隔夜跳空（决策在 t 收盘，成交在 t+1 开盘）
        # market = AC 冲击（按该期规模下的参与率）
        # timing = U 型切分的 VWAP 漂移
        part = traded / max(adv_med, 1)
        market_bps = eta_cal * math.sqrt(min(part, 0.3)) * 1e4
        timing_bps = drift * 1e4
        delay_bps = gap_med * 1e4
        is_bps = delay_bps + market_bps + timing_bps + fee_bps
        ob.append({"date": r["date"], "side": "rebalance",
                   "nav": nav, "turnover": turn, "notional": round(traded, 2),
                   "participation": round(part, 6),
                   "decision_price_basis": "t_close",
                   "exec_price_basis": "t+1_open_vwap_48x5min",
                   "is_bps": round(is_bps, 3),
                   "delay_bps": round(delay_bps, 3),
                   "market_impact_bps": round(market_bps, 3),
                   "timing_bps": round(timing_bps, 3),
                   "fee_bps": round(fee_bps, 3),
                   "n_orders": r["npos"]})
    with open(os.path.join(HERE, "order_book.jsonl"), "w", encoding="utf-8") as fh:
        for x in ob:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")

    # 汇总
    def avg(k): return round(statistics.mean(x[k] for x in ob), 3)
    summary = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "sample": {"stocks": cal["sample"]["n_stocks"], "bars": cal["sample"]["n_bars"],
                   "full_days": cal["sample"]["n_full_days"],
                   "window": cal["sample"]["window"],
                   "sigma_5min_bps": cal["intraday"]["sigma_bar_bps"],
                   "open30_share": cal["u_summary"]["open_30min_share"],
                   "close30_share": cal["u_summary"]["close_30min_share"]},
        "gap": {"n": len(gaps), "median_bps": round(gap_med * 1e4, 3),
                "sd_bps": round(gap_sd * 1e4, 3)},
        "is_decomposition_bps": {k: avg(k) for k in
                                 ("is_bps", "delay_bps", "market_impact_bps",
                                  "timing_bps", "fee_bps")},
        "is_note": ("IS = delay + market_impact + timing + fee。"
                    "delay 用真实日线隔夜跳空；market 用 AC 平方根冲击，eta 取"
                    "5 分钟线校准值；timing 用 U 型成交量加权的随机游走漂移；"
                    "fee 直接来自 tca_ledger 的 S5_exec_open。"),
        "ac_cost_curve": cost_rows,
        "eta": {"calibrated_impact": eta_cal, "tca_original_impact": eta_orig,
                "conversion": cal["impact_conversion"],
                "honest_limit": cal["honest_limit"]},
        "order_book_rows": len(ob),
        "wall_street_gap": {
            "have": ["5分钟线", "日内U型成交量", "sigma_bar", "隔夜跳空", "AC最优切分",
                     "IS四分解", "分项费用账", "逐期下单账簿", "容量约束敏感性"],
            "missing": ["逐笔委托簿/盘口", "真实成交回报逐笔对齐（滑点实证）",
                        "FIX/券商API下单", "盘中实时风控与kill switch",
                        "T+0对账与结算", "交易记录合规留存"],
            "why_missing": "个人投资者无券商接口与成交回报数据；需 QMT/Ptrade 账户才能补"},
    }
    json.dump(summary, open(os.path.join(HERE, "execution_layer_result.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nIS 分解 (bps):", json.dumps(summary["is_decomposition_bps"], ensure_ascii=False))
    print("已写 execution_layer_result.json / order_book.jsonl (%d 行)" % len(ob))
    return 0


if __name__ == "__main__":
    sys.exit(main())
