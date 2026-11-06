# -*- coding: utf-8 -*-
"""audit_backtest.py —— 回测可信度三层审计。

主流开源框架（qlib / backtrader / vnpy / Lean）都校验数据、校验成本，
但**没有一家**校验下面三件事，而这三者恰好是回测假阳性的最大来源：

  1. 信号索引错配：pred 行数 < calendar x universe，按行号截断会把分数
     配到别的股票/日期，框架不报错，只给一条看起来很合理的曲线。
  2. 前视对齐 + 幸存者偏差：信号用了当日收盘价却去赚当日收益；
     池子只含活到末期的股票。
  3. 不可买股票：一字涨停 / 停牌股按收盘价成交，而它们往往是收益最高的一批。

用法：
    from audit_backtest import audit
    report = audit(signal=SIG, close=C, open=O, amount=A,
                   cal=TD, codes=pool, topk=50, cost_bp=3.65)
    print(report["verdict"])

输入都是 (T, N) numpy 数组，T 交易日、N 股票；signal 中 NaN 表示无信号。
返回 dict，含三层各自的量化指标与总判定，可直接 json 落盘。
"""
import numpy as np

def _ret(P):
    R = np.full((P.shape[0]-1, P.shape[1]), np.nan)
    ok = np.isfinite(P[:-1]) & np.isfinite(P[1:]) & (P[:-1] > 0)
    R[ok] = P[1:][ok] / P[:-1][ok] - 1
    return R

def _nav(sig, R, topk, cost_bp, lag=1, block=None):
    T, N = sig.shape
    s = np.where(np.isfinite(sig), sig, -1e9)
    eq = [1.0]; prev = None; blocked = 0
    # 时间轴：SIG[t] = t 日收盘后算出的信号；R[k] = P[k+1]/P[k]-1。
    # 可执行：SIG[t] 配 R[t+1]（t 收盘下单，t+1 成交）。
    # 因此 lag=L 时用 s[k-L] 配 R[k]；lag=1 即 SIG[k-1] 配 R[k]，等价 qlib shift=1。
    # lag=0 是 SIG[k] 配 R[k]（同日收盘信号赚同日收益）= 前视。
    for k in range(lag, T - 1):
        sc = s[k - lag].copy()
        if block is not None:
            g = block[k + 1]          # 次日开盘跳空，决定 k 日收盘下单能否成交
            bad = ~np.isfinite(g) | (g > 0.095)
            sc[bad] = -1e9
        top = np.argsort(sc)[-topk:]
        if block is not None:
            blocked += int((~np.isfinite(block[k + 1, top]) |
                            (block[k + 1, top] > 0.095)).sum())
        r = np.nanmean(R[k, top])
        r = 0.0 if not np.isfinite(r) else r
        tv = 1.0 if prev is None else len(set(top) ^ set(prev)) / topk
        r -= cost_bp / 1e4 * tv
        prev = top
        eq.append(eq[-1] * (1 + r))
    return float(eq[-1]), blocked

def audit(signal, close, open_, amount, cal, codes, topk=50, cost_bp=3.65,
          limit_gap=0.095, signal_index=None):
    """signal_index: 可选，(datetime, instrument) 的 MultiIndex（或等长二元组列表）。
    传入后逐格核对信号是否落在期望的 (cal x codes) 网格上；不传则只能靠统计特征间接推断。
    这是抓「按行号截断导致整体错位」的唯一可靠办法。
    """
    S = np.asarray(signal, dtype=np.float64)
    C = np.asarray(close, dtype=np.float64)
    O = np.asarray(open_, dtype=np.float64)
    A = np.asarray(amount, dtype=np.float64)
    T, N = S.shape
    assert C.shape == O.shape == A.shape == (T, N), "shape mismatch"
    out = {"shape": [T, N], "cal": [str(cal[0]), str(cal[-1])],
           "codes": len(codes), "topk": topk, "cost_bp": cost_bp}

    # --- 层 1：信号索引完整性 ---
    finite = np.isfinite(S)
    sig_cells = int(finite.sum())
    expected = T * N
    out["L1_signal_index"] = {
        "finite_cells": sig_cells, "expected_cells": expected,
        "missing": expected - sig_cells,
        "missing_pct": round(100 * (expected - sig_cells) / expected, 3),
        "note": "行数不足却按行号对齐会静默错配；必须用原始 (datetime,instrument) 索引重建",
        "pass": bool(sig_cells >= expected * 0.5),
    }
    # 每日有效信号数是否稳定（突降说明索引错位/数据断点）
    per_day = finite.sum(axis=1)
    out["L1_signal_index"]["per_day_min"] = int(per_day.min())
    out["L1_signal_index"]["per_day_max"] = int(per_day.max())
    out["L1_signal_index"]["per_day_cv"] = round(float(per_day.std()/max(per_day.mean(),1)), 4)
    # 显式索引核对：能抓到"行数够但整体错位一格"这类统计特征看不出的错配
    if signal_index is not None:
        try:
            pairs = list(signal_index)
            dt = [str(x[0])[:10] for x in pairs]
            cd = [str(x[1]) for x in pairs]
            exp_dt = {str(d)[:10] for d in cal}
            exp_cd = {str(c) for c in codes}
            bad_dt = sum(1 for x in dt if x not in exp_dt)
            bad_cd = sum(1 for x in cd if x not in exp_cd)
            # 关键：每个 (dt, code) 必须唯一，且总对数与 T*N 一致
            uniq = len(set(zip(dt, cd)))
            mismatch = (bad_dt or bad_cd or uniq != len(pairs))
            out["L1_signal_index"].update({
                "pairs": len(pairs), "unique_pairs": uniq,
                "bad_datetime": bad_dt, "bad_code": bad_cd,
                "expected_grid": expected,
                "index_checked": True, "index_mismatch": bool(mismatch),
            })
            if mismatch:
                out["L1_signal_index"]["pass"] = False
                out["L1_signal_index"]["note"] += "；显式索引核对发现错配"
        except Exception as e:
            out["L1_signal_index"]["index_checked"] = False
            out["L1_signal_index"]["index_error"] = str(e)[:120]
    else:
        out["L1_signal_index"]["index_checked"] = False
        out["L1_signal_index"]["index_note"] = (
            "未传 signal_index：整体滚动一格这类错配无法可靠识别，"
            "强烈建议传入 model.predict() 的真实 MultiIndex")
        # 无显式索引时，用 lag 形状做间接提示（不据此判 fail，避免误报）
        out["L1_signal_index"]["indirect_hint"] = (
            "lag 扫描非单调可能提示错配，见 L2_alignment.nav_by_lag")

    RCC, ROO = _ret(C), _ret(O)
    GAP = np.full((T, N), np.nan)
    ok = np.isfinite(C[:-1]) & np.isfinite(O[1:]) & (C[:-1] > 0)
    GAP[1:][ok] = O[1:][ok] / C[:-1][ok] - 1
    # 停牌：成交额为 0/NaN，也视为不可买
    halted = ~np.isfinite(A) | (A <= 0)
    GAP[halted] = np.nan

    # --- 层 2：前视对齐 + 池子偏差 ---
    lag_nav = {}
    for lag in (0, 1, 2, 5):
        n, _ = _nav(S, RCC, topk, cost_bp, lag=lag)
        lag_nav[lag] = round(n, 4)
    lag0, lag1 = lag_nav[0], lag_nav[1]
    lookahead = (lag0 - lag1) / max(abs(lag1 - 1.0), 1e-9)
    out["L2_alignment"] = {
        "nav_by_lag": lag_nav, "lag0_minus_lag1": round(lag0 - lag1, 4),
        "lookahead_ratio": round(float(lookahead), 3),
        "note": "lag=0 用当日收盘信号赚当日收益即前视；qlib shift=1 对应 lag=1",
        "pass": bool(lag0 - lag1 < 0.15),
    }
    # 池子本身收益 vs 可交易状态
    alive = np.isfinite(C[0]) & np.isfinite(C[-1])
    # 统计指纹：逐 lag 算信号与未来收益的横截面 Spearman 相关均值。
    # 真 alpha 应在 lag>=1 仍有正/负 IC 且衰减平缓；
    # 只有 lag=0 强、lag>=1 塌到 0 = 信号里含当日价格信息（前视）。
    def _rankic(sig_row, ret_row):
        m = np.isfinite(sig_row) & np.isfinite(ret_row)
        if m.sum() < 20:
            return np.nan
        a, b = sig_row[m], ret_row[m]
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra*ra).sum() * (rb*rb).sum())
        return float((ra*rb).sum()/den) if den > 0 else np.nan
    ic_by_lag = {}
    for lag in (0, 1, 2, 3, 5):
        vals = []
        for k in range(lag, T-1):
            v = _rankic(S[k-lag], RCC[k])
            if np.isfinite(v): vals.append(v)
        ic_by_lag[lag] = round(float(np.mean(vals)), 5) if vals else None
    out["L2_alignment"]["rankic_by_lag"] = ic_by_lag
    ic0 = ic_by_lag.get(0); ic1 = ic_by_lag.get(1)
    ic_collapse = (ic0 is not None and ic1 is not None and abs(ic0) > 0.05
                   and abs(ic1) < abs(ic0) * 0.4)
    out["L2_alignment"]["ic_collapse"] = bool(ic_collapse)
    if ic_collapse:
        out["L2_alignment"]["pass"] = False
        out["L2_alignment"]["note"] += "；RankIC 在 lag=0 强、lag=1 塌，统计指纹指向前视"
    out["L2_alignment"]["survivor_fraction"] = round(float(alive.mean()), 4)
    out["L2_alignment"]["survivor_note"] = "池子只含首末日都有价的股票 = 潜在幸存者偏差；应核对 instruments 真实起止日"

    # --- 层 3：不可买股票 ---
    n_c2c, _ = _nav(S, RCC, topk, cost_bp, lag=1)
    n_c2c_f, bl_c = _nav(S, RCC, topk, cost_bp, lag=1, block=GAP)
    n_o2o, _ = _nav(S, ROO, topk, cost_bp, lag=1)
    n_o2o_f, bl_o = _nav(S, ROO, topk, cost_bp, lag=1, block=GAP)
    # 被排除股票在 top50 里的日均收益 vs 留下的
    s = np.where(finite, S, -1e9)
    ex_r, kept_r, overlap = [], [], []
    for k in range(1, T - 1):
        a = np.argsort(s[k - 1])[-topk:]
        b = np.where(~np.isfinite(GAP[k + 1]) | (GAP[k + 1] > limit_gap),
                     -1e9, s[k - 1])
        bb = np.argsort(b)[-topk:]
        overlap.append(len(set(a) & set(bb)))
        bad = ~np.isfinite(GAP[k + 1, a]) | (GAP[k + 1, a] > limit_gap)
        x = RCC[k, a][bad]; x = x[np.isfinite(x)]
        y = RCC[k, a][~bad]; y = y[np.isfinite(y)]
        if len(x): ex_r.append(float(x.mean()))
        if len(y): kept_r.append(float(y.mean()))
    ex = float(np.mean(ex_r)) if ex_r else 0.0
    kp = float(np.mean(kept_r)) if kept_r else 0.0
    out["L3_untradable"] = {
        "c2c_nav": round(n_c2c, 4), "c2c_filtered_nav": round(n_c2c_f, 4),
        "o2o_nav": round(n_o2o, 4), "o2o_filtered_nav": round(n_o2o_f, 4),
        "filtered_drop": round(n_c2c - n_c2c_f, 4),
        "top50_overlap_mean": round(float(np.mean(overlap)), 2),
        "excluded_mean_daily_ret": round(ex, 6),
        "kept_mean_daily_ret": round(kp, 6),
        "excluded_slots_blocked": int(bl_c),
        "note": "排除次日高开>9.5%或无开盘价/零成交的股票；若只换极少量持仓却大幅掉净值，说明超额全来自买不进去的涨停股",
        # 只判"过滤是否戳破虚假收益"，不判策略本身赚不赚钱：
        # 随机数据净值低于 1 是正常的，那属于 alpha 问题，不是可信度问题。
        "pass": bool(n_c2c - n_c2c_f < 0.10 and n_o2o - n_o2o_f < 0.10),
    }

    fails = [k for k in ("L1_signal_index", "L2_alignment", "L3_untradable") if not out[k]["pass"]]
    out["fails"] = fails
    out["verdict"] = ("回测不可信：%s 未通过" % ", ".join(fails)) if fails else "三层审计通过"
    out["executable_alpha_nav"] = round(n_o2o_f, 4)
    out["scope_warning"] = ("单年结果不构成可信度结论：需逐年 audit 后看跨年度稳定性，"
                            "一年通过不等于策略可用")
    out["remedies"] = _remedies(out)
    return out


def _remedies(out):
    """把每层失败映射成具体修法 + 量化影响。给人看的，不是给机器判的。"""
    r = []
    l1, l2, l3 = out["L1_signal_index"], out["L2_alignment"], out["L3_untradable"]
    if not l1["pass"] or not l1.get("index_checked"):
        r.append({
            "layer": "L1", "problem": "信号索引未核对或发现错配",
            "fix": ("不要 np.asarray(pred) 后按行号拼面板。用 model.predict() 的 "
                    "(datetime,instrument) MultiIndex 做 unstack().reindex(index=日历, columns=股票池)，"
                    "并把该索引传给 audit(signal_index=...)。"),
            "cost_hint": "本项目实测：修正后同一模型净值 1.1298 -> 4.2774（错配把好信号配丢了，也可能反过来虚高）",
        })
    if not l2["pass"]:
        nav = l2["nav_by_lag"]
        r.append({
            "layer": "L2", "problem": "前视 / 时间对齐错误",
            "fix": ("把回测改成 SIG[t] 配 R[t+1]（qlib 的 shift=1），"
                    "并检查特征是否用了当日收盘价；池子用当年真实已上市股票、保留后来退市的。"),
            "cost_hint": "lag0 %.4f -> lag1 %.4f，虚高 %.4f；RankIC %s" %
                         (nav.get(0, 0), nav.get(1, 0), l2["lag0_minus_lag1"],
                          l2.get("rankic_by_lag")),
        })
    if not l3["pass"]:
        r.append({
            "layer": "L3", "problem": "涨停/停牌股按收盘价虚假成交",
            "fix": ("次日开盘/前收-1 > 9.5%、或开盘价缺失、或成交额为 0 的股票，"
                    "在生成订单前从候选中剔除；用次日 open 成交而非当日 close。"),
            "cost_hint": ("C2C %.4f -> 过滤后 %.4f（掉 %.4f）；"
                          "被排除股票日均收益 %.4f vs 留下 %.4f，"
                          "top50 只换了 %.2f 只") %
                         (l3["c2c_nav"], l3["c2c_filtered_nav"], l3["filtered_drop"],
                          l3["excluded_mean_daily_ret"], l3["kept_mean_daily_ret"],
                          50 - l3["top50_overlap_mean"]),
        })
    if not r:
        r.append({"layer": "-", "problem": "三层都过",
                  "fix": "仍需跑 multi_year_verdict() 看跨年度稳定性，单年通过不算数",
                  "cost_hint": ""})
    return r


def multi_year_verdict(reports):
    """跨年度稳定性判定：单年三层通过不算数。

    reports: [(year, audit_result), ...]
    返回 dict，含逐年可执行净值、正收益年数、去掉最好一年后的合计，
    以及最终判定。经验阈值：至少 4/5 年正、去掉最好一年后仍 > 1.0。
    """
    ys = [y for y, _ in reports]
    navs = [r["executable_alpha_nav"] for _, r in reports]
    navs_arr = np.asarray(navs, dtype=np.float64)
    pos = int((navs_arr > 1.0).sum())
    prod = float(navs_arr.prod())
    best = int(np.argmax(navs_arr))
    ex_best = float(np.delete(navs_arr, best).prod())
    res = {
        "years": ys, "executable_nav": [round(x, 4) for x in navs],
        "positive_years": pos, "total_years": len(navs),
        "compound_nav": round(prod, 4),
        "best_year": ys[best], "best_year_nav": round(float(navs_arr[best]), 4),
        "compound_excluding_best": round(ex_best, 4),
        "all_three_layers_pass": all(not r["fails"] for _, r in reports),
        "note": ("单年三层通过不代表可信：多数年份不通过、或收益全靠最好一年，"
                 "都说明没有稳定可交易 alpha"),
    }
    res["pass"] = bool(res["all_three_layers_pass"] and pos >= len(navs) - 1
                       and ex_best > 1.0 and prod > 1.0)
    res["verdict"] = ("有稳定可执行 alpha" if res["pass"]
                      else "无稳定可执行 alpha：%d/%d 年为正，去掉 %s 后合计 %.4f"
                           % (pos, len(navs), ys[best], ex_best))
    return res

if __name__ == "__main__":
    import json, sys
    print("audit_backtest 是一个库；从量化项目里喂 (signal, close, open, amount, cal, codes) 调用 audit()。")
