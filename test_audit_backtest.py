# -*- coding: utf-8 -*-
"""test_audit_backtest.py —— 用合成数据验证三层审计能否正确判出已知 bug。

四种情形：
  A 干净：无前视、无涨停、信号完整 -> 三层通过、可执行 alpha > 1
  B 索引错配：把信号按行号滚动一格 -> L1 per_day_cv / 与价格错配
  C 前视：信号 = 当日收益（完美偷看）-> L2 lag0 远高于 lag1
  D 涨停不可买：把高收益股票设成次日一字涨停 -> L3 filtered 大幅掉净值
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_backtest import audit, multi_year_verdict

T, N, TOPK, COST = 240, 120, 20, 3.65
rng = np.random.default_rng(1234)

def base():
    # 随机游走价格，open = close 带小噪声，amount 恒正
    C = 10.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    O = C * (1 + rng.normal(0, 0.003, (T, N)))
    A = np.full((T, N), 1e6)
    cal = ["d%04d" % i for i in range(T)]
    codes = ["s%03d" % i for i in range(N)]
    return C, O, A, cal, codes

def signal_from(C, horizon=1):
    # 干净信号：用过去 5 日反转预测未来，绝不用当日收益
    S = np.full((T, N), np.nan)
    for t in range(6, T):
        S[t] = -(C[t-1] / C[t-6] - 1) + rng.normal(0, 0.005, N)
    return S

def test_A_clean():
    C, O, A, cal, codes = base()
    S = signal_from(C)
    r = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST)
    assert r["L1_signal_index"]["pass"], r["L1_signal_index"]
    assert r["L2_alignment"]["pass"], r["L2_alignment"]
    assert r["L3_untradable"]["pass"], r["L3_untradable"]
    assert r["verdict"].startswith("三层审计通过"), r["verdict"]
    print("[A] clean ->", r["verdict"], "exec", r["executable_alpha_nav"])

def test_B_index_shift():
    C, O, A, cal, codes = base()
    S = signal_from(C)
    # 模拟按行号截断导致的整体错位：信号整体向下滚一天
    S2 = np.vstack([S[:1], S[:-1]])
    r = audit(S2, C, O, A, cal, codes, topk=TOPK, cost_bp=COST)
    # 错位本身 L1 的 per_day 统计未必能抓，但 L2 的 lag 结构会变
    print("[B] shifted ->", r["verdict"], "fails", r["fails"],
          "lag_nav", r["L2_alignment"]["nav_by_lag"])
    assert r["L1_signal_index"]["finite_cells"] == int(np.isfinite(S2).sum())
    # 降级模式必须明确告知：不传 signal_index 就无法可靠抓整体错位
    assert r["L1_signal_index"]["index_checked"] is False
    assert "无法可靠识别" in r["L1_signal_index"]["index_note"]
    print("    degraded-mode warning present: 整体错位需显式 signal_index 才能判")

def test_C_lookahead():
    C, O, A, cal, codes = base()
    # 前视信号：直接用当日收益当分数（完美偷看）
    S = np.full((T, N), np.nan)
    R = np.diff(C, axis=0) / C[:-1]      # R[t] = C[t+1]/C[t]-1
    S[:T-1] = R                          # S[t] 直接偷看 t 日之后的收益
    r = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST)
    assert not r["L2_alignment"]["pass"], r["L2_alignment"]
    assert r["L2_alignment"]["nav_by_lag"][0] > r["L2_alignment"]["nav_by_lag"][1]
    print("[C] lookahead ->", r["verdict"], "lag0/lag1",
          r["L2_alignment"]["nav_by_lag"][0], r["L2_alignment"]["nav_by_lag"][1])

def test_D_untradable():
    C, O, A, cal, codes = base()
    S = signal_from(C)
    # 把模型最爱的 20 只在次日做成一字涨停（open 高开 10%）、当日零成交
    for t in range(6, T-2):
        top = np.argsort(np.nan_to_num(S[t], nan=-1e9))[-20:]
        O[t+1, top] = C[t, top] * 1.10     # 次日一字高开，买不进
        A[t+1, top] = 0                   # 零成交
        # 不改 close：让回测仍按原价成交，从而暴露"虚假成交"
    r = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST)
    assert not r["L3_untradable"]["pass"], r["L3_untradable"]
    assert r["L3_untradable"]["filtered_drop"] > 0.10, r["L3_untradable"]
    assert r["L3_untradable"]["excluded_mean_daily_ret"] > r["L3_untradable"]["kept_mean_daily_ret"]
    print("[D] untradable ->", r["verdict"], "drop",
          r["L3_untradable"]["filtered_drop"], "excluded_ret",
          r["L3_untradable"]["excluded_mean_daily_ret"])

def test_F_index_mismatch():
    C, O, A, cal, codes = base()
    S = signal_from(C)
    # 构造一个行数够但整体错一格、且含越界代码的假索引
    pairs = []
    for i in range(T):
        for j in range(N):
            pairs.append((cal[(i + 1) % T], codes[j]))
    pairs[3] = ("1999-01-01", "BADCODE")   # 越界
    r = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST,
              signal_index=pairs)
    assert r["L1_signal_index"]["index_checked"]
    assert r["L1_signal_index"]["index_mismatch"], r["L1_signal_index"]
    assert not r["L1_signal_index"]["pass"]
    assert "L1_signal_index" in r["fails"]
    print("[F] index mismatch ->", r["verdict"],
          "bad_dt", r["L1_signal_index"]["bad_datetime"],
          "bad_code", r["L1_signal_index"]["bad_code"])

def test_G_index_ok():
    C, O, A, cal, codes = base()
    S = signal_from(C)
    pairs = [(cal[i], codes[j]) for i in range(T) for j in range(N)]
    r = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST,
              signal_index=pairs)
    assert r["L1_signal_index"]["index_checked"]
    assert not r["L1_signal_index"]["index_mismatch"]
    assert r["L1_signal_index"]["pass"]
    print("[G] index ok ->", r["verdict"])

def test_H_ic_fingerprint():
    C, O, A, cal, codes = base()
    S = np.full((T, N), np.nan)
    R = np.diff(C, axis=0) / C[:-1]
    S[:T-1] = R                       # 完美前视
    r = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST)
    ic = r["L2_alignment"]["rankic_by_lag"]
    assert ic[0] is not None and abs(ic[0]) > 0.5, ic
    assert r["L2_alignment"]["ic_collapse"] or abs(ic[1]) < abs(ic[0])*0.4, ic
    assert not r["L2_alignment"]["pass"]
    print("[H] IC fingerprint ->", ic, "collapse",
          r["L2_alignment"]["ic_collapse"])

def test_I_remedies():
    C, O, A, cal, codes = base()
    S = np.full((T, N), np.nan)
    R = np.diff(C, axis=0) / C[:-1]
    S[:T-1] = R
    rep = audit(S, C, O, A, cal, codes, topk=TOPK, cost_bp=COST)
    rem = rep["remedies"]
    assert rem and any(x["layer"] == "L2" for x in rem), rem
    l2 = [x for x in rem if x["layer"] == "L2"][0]
    assert "shift=1" in l2["fix"] and "lag0" in l2["cost_hint"]
    print("[I] remedies ->", [x["layer"] for x in rem],
          "L2 hint", l2["cost_hint"][:110])

def test_E_multi_year():
    reps = [(2022, {"executable_alpha_nav": 0.94, "fails": ["L3_untradable"]}),
            (2023, {"executable_alpha_nav": 1.20, "fails": []}),
            (2024, {"executable_alpha_nav": 0.79, "fails": ["L3_untradable"]}),
            (2025, {"executable_alpha_nav": 1.62, "fails": []}),
            (2026, {"executable_alpha_nav": 0.96, "fails": []})]
    v = multi_year_verdict(reps)
    assert not v["pass"]
    assert v["positive_years"] == 2
    print("[E] multi-year ->", v["verdict"])

if __name__ == "__main__":
    for fn in (test_A_clean, test_B_index_shift, test_C_lookahead,
               test_D_untradable, test_E_multi_year, test_F_index_mismatch,
               test_G_index_ok, test_H_ic_fingerprint, test_I_remedies):
        fn()
    print("ALL TESTS PASSED")
