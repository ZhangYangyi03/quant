#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""机构级研究栈端到端演示与验收。

    python run_institutional.py
    python run_institutional.py --codes 40 --days 520 --out results/demo.json

**重要**：本脚本运行在**确定性合成数据**上，唯一目的是验证「代码正确性与
验收流程有效」。输出中的任何收益率、夏普、DSR 都**不构成**策略有效的证据，
也不代表原仓库历史结果。真实结论必须接入 PIT 行情/财报后重跑。

验收闸门（任一失败则退出码非 0）：
  G1 无未来函数（截断重跑 + 蜜罐反例）
  G2 参数选择不依赖未来（平行世界检验）
  G3 真 walk-forward（只用历史选参、样本外结算）
  G4 完全可复现（同配置同数据逐位一致）
  G5 基准不是固定篮子（修复历史 eq_nofilter 缺陷）
  G6 运行留痕完整（配置指纹 / 数据指纹 / 种子）
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantlab.backtest.engine import run_backtest, run_benchmark, default_signal
from quantlab.config import Config, BacktestConfig
from quantlab.data import generate_synthetic_market
from quantlab.reporting import SCHEMA, result_record, write_run
from quantlab.validate.no_lookahead import verify_no_lookahead
from quantlab.validate.stats import (newey_west_t, deflated_sharpe, pbo,
                                     bootstrap_max_sharpe_p, annualized_sharpe,
                                     performance)
from quantlab.validate.walkforward import run_walkforward, detect_lookahead

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = ("ep", "bp", "size", "lowvol", "random")


def _line(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _sharpe_of_daily(xs):
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return mu / sd if sd > 0 else 0.0


def best_by_history(hist, future=None):
    """诚实的选参器：只用历史夏普。future 是蜜罐参数，故意忽略。"""
    return max(hist, key=lambda k: _sharpe_of_daily(hist[k]))


def _variance(xs):
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="quantlab 机构级验收演示")
    ap.add_argument("--codes", type=int, default=60, help="合成股票数")
    ap.add_argument("--days", type=int, default=760, help="合成交易日数")
    ap.add_argument("--seed", type=int, default=20261105, help="数据生成种子")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "institutional_run.json"))
    args = ap.parse_args()
    if args.codes < 10 or args.days < 120:
        ap.error("--codes 至少 10、--days 至少 120（否则无未来函数验收样本不足）")

    _line("0. 数据与配置（合成数据，仅验证代码）")
    market = generate_synthetic_market(n_codes=args.codes, n_days=args.days,
                                       seed=args.seed)
    n_sessions = len(market.calendar)
    # 参数自适应：股票池变小时同步缩小持仓数，否则 G5「基准非固定篮子」
    # 会因为池子里根本没有超过 top_k 只票而误判失败。
    top_k = min(20, max(2, args.codes // 3))
    # 演示区间精确覆盖全部合成交易日，不留未来也不留空档
    cfg = Config(backtest=BacktestConfig(
        start=market.calendar.dates[0], end=market.calendar.dates[-1],
        seed=args.seed, signal="ep", top_k=top_k, rebalance="monthly"))
    n_bars = len(market.panel)
    n_delisted = sum(1 for c in market.universe.all_codes()
                     if market.universe.get(c).delisted_on)
    print(f"合成标的 {args.codes} 只 / 交易日 {n_sessions} 天 / K 线 {n_bars} 根")
    print(f"其中含退市 {n_delisted} 只、停牌/一字板/财报修订/ST 窗口（见 tests）")
    print(f"config_fingerprint = {cfg.fingerprint()}")
    print(f"data_fingerprint   = {market.fingerprint()}")

    _line("1. 策略 vs 同池随机 K 等权基准")
    strategy = run_backtest(market, cfg, label="ep_top20")
    benchmark = run_benchmark(market, cfg)
    repeat = run_backtest(market, cfg, label="ep_top20")
    sm, bm = strategy.metrics(cfg), benchmark.metrics(cfg)
    print(f"{'指标':<22}{'EP 策略':>16}{'随机K基准':>16}")
    for key, label in (("cagr_pct", "CAGR %"), ("ann_vol_pct", "年化波动 %"),
                       ("sharpe", "Sharpe"), ("sortino", "Sortino"),
                       ("max_drawdown_pct", "最大回撤 %"),
                       ("total_multiple", "净值倍数")):
        print(f"{label:<22}{sm[key]:>16}{bm[key]:>16}")
    excess = [a - b for a, b in zip(strategy.daily_returns(),
                                    benchmark.daily_returns())]
    print(f"{'超额 CAGR (pp)':<22}{sm['cagr_pct'] - bm['cagr_pct']:>16.4f}")

    _line("2. 统计检验（合成数据上只作演示，不作显著性结论）")
    v = cfg.validation
    sr_daily = _sharpe_of_daily(strategy.daily_returns())
    grid = {}
    for name in GRID:
        r = run_backtest(market, cfg, signal_fn=default_signal(name, cfg.backtest.seed),
                         label=f"grid_{name}")
        grid[name] = r.daily_returns()
    sharpes = [_sharpe_of_daily(x) for x in grid.values()]
    stats = {
        "newey_west_t_excess": round(newey_west_t(excess, v.nw_lags), 4),
        "annualized_sharpe_strategy": round(annualized_sharpe(strategy.daily_returns(), v), 4),
        "deflated_sharpe": round(deflated_sharpe(
            sr_daily, n_trials=len(GRID), n_obs=len(strategy.daily_returns()),
            skew=sm["skew"], kurtosis=sm["excess_kurtosis"] + 3.0,
            variance_of_sr=_variance(sharpes)), 4),
        "pbo": round(pbo(grid, slices=v.pbo_slices), 4),
        "bootstrap_reality_check_p": round(bootstrap_max_sharpe_p(
            grid, draws=v.bootstrap_draws, seed=7), 4),
        "n_trials": len(GRID),
    }
    for k, val in stats.items():
        print(f"  {k:<34} {val}")
    print("  注：DSR/PBO/RC 均需真实数据与真实试验次数才有解释力。")

    _line("3. 无未来函数验收（G1）")
    acceptance = verify_no_lookahead(market, cfg)
    print(f"  截断点                 {acceptance['cut']}")
    print(f"  删除的未来 K 线        {acceptance['data_removed']['full_bars_after_cut']}")
    print(f"  删除的未来财报        {acceptance['data_removed']['full_reports_after_cut']}")
    print(f"  截面 PIT 合规          {acceptance['cross_section_pit']}")
    print(f"  截断重跑最大净值差     {acceptance['strategy_prefix']['max_abs_equity_diff']}")
    print(f"  蜜罐作弊被抓到         {acceptance['honeypot_caught']} "
          f"(作弊净值差 {acceptance['honeypot_max_abs_equity_diff']})")
    print(f"  => {'PASS' if acceptance['passed'] else 'FAIL'}")

    _line("4. 真 walk-forward 与选参泄漏检测（G2 / G3）")
    # 短区间时缩小历史窗口，保证至少留出 3 个样本外期；默认 760 天仍是 120/20
    n_rets = n_sessions - 1
    wf_hist = min(120, max(30, n_rets // 4))
    wf_oos = max(10, min(20, n_rets // 10))
    wf = run_walkforward(grid, best_by_history, min_history=wf_hist, oos_len=wf_oos)
    leak = detect_lookahead(best_by_history, grid,
                            min_history=wf_hist, oos_len=wf_oos)
    wf_summary = {
        "min_history": wf_hist,
        "oos_len": wf_oos,
        "top_k": top_k,
        "periods": len(wf),
        "oos_total_return": round(sum(r.oos_return for r in wf), 6),
        "positive_periods": sum(1 for r in wf if r.oos_return > 0),
        "mean_is_sharpe": round(sum(r.is_sharpe for r in wf) / len(wf), 4) if wf else None,
        "first_history_len": wf[0].history_len if wf else None,
        "last_history_len": wf[-1].history_len if wf else None,
    }
    print(f"  参数 min_history={wf_hist} / oos_len={wf_oos} / top_k={top_k}")
    print(f"  样本外期数 {wf_summary['periods']} / 正收益期 {wf_summary['positive_periods']}")
    print(f"  历史窗口 {wf_summary['first_history_len']} -> {wf_summary['last_history_len']}")
    print(f"  IS 平均夏普 {wf_summary['mean_is_sharpe']} / "
          f"OOS 累计收益 {wf_summary['oos_total_return']}")
    print(f"  平行世界 changed_ratio = {leak['changed_ratio']} "
          f"(tests={leak['tests']}) => {'PASS' if not leak['leak_detected'] else 'FAIL'}")

    _line("5. 验收闸门")
    traded = {f.order.code for f in benchmark.fills if f.status != "REJECTED"}
    gates = {
        "G1 无未来函数": acceptance["passed"],
        "G2 选参不依赖未来": not leak["leak_detected"],
        "G3 真 walk-forward": (len(wf) >= 3 and
                               all(wf[i].history_len < wf[i + 1].history_len
                                   for i in range(len(wf) - 1))),
        "G4 完全可复现": (strategy.equity_curve == repeat.equity_curve and
                          strategy.config_fingerprint == repeat.config_fingerprint),
        "G5 基准非固定篮子": (len(traded) > cfg.backtest.top_k and
                             benchmark.data_fingerprint == strategy.data_fingerprint),
        "G6 运行留痕完整": bool(strategy.config_fingerprint and
                               strategy.data_fingerprint and cfg.backtest.seed),
    }
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_pass = all(gates.values())

    _line("6. 落盘（可追溯到唯一一次运行）")
    payload = {
        "disclaimer": "合成数据，仅验证代码正确性与验收流程；不含任何真实收益结论。",
        "data": {"n_codes": args.codes, "n_days": args.days, "seed": args.seed,
                 "n_sessions": n_sessions, "n_bars": n_bars,
                 "n_delisted": n_delisted,
                 "data_fingerprint": market.fingerprint()},
        "config_fingerprint": cfg.fingerprint(),
        "config": cfg.as_dict(),
        "strategy": result_record(strategy, cfg),
        "benchmark": result_record(benchmark, cfg),
        "statistics": stats,
        "no_lookahead_acceptance": acceptance,
        "walkforward": wf_summary,
        "walkforward_leak_check": leak,
        "gates": gates,
        "all_gates_passed": all_pass,
        "schema": SCHEMA,
    }
    path = write_run(args.out, payload)
    print(f"  已写入 {path}")
    print(f"  文件大小 {os.path.getsize(path):,} 字节；含完整净值曲线与全部指纹")

    _line("结论")
    print("  全部闸门通过。" if all_pass else "  存在未通过闸门，请检查上方 FAIL 项。")
    print("  原仓库历史数字（24.14% / 31.46x / 超额 17.45pp / 逐年 11.19%）")
    print("  仍为作废状态：缺少 .bin 行情矩阵，需接入真实 PIT 数据后重算。")
    print("  详见 INSTITUTIONAL.md。")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
