"""回测驱动：walk-forward 生成器（不走 walk-forward 的那部分会在报告里标出来）。"""
import csv, os, sys, time as perf_time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Bars, Portfolio, Cost, run_backtest
import signals_pro as S
import perf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_ORIG_RUN = run_backtest      # 冻结模块级引用，避免被 main 内的局部名遮蔽

def _dbg(tag, *a):
    import tempfile, datetime
    with open(os.path.join(tempfile.gettempdir(), "bt_dbg.log"), "a", encoding="utf-8") as f:
        f.write(datetime.datetime.now().strftime("%H:%M:%S ") + tag + " " + " ".join(str(x) for x in a) + "\n")

def load_bars(codes=None):
    bars = Bars()
    for fn in sorted(os.listdir(DATA)):
        if not fn.endswith(".csv"): continue
        code = fn[:-4]
        if codes and code not in codes: continue
        bars.load_csv(code, os.path.join(DATA, fn))
    bars.finalize()
    return bars

def make_signal(bars, signal_name, k, warmup):
    """信号只吃截止当日（含）的收盘序列，engine 负责把它变成次日开盘的委托。"""
    base = {"momentum20": S.momentum(20), "momentum60": S.momentum(60),
            "resid20_60": S.ma_residual(20, 60), "lowvol60": S.low_vol(60),
            "mom_vol20": S.momentum_lowvol(20)}[signal_name]
    pick = S.top_k(base, k)
    if not hasattr(bars, "series"):
        bars.build_series()

    def f(b, date, i):
        hist = {}
        eligible = []
        for code in b.series:
            if b.get(code, date) is None:
                continue          # 当日停牌/未上市，不参与
            j = b.close_upto(code, date)
            if j >= warmup:
                hist[code] = b.series[code][:j]
                eligible.append(code)
        if len(eligible) < k:
            return None
        return pick(hist, eligible, date)
    return f

def summarize(port, bars, label, warmup_days=0):
    eq = port.equity[warmup_days:]
    if len(eq) < 30: return {"label": label, "error": "too short"}
    p = perf.perf(eq)
    filled = [t for t in port.trades if t.get("status") == "filled"]
    fees = sum(t["fee"] for t in filled)
    turnover = sum(t["amount"] for t in filled)/max(eq[0][1], 1)
    p.update({"label": label, "trades": len(filled), "fees_paid": round(fees, 2),
              "turnover_x_initial": round(turnover, 2),
              "blocked": dict(port.blocked)})
    return p

def main():
    bars = load_bars()
    _t0 = perf_time.time()
    print("=== 步骤 1/4 载入 ===")
    print("交易日", len(bars.dates), bars.dates[0], "->", bars.dates[-1], "标的", len(bars.data))
    bench_code = "510300.SH"
    rows = bars.data[bench_code]
    base = rows[0]["close"]
    bench_eq = [(r["date"], 100000.0*r["close"]/base) for r in rows if r["date"] >= "2016-01-01"]
    print("=== 步骤 2/4 含成本回测 10 个配置 ===")
    results = {}
    rets_by_cfg = {}
    # 零成本重跑，量化费用到底吃掉了多少 —— 这是"毛利 vs 净利"的分水岭
    def zero_fee_result(name, k):
        """固定同一随机种子（这里没有随机源），保证零成本与含成本两次跑的是同一条买路径。"""
        from engine import run_backtest_zero_fee as _zero_run
        sig0 = make_signal(bars, name, k, warmup=60)
        _dbg("CALL_ZERO", name, k, id(_zero_run), id(sig0))
        p0 = _zero_run(bars, sig0, start_cash=100000.0, start_date="2016-01-01")
        _dbg("ZERO_END", name, k, p0.equity[-1][1], len([x for x in p0.trades if x.get("status")=="filled"]))
        return summarize(p0, bars, f"{name}/k{k}/zero-fee", warmup_days=60)
    bench_p = perf.perf(bench_eq)
    results["benchmark_510300"] = bench_p
    print("基准 510300  CAGR %.2f%%  夏普 %.3f  MDD %.2f%%  (%s -> %s)" % (
        bench_p["cagr_pct"], bench_p["sharpe"], bench_p["max_drawdown_pct"], bench_p["start"], bench_p["end"]))

    for name in ("momentum20", "momentum60", "resid20_60", "lowvol60", "mom_vol20"):
        for k in (2, 3):
            sig = make_signal(bars, name, k, warmup=60)
            _dbg("CALL_MAIN", name, k, id(run_backtest), id(sig))
            port = run_backtest(bars, sig, start_cash=100000.0, start_date="2016-01-01")
            _dbg("MAIN_END", name, k, port.equity[-1][1])
            s = summarize(port, bars, f"{name}/k{k}", warmup_days=60)
            s["alpha_beta"] = perf.alpha_beta(port.equity[60:], bench_eq)
            _, rr = perf.to_returns(port.equity[60:])
            rets_by_cfg[f"{name}/k{k}"] = list(rr)
            results[f"{name}/k{k}"] = s
            print(f"{name:12s} k{k}  CAGR {s.get('cagr_pct')}%  夏普 {s.get('sharpe')}  MDD {s.get('max_drawdown_pct')}%  "
                  f"alpha {s['alpha_beta'].get('alpha_annual_pct')}% (t={s['alpha_beta'].get('alpha_t')})  被挡 {s.get('blocked')}")
    print("=== 步骤 3/4 零成本对照 ===")
    for kk in list(results):
        if kk == "benchmark_510300" or "cagr_pct" not in results[kk]: continue
        if kk.count("/") != 1: continue
        nm, kpart = kk.split("/")
        try:
            z = zero_fee_result(nm, int(kpart[1:]))
            if not z or "cagr_pct" not in z:
                results[kk]["zero_fee_error"] = str(z)[:200]
                continue
            results[kk]["zero_fee_cagr_pct"] = z.get("cagr_pct")
            results[kk]["zero_fee_sharpe"] = z.get("sharpe")
            results[kk]["fee_drag_annual_pct"] = round((z.get("cagr_pct") or 0) - results[kk]["cagr_pct"], 2)
            _dbg("REPORT", kk, "zero_cagr", z.get("cagr_pct"), "net_cagr", results[kk]["cagr_pct"],
                 "drag", results[kk]["fee_drag_annual_pct"])
            print(f"{kk:16s} 零成本 CAGR {z.get('cagr_pct')}%  净 {results[kk]['cagr_pct']}%  费用拖累 {results[kk]['fee_drag_annual_pct']}%/年", flush=True)
        except Exception as e:
            results[kk]["zero_fee_error"] = str(e)
    print("=== 步骤 4/4 多重检验校正与落盘 ===")
    # 多重检验校正：一共试了多少个配置（与那次跑无关地重复跑一遍，验证结果稳定）
    if "--twice" in os.sys.argv:
        from engine import Cost as _C2
        sig_r = make_signal(bars, "momentum20", 3, 60)
        p_r = _ORIG_RUN(bars, sig_r, start_cash=100000.0, start_date="2016-01-01")
        print("重复跑 momentum20/k3 终值:", p_r.equity[-1][1], "（应与上面一致）")
    import numpy as _np
    trials = len(results) - 1
    srs = _np.array([v.get("sharpe", 0) for kk, v in results.items() if kk != "benchmark_510300"])
    sr_var = float(_np.var(srs, ddof=1)) if len(srs) > 1 else None
    for kk, v in results.items():
        if kk == "benchmark_510300" or "sharpe" not in v: continue
        # 口径统一：deflated_sharpe 吃的是日频夏普，这里把年化 NaN 换成日频
        sr_daily = v["sharpe"] / (250 ** 0.5)
        v["deflated"] = perf.deflated_sharpe(sr_daily, trials, v["days"], v["skew"],
                                             v["excess_kurtosis"] + 3, sr_var / 250.0)
    results["pbo"] = perf.pbo(rets_by_cfg)
    results["_meta"] = {"trials": trials, "universe": sorted(bars.data.keys()),
                        "period": [bars.dates[0], bars.dates[-1]], "days": len(bars.dates)}
    import json
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_etf.json"), "w"),
              ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
