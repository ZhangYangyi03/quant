"""无未来函数验收：截断重跑 + 蜜罐反例 + 结构性不变量。

单靠 code review 无法证明「没偷看未来」。这里用三条互相独立的证据：

  1. **截面不变量**：每个 CrossSection 里的 Bar / history 的日期都 <= asof，
     且 PITFundamentals 的查询按公告日过滤。
  2. **截断重跑（核心）**：把市场在 cut 日物理截断（之后的 K 线和财报全部删掉），
     用同一配置、同一种子重跑，比较截止 cut 的净值曲线。若策略依赖了未来，
     两条曲线必然分叉。这是可复现、可自动化的因果检验。
  3. **蜜罐反例（检验检验本身）**：故意构造一个偷看未来财报的信号，
     要求验收器**必须**把它抓出来。若连明显的作弊都检测不到，
     前两条的「通过」就是假阳性，应判为验收失败。

注意：合成数据只用于验证代码正确性，本模块的一切结论都与收益数字无关。
"""
from __future__ import annotations

from hashlib import sha256

from ..clock import AsOf
from ..config import Config
from ..data.panel import Market


def _stable_unit(code: str) -> float:
    """代码 -> (0,1) 稳定值，用于让蜜罐信号在两个世界产生不同排序。"""
    h = sha256(code.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def make_cheating_signal(cut: str):
    """返回一个**故意**读取 cut 之后才公告的财报的信号函数。

    仅用于验收器的功效测试，绝不用于任何策略或收益结论。
    """
    future = AsOf("9999-12-31")

    def cheat(cs) -> dict[str, float]:
        out = {}
        for code in cs.members:
            late = [r for r in cs.fundamentals.visible(code, future)
                    if r.announced_at > cut]
            # 未来 EPS 之和乘以大系数，主导排序；截断世界里 late 恒为空，
            # 排序退化为 _stable_unit，两者选出的股票必然不同。
            leak = sum((r.eps or 0.0) for r in late) * 1000.0
            out[code] = leak + _stable_unit(code)
        return out

    return cheat


def _reports_after(market: Market, cut: str) -> int:
    future = AsOf("9999-12-31")
    return sum(1 for c in market.universe.all_codes()
               for r in market.fundamentals.visible(c, future)
               if r.announced_at > cut)


def _signal_signature(market: Market, signal_fn, dates: list[str]) -> str:
    """若干采样日上的打分快照哈希，用来低成本判定信号是否看到了未来。"""
    chunks = []
    for d in dates:
        cs = market.view(d)
        scores = signal_fn(cs)
        chunks.append(d + "|" + ",".join(
            f"{c}:{scores[c]:.8f}" for c in sorted(scores) if scores[c] is not None))
    return sha256("\n".join(chunks).encode("utf-8")).hexdigest()[:16]


def _cross_section_pit(market: Market, dates: list[str]) -> bool:
    """结构性检查：任何视图都不得含有 asof 之后的 Bar。"""
    for d in dates:
        cs = market.view(d)
        if any(b.date > d for b in cs.bars.values()):
            return False
        if any(any(b.date > d for b in bars) for bars in cs.history.values()):
            return False
        for code in cs.members:
            if any(r.announced_at > d
                   for r in cs.fundamentals.visible(code, cs.at)):
                return False
    return True


def _compare_prefix(full_curve, part_curve, cut: str):
    full = {d: v for d, v in full_curve if d <= cut}
    part = dict(part_curve)
    common = sorted(set(full) & set(part))
    aligned = (len(common) == len(full) == len(part) and len(common) > 0)
    diffs = [abs(full[d] - part[d]) for d in common]
    rel = [abs(full[d] - part[d]) / max(abs(full[d]), 1.0) for d in common]
    return {
        "aligned": aligned,
        "compared_days": len(common),
        "max_abs_equity_diff": round(max(diffs), 8) if diffs else None,
        "max_rel_equity_diff": round(max(rel), 10) if rel else None,
    }


def verify_no_lookahead(market: Market, cfg: Config, signal_fn=None,
                        cut: str | None = None, tol: float = 1e-6,
                        run_fn=None, deep_power_check: bool = True) -> dict:
    """对一次回测做完整的无未来函数验收，返回可落盘的结构化报告。

    run_fn 可注入自定义回测入口，签名需与 backtest.engine.run_backtest 一致。
    deep_power_check 会额外用蜜罐信号跑两遍完整回测（更慢但更严格）；
    关闭时退化为「打分快照」比较，仍足以证明数据确实被截断。
    """
    if run_fn is None:
        from ..backtest.engine import run_backtest
        run_fn = run_backtest
    if signal_fn is None:
        from ..backtest.engine import default_signal
        signal_fn = default_signal(cfg.backtest.signal, cfg.backtest.seed)

    dates = list(market.calendar.between(cfg.backtest.start, cfg.backtest.end))
    if len(dates) < 40:
        raise ValueError("无未来函数验收至少需要 40 个交易日")
    if cut is None:
        cut = dates[int(len(dates) * 0.6)]
    if cut not in market.calendar.dates:
        raise ValueError(f"截断点 {cut} 必须是交易日")

    truncated = market.truncate(cut)

    # --- 证据 1：数据确实被物理删除（否则「曲线一致」毫无意义） ---
    removed = {
        "full_bars_after_cut": market.panel.bars_after(cut),
        "truncated_bars_after_cut": truncated.panel.bars_after(cut),
        "full_reports_after_cut": _reports_after(market, cut),
        "truncated_reports_after_cut": _reports_after(truncated, cut),
    }
    data_actually_removed = (
        removed["full_bars_after_cut"] > 0
        and removed["full_reports_after_cut"] > 0
        and removed["truncated_bars_after_cut"] == 0
        and removed["truncated_reports_after_cut"] == 0
    )

    sample = [dates[int(len(dates) * f)] for f in (0.10, 0.25, 0.40, 0.55)]
    pit_ok = _cross_section_pit(market, sample)

    # --- 证据 2：截断重跑，净值前缀必须逐日一致 ---
    full_res = run_fn(market, cfg, signal_fn, label="accept_full")
    part_res = run_fn(truncated, cfg, signal_fn, label="accept_truncated")
    cmp_strategy = _compare_prefix(full_res.equity_curve, part_res.equity_curve, cut)
    abs_diff = cmp_strategy["max_abs_equity_diff"]
    equity_invariant = bool(
        cmp_strategy["aligned"]
        and abs_diff is not None
        and abs_diff <= tol
    )

    # --- 证据 3：蜜罐反例必须被抓住 ---
    cheat = make_cheating_signal(cut)
    sig_full = _signal_signature(market, cheat, sample)
    sig_part = _signal_signature(truncated, cheat, sample)
    honeypot_scores_differ = sig_full != sig_part

    cheater_equity_diff = None
    if deep_power_check:
        cheat_full = run_fn(market, cfg, cheat, label="honeypot_full")
        cheat_part = run_fn(truncated, cfg, cheat, label="honeypot_truncated")
        cmp_cheat = _compare_prefix(cheat_full.equity_curve,
                                    cheat_part.equity_curve, cut)
        cheater_equity_diff = cmp_cheat["max_abs_equity_diff"]
        cheater_caught = bool(
            honeypot_scores_differ
            and cheater_equity_diff is not None
            and cheater_equity_diff > tol
        )
    else:
        cheater_caught = honeypot_scores_differ

    passed = bool(data_actually_removed and pit_ok and equity_invariant
                  and cheater_caught)
    return {
        "cut": cut,
        "tol": tol,
        "cross_section_pit": pit_ok,
        "data_removed": removed,
        "data_actually_removed": data_actually_removed,
        "strategy_prefix": cmp_strategy,
        "equity_invariant": equity_invariant,
        "honeypot_scores_differ": honeypot_scores_differ,
        "honeypot_max_abs_equity_diff": cheater_equity_diff,
        "honeypot_caught": cheater_caught,
        "deep_power_check": deep_power_check,
        "passed": passed,
    }
