"""事件驱动主循环：t 日收盘出信号 -> t+1 日开盘撮合。

这条「信号与成交强制隔夜」是唯一能拦住未来函数的硬规则，本引擎通过
CrossSection 视图 + 延迟一日的 pending 队列双重保证，策略函数无从越界。
"""
from dataclasses import dataclass, field
from hashlib import sha256
import math

from ..config import Config
from ..data.panel import CrossSection, Market
from ..model.portfolio import build_weights
from .accounting import Book
from .execution import Executor, Order


def default_signal(name: str = "ep", seed: int = 0):
    """返回只依赖 CrossSection 的打分函数。random 也用固定哈希，保证可复现。"""
    def _hash_score(cs: CrossSection, code: str) -> float:
        h = sha256(f"{seed}|{cs.at.date}|{code}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64

    def fn(cs: CrossSection) -> dict[str, float]:
        out = {}
        for c in cs.members:
            if name == "ep":
                out[c] = cs.ep(c)
            elif name == "bp":
                out[c] = cs.bp(c)
            elif name == "size":
                cap = cs.mktcap(c)
                out[c] = -math.log(cap) if cap else None
            elif name == "lowvol":
                v = cs.volatility(c)
                out[c] = -v if v else None
            elif name == "random":
                out[c] = _hash_score(cs, c)
            else:
                raise ValueError(f"未知信号: {name}")
        return out
    return fn


@dataclass
class Result:
    equity_curve: list[tuple[str, float]]
    fills: list
    blocked: dict[str, int]
    fees_paid: float
    turnover: float
    config_fingerprint: str
    data_fingerprint: str
    seed: int
    label: str = "strategy"
    rejected: list = field(default_factory=list)

    def daily_returns(self) -> list[float]:
        eq = [v for _, v in self.equity_curve]
        return [eq[i] / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]

    def metrics(self, cfg: Config) -> dict:
        from ..validate.stats import performance
        return performance(self.equity_curve, cfg.validation)


def _orders_from_target(book: Book, target: dict[str, float], cs: CrossSection,
                        equity: float, min_ticket: float = 2000.0) -> list[Order]:
    """目标权重 -> 订单。先卖后买，卖出只使用 T+1 可用股份。"""
    orders: list[Order] = []
    px = {c: cs.close(c) for c in set(book.positions) | set(target)}
    for code in list(book.positions):
        if code not in target:
            sh = book.available_shares(code)
            if sh > 0 and (px.get(code) or 0) > 0:
                orders.append(Order(code, "SELL", sh, "drop_from_target"))
    for code, w in target.items():
        price = px.get(code)
        if not price or price <= 0:
            continue
        want = w * equity
        have = book.positions.get(code, 0) * price
        diff = want - have
        if diff < -min_ticket:
            sh = min(int(-diff / price), book.available_shares(code))
            if sh > 0:
                orders.append(Order(code, "SELL", sh, "trim_to_target"))
        elif diff > min_ticket:
            sh = int(diff / price)
            if sh > 0:
                orders.append(Order(code, "BUY", sh, "raise_to_target"))
    return orders


def run_backtest(market: Market, cfg: Config, signal_fn=None,
                 label: str = "strategy") -> Result:
    bt = cfg.backtest
    if signal_fn is None:
        signal_fn = default_signal(bt.signal, bt.seed)
    book = Book(cfg)
    exe = Executor(cfg)
    dates = market.calendar.between(bt.start, bt.end)
    if len(dates) < 30:
        raise ValueError("回测区间不足 30 个交易日")
    rebal = set(market.calendar.rebalance_dates(bt.start, bt.end, bt.rebalance))
    pending: list[Order] = []

    for d in dates:
        cs = market.view(d)
        book.release_locks()
        for o in pending:
            fill = exe.execute(o, d, cs.bar(o.code), cs.adv(o.code), book)
            book.apply_fill(fill)
        pending = []
        book.mark(d, cs)
        if d in rebal:
            scores = signal_fn(cs)
            weights = build_weights(scores, cs, bt.top_k, bt.weighting,
                                    min_price=cfg.execution.min_price)
            pending = _orders_from_target(book, weights, cs, book.equity())

    result = Result(
        equity_curve=book.equity_curve, fills=book.fills,
        blocked=dict(book.blocked), fees_paid=book.fees_paid,
        turnover=book.turnover, config_fingerprint=cfg.fingerprint(),
        data_fingerprint=market.fingerprint(), seed=bt.seed, label=label,
    )
    result.rejected = [f for f in book.fills if f.status == "REJECTED"]
    return result


def run_benchmark(market: Market, cfg: Config) -> Result:
    """同池、同 k、同调仓频率、同费用的随机抽样等权基准。

    这直接修复历史项目里「eq_nofilter 落在代码最小的 50 只固定老股、
    16 年只成交 993 笔」的基准缺陷。
    """
    from dataclasses import replace
    bcfg = replace(cfg.backtest, signal="random", weighting="equal")
    bfull = Config(cost=cfg.cost, execution=cfg.execution,
                   backtest=bcfg, validation=cfg.validation)
    return run_backtest(market, bfull,
                        signal_fn=default_signal("random", cfg.backtest.seed + 1),
                        label="benchmark_random_k_equal")
