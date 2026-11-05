"""行情面板与 CrossSection：策略只能看到 asof 及之前的信息。

关键设计：Market.view(date) 返回一个**独立的只读截面**，其中不含任何
date 之后的 Bar。策略函数接收 CrossSection 而非 Market，因此即使有人
故意写 `bars[date+1]` 也拿不到数据 —— 未来函数从「纪律问题」变成「物理不可能」。
"""
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json

from ..clock import AsOf
from .pit import PITFundamentals
from .universe import Universe
from .calendar import TradingCalendar


@dataclass(frozen=True)
class Bar:
    date: str
    code: str
    open: float
    high: float
    low: float
    close: float
    amount: float          # 当日成交额（元）
    volume: float
    mktcap: float          # 当日总市值（元），非成交额代理
    suspended: bool = False
    limit_up: bool = False   # 一字涨停，买不进
    limit_down: bool = False  # 一字跌停，卖不出

    @property
    def tradable(self) -> bool:
        return (not self.suspended) and self.volume > 0 and self.open > 0


class Panel:
    def __init__(self) -> None:
        self._bars: dict[str, dict[str, Bar]] = {}
        self._sorted_cache: dict[str, tuple[str, ...]] = {}

    def add(self, bar: Bar) -> None:
        self._bars.setdefault(bar.code, {})[bar.date] = bar
        # 增量写入时让该代码的排序缓存失效，避免 history() 每次全量排序。
        self._sorted_cache.pop(bar.code, None)

    def __len__(self) -> int:
        return sum(len(v) for v in self._bars.values())

    def codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._bars))

    def _sorted_dates(self, code: str) -> tuple[str, ...]:
        keys = self._sorted_cache.get(code)
        if keys is None:
            keys = tuple(sorted(self._bars.get(code, {})))
            self._sorted_cache[code] = keys
        return keys

    def get(self, code: str, date: str, at: AsOf) -> Bar | None:
        at.assert_visible(date, f"Bar({code})")
        return self._bars.get(code, {}).get(date)

    def history(self, code: str, at: AsOf, lookback: int = 20) -> tuple[Bar, ...]:
        """截至 asof 的最近 lookback 根 K 线，按时间升序。

        用二分定位上界，物理上不可能返回 asof 之后的 Bar；缓存排序键使
        事件驱动循环里每根 K 线的查询从 O(D log D) 降到 O(log D)。
        """
        rows = self._bars.get(code, {})
        if not rows:
            return ()
        keys = self._sorted_dates(code)
        hi = bisect_right(keys, at.date)
        lo = max(0, hi - max(1, lookback))
        return tuple(rows[d] for d in keys[lo:hi])

    def bars_after(self, date: str) -> int:
        """date 之后的 Bar 数量，供无未来函数验收做「数据确实被删掉」的证据。"""
        return sum(len(self._sorted_dates(c)) - bisect_right(self._sorted_dates(c), date)
                   for c in self._bars)

    def prev_close(self, code: str, date: str, at: AsOf) -> float | None:
        h = self.history(code, AsOf(date), 2)
        return h[0].close if len(h) >= 2 else None

    def fingerprint(self) -> str:
        blob = json.dumps(
            {c: {d: [round(b.open, 4), round(b.close, 4), round(b.amount, 2)]
                 for d, b in sorted(v.items())}
             for c, v in sorted(self._bars.items())},
            sort_keys=True, separators=(",", ":"))
        return sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CrossSection:
    """某交易日收盘时点的完整信息集。策略函数的唯一输入。"""
    at: AsOf
    members: tuple[str, ...]
    bars: dict[str, Bar]
    history: dict[str, tuple[Bar, ...]]
    fundamentals: PITFundamentals
    universe: Universe

    def bar(self, code: str) -> Bar | None:
        b = self.bars.get(code)
        if b is not None:
            self.at.assert_visible(b.date, f"Bar({code})")
        return b

    def close(self, code: str) -> float | None:
        b = self.bar(code)
        return b.close if b and b.close > 0 else None

    def mktcap(self, code: str) -> float | None:
        b = self.bar(code)
        return b.mktcap if b and b.mktcap > 0 else None

    def adv(self, code: str, lookback: int = 20) -> float:
        h = self.history.get(code, ())
        vals = [b.amount for b in h[-lookback:] if b.amount > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def ep(self, code: str) -> float | None:
        """E/P：已公告 TTM EPS / 当日收盘价。"""
        px = self.close(code)
        eps = self.fundamentals.ttm_eps(code, self.at)
        if px is None or eps is None or px <= 0:
            return None
        return eps / px

    def bp(self, code: str) -> float | None:
        px = self.close(code)
        bps = self.fundamentals.field(code, self.at, "bps")
        return bps / px if px and bps is not None and px > 0 else None

    def volatility(self, code: str, lookback: int = 20) -> float | None:
        h = self.history.get(code, ())
        cs = [b.close for b in h[-(lookback + 1):] if b.close > 0]
        if len(cs) < 3:
            return None
        rs = [cs[i] / cs[i - 1] - 1 for i in range(1, len(cs))]
        mu = sum(rs) / len(rs)
        return (sum((r - mu) ** 2 for r in rs) / (len(rs) - 1)) ** 0.5


class Market:
    """聚合数据源。所有访问经由 view(date) 构造只读截面。"""

    def __init__(self, calendar: TradingCalendar, universe: Universe,
                 panel: Panel, fundamentals: PITFundamentals) -> None:
        self.calendar = calendar
        self.universe = universe
        self.panel = panel
        self.fundamentals = fundamentals

    def view(self, date: str) -> CrossSection:
        if not self.calendar.contains(date):
            raise KeyError(f"{date} 不是交易日")
        at = AsOf(date)
        members = self.universe.members(at)
        bars, hist = {}, {}
        for code in self.universe.all_codes():
            b = self.panel.get(code, date, at)
            if b is not None:
                bars[code] = b
            hist[code] = self.panel.history(code, at, 21)
        return CrossSection(at=at, members=members, bars=bars, history=hist,
                            fundamentals=self.fundamentals, universe=self.universe)

    def truncate(self, date: str) -> "Market":
        """生成只含 <= date 数据的 Market，用于无未来函数验收。"""
        cal = TradingCalendar(d for d in self.calendar if d <= date)
        p = Panel()
        for code in self.panel.codes():
            for b in self.panel.history(code, AsOf(date), 10 ** 9):
                p.add(b)
        f = PITFundamentals()
        for code in self.universe.all_codes():
            for r in self.fundamentals.visible(code, AsOf(date)):
                f.add(r)
        return Market(cal, self.universe, p, f)

    def fingerprint(self) -> str:
        return self.panel.fingerprint()
