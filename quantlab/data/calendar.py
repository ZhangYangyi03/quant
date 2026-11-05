"""交易日历：调仓日必须落在真实交易日上，不允许 i % 20 == 0 这类错位切片。"""
from datetime import date, timedelta

from ..clock import AsOf


def _parse(s: str) -> date:
    y, m, d = (int(x) for x in s.split("-"))
    return date(y, m, d)


class TradingCalendar:
    def __init__(self, dates) -> None:
        self._dates = tuple(sorted({str(d) for d in dates}))
        self._idx = {d: i for i, d in enumerate(self._dates)}

    def __len__(self) -> int:
        return len(self._dates)

    def __iter__(self):
        return iter(self._dates)

    def __getitem__(self, i: int) -> str:
        return self._dates[i]

    @property
    def dates(self) -> tuple[str, ...]:
        return self._dates

    def index(self, d: str) -> int:
        return self._idx[d]

    def contains(self, d: str) -> bool:
        return d in self._idx

    def next_session(self, d: str) -> str | None:
        i = self._idx.get(d)
        if i is None:
            i = next((j for j, x in enumerate(self._dates) if x > d), None)
            return self._dates[i] if i is not None else None
        return self._dates[i + 1] if i + 1 < len(self._dates) else None

    def between(self, start: str, end: str) -> tuple[str, ...]:
        return tuple(d for d in self._dates if start <= d <= end)

    def rebalance_dates(self, start: str, end: str, freq: str = "monthly") -> tuple[str, ...]:
        """每个周期的第一个交易日。"""
        span = self.between(start, end)
        if not span:
            return ()
        if freq == "weekly":
            key = lambda d: (int(d[:4]), _parse(d).isocalendar()[1])
        elif freq == "quarterly":
            key = lambda d: (d[:4], (int(d[5:7]) - 1) // 3)
        elif freq == "monthly":
            key = lambda d: d[:7]
        else:
            raise ValueError(f"未知调仓频率: {freq}")
        out, seen = [], set()
        for d in span:
            k = key(d)
            if k not in seen:
                seen.add(k)
                out.append(d)
        return tuple(out)

    @staticmethod
    def business_days(start: str, count: int) -> tuple[str, ...]:
        cur = _parse(start)
        out = []
        while len(out) < count:
            if cur.weekday() < 5:
                out.append(cur.isoformat())
            cur += timedelta(days=1)
        return tuple(out)

    def asof(self, d: str) -> AsOf:
        return AsOf(d)
