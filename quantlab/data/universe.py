"""Point-in-time 股票池。

反幸存者偏差的关键：退市公司在退市日之前必须一直留在池子里，
退市后才从可选域移除；持仓中的退市股按清偿率计价，不产生现金。
ST 状态同样按时间窗管理，绝不能把「ST 数据缺失」当成「非 ST」。
"""
from dataclasses import dataclass

from ..clock import AsOf


@dataclass(frozen=True)
class Listing:
    code: str
    listed_on: str
    delisted_on: str | None = None
    board: str = "MAIN"                       # MAIN | GEM | STAR
    st_windows: tuple[tuple[str, str], ...] = ()  # 闭区间 [start, end]
    industry: str = "UNKNOWN"


class Universe:
    def __init__(self) -> None:
        self._items: dict[str, Listing] = {}

    def add(self, item: Listing) -> None:
        if item.delisted_on and item.delisted_on <= item.listed_on:
            raise ValueError("退市日必须晚于上市日")
        self._items[item.code] = item

    def __len__(self) -> int:
        return len(self._items)

    def all_codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def get(self, code: str) -> Listing | None:
        return self._items.get(code)

    def members(self, at: AsOf) -> tuple[str, ...]:
        """asof 当日可交易的标的（含尚未退市的历史退市股）。"""
        out = []
        for code, it in self._items.items():
            if it.listed_on <= at.date and (it.delisted_on is None or at.date < it.delisted_on):
                out.append(code)
        return tuple(sorted(out))

    def is_st(self, code: str, at: AsOf) -> bool:
        it = self._items.get(code)
        if it is None:
            return False
        return any(a <= at.date <= b for a, b in it.st_windows)

    def is_delisted(self, code: str, at: AsOf) -> bool:
        it = self._items.get(code)
        return bool(it and it.delisted_on and at.date >= it.delisted_on)

    def board(self, code: str) -> str:
        it = self._items.get(code)
        return it.board if it else "MAIN"

    def industry(self, code: str) -> str:
        it = self._items.get(code)
        return it.industry if it else "UNKNOWN"

    def last_listing_date(self, code: str) -> str | None:
        it = self._items.get(code)
        return it.delisted_on

    def survivors_only_count(self, at: AsOf) -> int:
        """只含「活到最后」的标的数，用于量化幸存者偏差。"""
        return sum(1 for it in self._items.values()
                   if it.delisted_on is None and it.listed_on <= at.date)
