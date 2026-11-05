"""Point-in-time 财务数据。

两条机构级硬规则：
  1. 财报按「公告日」生效，不按报告期；报告期 + 法定披露截止日只作为公告日的保守兜底。
  2. 修订版本单独存一条记录。用今天看到的修订值回答过去的调仓问题就是未来函数，
     因此 asof 查询只返回「当时已公告」的记录中版本最新的那条。

TS 该模块不联网、不读数据库，由上层 feed 注入；这样单元测试可以精确构造反例。
"""
from collections import defaultdict
from dataclasses import dataclass

from ..clock import AsOf


def disclosure_deadline(period: str) -> str:
    """报告期 -> 法定最晚披露日。年报 120 天、一季 30、半年 62、三季 31。"""
    y, m, d = (int(x) for x in period.split("-"))
    table = {(3, 31): f"{y}-04-30", (6, 30): f"{y}-08-31",
             (9, 30): f"{y}-10-31", (12, 31): f"{y + 1}-04-30"}
    try:
        return table[(m, d)]
    except KeyError as exc:
        raise ValueError(f"非标准报告期: {period}") from exc


@dataclass(frozen=True)
class Report:
    code: str
    period: str           # 报告期，如 2022-03-31
    announced_at: str     # 实际公告日（或兜底截止日）
    eps: float | None = None
    bps: float | None = None
    roe: float | None = None
    version: int = 1      # 同一报告期的第几次披露（修订递增）
    restated: bool = False


class PITFundamentals:
    """按 asof 查询的时序财报库。"""

    def __init__(self) -> None:
        self._rows: dict[str, list[Report]] = defaultdict(list)

    def add(self, report: Report) -> None:
        if report.announced_at < report.period:
            raise ValueError("公告日不得早于报告期")
        self._rows[report.code].append(report)

    def __len__(self) -> int:
        return sum(len(v) for v in self._rows.values())

    def visible(self, code: str, at: AsOf) -> list[Report]:
        """当时能看到的全部记录。"""
        return [r for r in self._rows.get(code, ()) if r.announced_at <= at.date]

    def latest_per_period(self, code: str, at: AsOf) -> dict[str, Report]:
        """每个报告期取 asof 可见的最新版本；旧版本与未来修订都不可见。"""
        out: dict[str, Report] = {}
        for r in self.visible(code, at):
            old = out.get(r.period)
            if old is None or (r.version, r.announced_at) > (old.version, old.announced_at):
                out[r.period] = r
        return out

    def field(self, code: str, at: AsOf, name: str = "eps") -> float | None:
        """最新一期已公告的某个字段值。"""
        per = self.latest_per_period(code, at)
        if not per:
            return None
        latest = per[max(per)]
        return getattr(latest, name, None)

    def ttm_eps(self, code: str, at: AsOf) -> float | None:
        """滚动四季 EPS：年报直接取，季报用 本期 + 上年年报 - 上年同期。

        所有参与计算的记录都必须已经公告，否则退回最新可得值（保守）。
        """
        per = self.latest_per_period(code, at)
        if not per:
            return None
        periods = sorted(per)
        last = periods[-1]
        cur = per[last]
        if cur.eps is None:
            return None
        if last.endswith("-12-31"):
            return cur.eps
        y = int(last[:4])
        mmdd = last[5:]
        prev_annual = per.get(f"{y - 1}-12-31")
        prev_same = per.get(f"{y - 1}-{mmdd}")
        if (prev_annual and prev_same and prev_annual.eps is not None
                and prev_same.eps is not None):
            return cur.eps + prev_annual.eps - prev_same.eps
        return cur.eps
