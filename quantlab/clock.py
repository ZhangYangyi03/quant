"""PIT 时钟与越界守卫。

机构研究栈与散装脚本最本质的区别：策略函数拿到的不是「全量数据 + 自觉」，
而是一个物理上只包含 asof 及之前信息的视图。任何对未来数据的访问都会抛异常，
而不是靠 code review 发现。
"""
from dataclasses import dataclass


class LookaheadError(RuntimeError):
    """试图读取 asof 之后才可获得的信息。"""


@dataclass(frozen=True, order=True)
class AsOf:
    """信息时点。date 为 ISO 日期字符串，字典序即时间序。"""
    date: str

    def assert_visible(self, observed_at: str, what: str = "data") -> None:
        if observed_at > self.date:
            raise LookaheadError(
                f"未来函数：在 {self.date} 试图读取 {observed_at} 才可知的 {what}"
            )

    def __str__(self) -> str:
        return self.date


def asof(date: str) -> AsOf:
    return AsOf(date)
