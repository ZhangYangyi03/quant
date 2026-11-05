"""真正的 walk-forward：参数只用历史信息选择，随后在样本外结算。

原项目 walkforward.py 的 docstring 承诺了这件事，代码却从未实现；
本模块把它做实，并且额外提供一个「未来信息依赖检测器」——
通过在平行世界里翻转未来收益，可以机械地判定选参函数是否偷看了未来。
"""
from dataclasses import dataclass
import inspect
from typing import Callable


class WalkForwardLeakError(RuntimeError):
    """选参函数的输出依赖了它本不该看到的未来数据。"""


@dataclass(frozen=True)
class WalkForwardResult:
    period: int
    chosen: str
    is_sharpe: float
    oos_return: float
    history_len: int


def _sharpe(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return mu / sd if sd > 0 else 0.0


def _accepts_future(choose: Callable) -> bool:
    """选参函数是否声明了 future 形参（或 **kwargs）。"""
    try:
        sig = inspect.signature(choose)
    except (TypeError, ValueError):
        return False
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD or p.name == "future":
            return True
    return False


def _call_choose(choose: Callable, hist: dict[str, list[float]],
                 future: dict[str, list[float]] | None) -> str:
    """future 是蜜罐：只在 detect_lookahead 里喂数据，正常 walk-forward 恒为 None。

    只接受 history 的函数永远走 choose(hist)，不会因为多传一个关键字而 TypeError。
    """
    if _accepts_future(choose):
        return choose(hist, future=future)
    return choose(hist)


def run_walkforward(returns_by_param: dict[str, list[float]],
                    choose: Callable[[dict[str, list[float]]], str],
                    min_history: int = 60, oos_len: int = 20,
                    strict: bool = True) -> list[WalkForwardResult]:
    """按时间切块：history -> 选参 -> 下一段样本外收益。

    choose 只收到 history 切片（dict[param, list[float]]），
    物理上拿不到未来；若它试图读取全局数据，由 detect_lookahead 负责抓。
    """
    keys = list(returns_by_param)
    if not keys:
        return []
    n = min(len(v) for v in returns_by_param.values())
    out = []
    t = min_history
    while t + oos_len <= n:
        hist = {k: list(returns_by_param[k][:t]) for k in keys}
        chosen = _call_choose(choose, hist, None)
        if chosen not in returns_by_param:
            raise ValueError(f"选参函数返回了未知参数: {chosen}")
        oos = returns_by_param[chosen][t:t + oos_len]
        out.append(WalkForwardResult(
            period=len(out), chosen=chosen,
            is_sharpe=round(_sharpe(hist[chosen]), 6),
            oos_return=sum(oos), history_len=t))
        if strict:
            # 防呆：选参函数不得修改历史切片
            for k in keys:
                if len(hist[k]) != t:
                    raise WalkForwardLeakError("选参函数修改了历史数据")
        t += oos_len
    return out


def detect_lookahead(choose: Callable[[dict[str, list[float]]], str],
                     returns_by_param: dict[str, list[float]],
                     min_history: int = 60, oos_len: int = 20) -> dict:
    """平行世界 / 蜜罐检验：历史完全相同，只把未来收益整体取反。

    两条独立的作弊路径都要能抓到：
      1. 选参函数声明 future 形参（或 **kwargs），直接读我们喂进去的蜜罐数据；
      2. 选参函数通过闭包引用调用方的 returns_by_param —— 因此调用期间我们
         **原地**翻转它的未来段，结束后在 finally 里逐元素恢复。

    诚实的 choose 在两个世界必须给出同一选择（changed_ratio == 0）。
    注意这是「意图/依赖」诊断而非数学证明：future 是诱饵，正常 walk-forward
    永远不会传入真实未来数据。
    """
    keys = list(returns_by_param)
    if not keys:
        return {"changed_ratio": 0.0, "tests": 0, "leak_detected": False}
    n = min(len(v) for v in returns_by_param.values())
    if n < min_history + oos_len:
        return {"changed_ratio": 0.0, "tests": 0, "leak_detected": False}
    mutable = all(isinstance(returns_by_param[k], list) for k in keys)
    changed, tests = 0, 0
    for t in range(min_history, n - oos_len + 1, max(oos_len, 1)):
        world_a = {k: list(returns_by_param[k]) for k in keys}
        world_b = {k: returns_by_param[k][:t] +
                   [-x for x in returns_by_param[k][t:]] for k in keys}
        hist_a = {k: list(v[:t]) for k, v in world_a.items()}
        hist_b = {k: list(v[:t]) for k, v in world_b.items()}

        a = _call_choose(choose, hist_a, world_a)

        if mutable:
            for k in keys:
                seq = returns_by_param[k]
                for j in range(t, len(seq)):
                    seq[j] = -seq[j]
        try:
            b = _call_choose(choose, hist_b, world_b)
        finally:
            if mutable:
                for k in keys:
                    seq = returns_by_param[k]
                    for j in range(t, len(seq)):
                        seq[j] = -seq[j]

        tests += 1
        if a != b:
            changed += 1
    ratio = changed / tests if tests else 0.0
    return {"changed_ratio": round(ratio, 4), "tests": tests,
            "leak_detected": ratio > 0.0, "honeypot": True}
