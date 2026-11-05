"""集中配置：所有业务数字只能在这里出现，禁止散落为魔法数字。

frozen dataclass 保证一次运行内配置不可变；fingerprint 把配置哈希写进结果文件，
使任何一条净值曲线都能追溯到确切的参数集合。
"""
from dataclasses import dataclass, field, asdict
from hashlib import sha256
import json


@dataclass(frozen=True)
class CostConfig:
    """A 股成本口径。bps = 万分之一。"""
    commission_bps: float = 2.5          # 佣金，双边
    min_commission: float = 5.0          # 单笔最低佣金（元）
    transfer_bps: float = 1.0            # 过户费，双边
    stamp_bps_before: float = 10.0       # 卖出印花税，2023-08-28 之前
    stamp_bps_after: float = 5.0         # 2023-08-28 起减半
    stamp_cut_date: str = "2023-08-28"
    slippage_bps: float = 10.0           # 固定滑点（与冲击二选一，避免重复计费）
    impact_eta: float = 0.35             # 平方根冲击系数 Y*eta*sigma*sqrt(Q/ADV)
    impact_sigma: float = 0.02           # 日波动率，用于冲击模型
    adv_cap: float = 0.10                # 单票单日不超过当日成交额的比例
    use_impact_model: bool = True        # True: 用冲击代替固定滑点；False: 用固定滑点


@dataclass(frozen=True)
class ExecutionConfig:
    lot: int = 100
    t_plus_1: bool = True                # 当日买入次日才可卖
    limit_main: float = 0.10             # 主板涨跌停
    limit_gem: float = 0.20              # 创业板/科创板
    delisting_recovery: float = 0.0      # 退市清偿率；0 表示归零且不产生现金
    min_price: float = 2.0               # 低于此价不参与选股
    require_open_gt_zero: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "2020-01-01"
    end: str = "2023-12-31"
    cash: float = 1_000_000.0
    top_k: int = 20
    rebalance: str = "monthly"           # monthly | weekly | quarterly
    signal: str = "ep"                   # ep | size | lowvol | random
    weighting: str = "equal"             # equal | inverse_vol | min_variance
    benchmark: str = "random_k_equal"    # 同池、同 k、同调仓频率的随机抽样基准
    seed: int = 20261105


@dataclass(frozen=True)
class ValidationConfig:
    trading_days: int = 250              # 年化口径（与本项目历史结果一致）
    nw_lags: int | None = None           # None -> floor(4*(T/100)^(2/9))
    bootstrap_draws: int = 500
    pbo_slices: int = 8
    placebo_draws: int = 200
    alpha: float = 0.05


@dataclass(frozen=True)
class Config:
    cost: CostConfig = field(default_factory=CostConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        return sha256(blob.encode("utf-8")).hexdigest()[:16]
