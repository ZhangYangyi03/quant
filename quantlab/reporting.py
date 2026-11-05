"""运行留痕（audit trail）：让每一条净值曲线都能追溯到唯一一次运行。

历史项目里最重要的工程缺陷不是收益数字，而是「数字与产出它的代码/数据之间
没有绑定关系」——改了成本口径、换了数据文件，旧 JSON 还躺在仓库里被继续引用。
本模块强制把 配置指纹 + 数据指纹 + 随机种子 + 环境 + 完整净值 写进同一份产物。
"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone

from .config import Config
from .backtest.engine import Result

SCHEMA = "quantlab.run.v1"


def result_record(result: Result, cfg: Config, extra: dict | None = None) -> dict:
    """把一次回测压成可复现、可 diff 的字典。"""
    rec = {
        "schema": SCHEMA,
        "label": result.label,
        "config_fingerprint": result.config_fingerprint,
        "data_fingerprint": result.data_fingerprint,
        "seed": result.seed,
        "config": cfg.as_dict(),
        "metrics": result.metrics(cfg),
        "trades": {
            "n_orders": len(result.fills),
            "n_rejected": len(result.rejected),
            "blocked_reasons": dict(result.blocked),
            "fees_paid": round(result.fees_paid, 2),
            "turnover": round(result.turnover, 2),
        },
        "equity_curve": [[d, v] for d, v in result.equity_curve],
    }
    if extra:
        rec.update(extra)
    return rec


def write_run(path: str, payload: dict) -> str:
    """原子写入 JSON 产物，返回绝对路径。"""
    full = os.path.abspath(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    envelope = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "schema": SCHEMA,
        "payload": payload,
    }
    tmp = full + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(envelope, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, full)
    return full
