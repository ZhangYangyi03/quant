"""执行层：订单管理（OMS）、盘前风控、券商适配与成交回报对账。

与 backtest/execution.py 的区别：那里是回测撮合器（给 Book 记账），
这里是**可对接实盘的订单生命周期**：
  target weights -> order plan -> pre-trade risk -> broker -> fills -> reconcile
每一步都留痕，任何拒绝都带原因，任何未成交都能在 reconcile 里追到。

设计约束：
  - 不联网、不下真单；BrokerAdapter 只提供 paper（历史开盘价撮合）与 csv（导出）。
  - 订单只含纯数据，能被 json 序列化，便于对账与审计。
  - 与 quantlab.model.cost / quantlab.backtest.execution 共用同一套成本与拒单原因。
"""
from .oms import OrderPlan, PreTradeRisk, OrderRecord, FillRecord, reconcile
from .broker import BrokerAdapter, PaperBroker, CsvBroker

__all__ = ["OrderPlan", "PreTradeRisk", "OrderRecord", "FillRecord", "reconcile",
           "BrokerAdapter", "PaperBroker", "CsvBroker"]
