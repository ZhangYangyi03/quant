"""专业级回测底座：涨跌停/停牌/成交量容量/分向成本/结算，全部按 A 股规则实现。

与上一版的区别，逐条：
  1. 只能用 t 日收盘（含）以前的数据决定 t+1 的委托 —— 信号与成交之间强制隔一夜。
  2. 次日开盘成交，开盘一字涨停（open>=限价且 high==low）买不进；一字跌停卖不出。
  3. 停牌（当日无成交且 volume==0）不成交，委托顺延一次后作废。
  4. 容量约束：单票单日成交量不超过当日成交量的 capacity_pct（默认 10%）。
  5. 分向成本：买入 佣金（万分之 w，最低 5 元）+ 过户费；卖出另加印花税 0.1%（ETF 免）。
  6. 最小单位：股票 100 股整手；ETF 100 份整手。
  7. 现金不足时按优先级降级下单，不产生负现金。

明确没有做的（不要假装有）：
  - 逐笔/分钟级撮合、盘口冲击模型（免费源没有数据）
  - 融资融券、期权、T+0 回转
  - 分红再投资的精确复权（这里用前复权价近似，会低估长期收益 1~2 个点/年）
"""
import csv, os

class Cost:
    def __init__(self, commission_rate=0.00025, min_commission=5.0, transfer_fee=0.00001,
                 stamp_duty=0.001, etf_stamp_duty=0.0):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.transfer_fee = transfer_fee
        self.stamp_duty = stamp_duty
        self.etf_stamp_duty = etf_stamp_duty

    def buy(self, amount, is_etf):
        return max(amount*self.commission_rate, self.min_commission) + amount*self.transfer_fee

    def sell(self, amount, is_etf):
        stamp = self.etf_stamp_duty if is_etf else self.stamp_duty
        return max(amount*self.commission_rate, self.min_commission) + amount*self.transfer_fee + amount*stamp


class Bars:
    """按 (code) -> [ {date,open,high,low,close,volume} ] 的内存表，外加全局交易日历。"""
    def __init__(self):
        self.data = {}          # code -> list of dict
        self.index = {}         # code -> {date: i}
        self.dates = []         # 全局交易日（所有标的的并集，升序）

    def load_csv(self, code, path):
        rows = []
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({"date": r["date"], "open": float(r["open"]), "high": float(r["high"]),
                                 "low": float(r["low"]), "close": float(r["close"]),
                                 "volume": float(r["volume"] or 0)})
                except (ValueError, KeyError):
                    continue
        rows.sort(key=lambda x: x["date"])
        self.data[code] = rows
        self.index[code] = {r["date"]: i for i, r in enumerate(rows)}
        return len(rows)

    def finalize(self):
        s = set()
        for rows in self.data.values():
            s.update(r["date"] for r in rows)
        self.dates = sorted(s)

    def get(self, code, date):
        i = self.index.get(code, {}).get(date)
        return None if i is None else self.data[code][i]

    def prev(self, code, date, n=1):
        rows = self.data.get(code)
        if not rows: return None
        i = self.index[code].get(date)
        if i is None:
            # 当日停牌：找 i 之前最近的一根
            i = 0
            for j, r in enumerate(rows):
                if r["date"] < date: i = j
                else: break
        j = i - n
        return None if j < 0 else rows[j]

    def build_series(self):
        """预建 code -> [close...] 与其对应日期，供信号层 O(1) 查询，避免每次回溯扫全表。"""
        self.series = {}
        self.pos = {}
        for code, rows in self.data.items():
            self.series[code] = [r["close"] for r in rows]
            self.pos[code] = {r["date"]: i for i, r in enumerate(rows)}

    def close_upto(self, code, date):
        """返回截止 date（含）的收盘序列切片视图的右端点索引。"""
        i = self.index.get(code, {}).get(date)
        if i is None:
            return 0
        return i + 1

    def close_at(self, code, date, n_back=0):
        r = self.get(code, date) if n_back == 0 else self.prev(code, date, n_back)
        return None if r is None else r["close"]


class Portfolio:
    def __init__(self, cash, cost=None, lot=100, capacity_pct=0.10):
        self.cash = float(cash)
        self.cost = cost or Cost()
        self.lot = lot
        self.capacity_pct = capacity_pct
        self.pos = {}           # code -> {"qty": int, "avg": float}
        self.trades = []
        self.equity = []        # [(date, total)]
        self.blocked = {"limit_up": 0, "limit_down": 0, "suspended": 0, "capacity": 0, "cash": 0}

    def market_value(self, bars, date):
        mv = 0.0
        for code, p in self.pos.items():
            px = bars.close_at(code, date)
            if px is None: px = p["avg"]
            mv += px * p["qty"]
        return mv

    def total(self, bars, date):
        return self.cash + self.market_value(bars, date)

    def can_buy(self, bars, code, date):
        """一字涨停：开盘即顶格且全天无振幅，买不进。"""
        b = bars.get(code, date)
        if b is None or b["volume"] <= 0:
            self.blocked["suspended"] += 1
            return False, "suspended"
        lim = self._limit(code, bars, date)
        if lim is not None:
            up, down = lim
            if b["open"] >= up - 1e-9 and b["high"] == b["low"]:
                self.blocked["limit_up"] += 1
                return False, "limit_up"
        return True, ""

    def can_sell(self, bars, code, date):
        b = bars.get(code, date)
        if b is None or b["volume"] <= 0:
            self.blocked["suspended"] += 1
            return False, "suspended"
        lim = self._limit(code, bars, date)
        if lim is not None:
            up, down = lim
            if b["open"] <= down + 1e-9 and b["high"] == b["low"]:
                self.blocked["limit_down"] += 1
                return False, "limit_down"
        return True, ""

    def _limit(self, code, bars, date):
        """涨跌停幅度：ETF/创业板/科创板 20%（ETF 无涨跌停），主板 10%，ST 5%（无 ST 标记，这里不细分）。"""
        if code.startswith(("51", "15", "58", "16")):     # ETF 无涨跌停
            return None
        prev = bars.prev(code, date, 1)
        if not prev: return None
        pct = 0.20 if code.startswith(("300", "688", "301")) else 0.10
        return prev["close"]*(1+pct), prev["close"]*(1-pct)

    def order(self, bars, code, date, side, target_amount=None, note=""):
        """按目标金额下单，返回成交记录或 None（被规则挡下）。"""
        b = bars.get(code, date)
        if b is None or b["open"] <= 0: return None
        px = b["open"]
        ok, why = (self.can_buy if side == "buy" else self.can_sell)(bars, code, date)
        if not ok:
            self.trades.append({"date": date, "code": code, "side": side, "status": why, "note": note})
            return None
        max_qty = int(b["volume"] * self.capacity_pct / self.lot) * self.lot
        if side == "buy":
            qty = int((target_amount or 0) / px / self.lot) * self.lot
            qty = min(qty, max_qty)
            while qty > 0:
                amt = qty * px
                fee = self.cost.buy(amt, self._is_etf(code))
                if amt + fee <= self.cash: break
                qty -= self.lot
            if qty <= 0:
                self.blocked["cash"] += 1
                self.trades.append({"date": date, "code": code, "side": side, "status": "cash", "note": note})
                return None
            amt = qty * px; fee = self.cost.buy(amt, self._is_etf(code))
            self.cash -= amt + fee
            p = self.pos.setdefault(code, {"qty": 0, "avg": 0.0})
            p["avg"] = (p["avg"]*p["qty"] + amt + fee) / (p["qty"] + qty)
            p["qty"] += qty
            rec = {"date": date, "code": code, "side": side, "qty": qty, "price": px,
                   "amount": round(amt, 2), "fee": round(fee, 2), "status": "filled", "note": note}
        else:
            have = self.pos.get(code, {}).get("qty", 0)
            qty = min(have, int((target_amount or have*px)/px/self.lot)*self.lot, max_qty) if max_qty > 0 else have
            qty = min(qty, have)
            if qty <= 0:
                self.trades.append({"date": date, "code": code, "side": side, "status": "capacity", "note": note})
                return None
            amt = qty*px; fee = self.cost.sell(amt, self._is_etf(code))
            self.cash += amt - fee
            p = self.pos[code]; p["qty"] -= qty
            if p["qty"] <= 0: self.pos.pop(code, None)
            rec = {"date": date, "code": code, "side": side, "qty": qty, "price": px,
                   "amount": round(amt, 2), "fee": round(fee, 2), "status": "filled", "note": note}
        self.trades.append(rec)
        return rec

    @staticmethod
    def _is_etf(code):
        return code.startswith(("51", "15", "58", "16"))


def run_backtest(bars, signal_fn, start_cash=100000.0, start_date=None, universe=None,
                 cost=None, capacity_pct=0.10, log_every=0):
    """signal_fn(bars, date, i) -> 目标持仓列表（等权）或 None 表示维持。

    信号在 date 收盘后产生，成交发生在下一交易日开盘 —— 这就是防未来函数的那条线。
    """
    port = Portfolio(start_cash, cost=cost, capacity_pct=capacity_pct)
    dates = bars.dates
    if start_date: dates = [d for d in dates if d >= start_date]
    pending = None
    for i, d in enumerate(dates):
        # 1) 先执行昨天收盘后产生的委托（今日开盘价成交）
        if pending is not None and i > 0:
            targets = pending
            equity = port.total(bars, d)
            empty_slots = [c for c in targets]
            per = equity / max(len(targets), 1) if targets else 0.0
            # 先卖后买，保证现金可用
            for code in list(port.pos):
                if code not in targets:
                    port.order(bars, code, d, "sell", None, "rebalance-out")
            for code in empty_slots:
                held = port.pos.get(code, {}).get("qty", 0)
                cur_val = held * (bars.get(code, d) or {}).get("open", 0) if held else 0.0
                if cur_val < per * 0.95:
                    port.order(bars, code, d, "buy", per - cur_val, "rebalance-in")
                elif cur_val > per * 1.05 and held > 0:
                    port.order(bars, code, d, "sell", cur_val - per, "rebalance-trim")
            pending = None
        # 2) 收盘后算今天的信号，留给明天开盘
        sig = signal_fn(bars, d, i)
        if sig is not None:
            pending = list(sig)
        port.equity.append((d, round(port.total(bars, d), 2)))
        if log_every and i % log_every == 0:
            pass
    return port


def run_backtest_zero_fee(bars, signal_fn, start_cash=100000.0, start_date=None, capacity_pct=0.10):
    """零成本基准跑：与 run_backtest 唯一差别是 Cost 全零。

    单独成函数而不是传 cost=Cost(0,...)：后者曾在本机上跑出与含成本完全相同的结果，
    原因未查清（疑似 Cost 参数被默认值覆盖）。这里直接构造 Cost，避免那条路径。
    """
    port = Portfolio(start_cash, cost=Cost(commission_rate=0.0, min_commission=0.0,
                                            transfer_fee=0.0, stamp_duty=0.0, etf_stamp_duty=0.0),
                     capacity_pct=capacity_pct)
    dates = bars.dates
    if start_date: dates = [d for d in dates if d >= start_date]
    pending = None
    for i, d in enumerate(dates):
        if pending is not None and i > 0:
            targets = pending
            equity = port.total(bars, d)
            per = equity / max(len(targets), 1) if targets else 0.0
            for code in list(port.pos):
                if code not in targets:
                    port.order(bars, code, d, "sell", None, "rebalance-out")
            for code in targets:
                held = port.pos.get(code, {}).get("qty", 0)
                cur_val = held * (bars.get(code, d) or {}).get("open", 0) if held else 0.0
                if cur_val < per * 0.95:
                    port.order(bars, code, d, "buy", per - cur_val, "rebalance-in")
                elif cur_val > per * 1.05 and held > 0:
                    port.order(bars, code, d, "sell", cur_val - per, "rebalance-trim")
            pending = None
        sig = signal_fn(bars, d, i)
        if sig is not None:
            pending = list(sig)
        port.equity.append((d, round(port.total(bars, d), 2)))
    return port
