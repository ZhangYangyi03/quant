"""纸面交易账本：按真实收盘价记账，不产生任何真实委托。"""
import sqlite3, time

def nav(con, date):
    """总资产 = 现金 + 持仓市值；缺价用最近一根。"""
    cash = con.execute("select cash from paper_account where id=1").fetchone()[0]
    mv = 0.0
    for sym, qty, _ in con.execute("select symbol,qty,cost from paper_position"):
        row = con.execute("select close from bars where ts_code=? and trade_date<=? order by trade_date desc limit 1",(sym,date)).fetchone()
        mv += (row[0] if row else 0) * qty
    return cash, mv, cash+mv

def rebalance(con, holdings, date=None, cash_floor=0.0):
    """把持仓调整到 holdings（等权）。1 元账户买不起一手，只会记下“意图”而不是假成交。"""
    date = date or con.execute("select max(trade_date) from bars").fetchone()[0]
    cash = con.execute("select cash from paper_account where id=1").fetchone()[0]
    if not holdings:
        return {"ok": True, "action": "空仓", "cash": cash}
    per = (cash - cash_floor) / len(holdings)
    plans = []
    for sym in holdings:
        px = con.execute("select close from bars where ts_code=? and trade_date<=? order by trade_date desc limit 1",(sym,date)).fetchone()
        px = px[0] if px else None
        lot_cost = px * 100 if px else None
        plans.append({"symbol": sym, "price": px, "lot_cost_100": round(lot_cost,2) if lot_cost else None,
                      "affordable": bool(lot_cost and lot_cost <= per)})
    return {"ok": True, "date": date, "cash": round(cash,2), "per_slot": round(per,2), "plans": plans,
            "note": "任何 affordable=False 的标的都没有下单；不写假成交"}
