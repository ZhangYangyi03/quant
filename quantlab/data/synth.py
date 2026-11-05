"""确定性合成市场：用于在没有真实行情的机器上验证整条管线。

郑重声明：合成数据只用于验证「代码正确性」，绝不用于宣称任何收益结论。
它刻意注入了机构回测最容易出错的情形：退市、停牌、一字涨跌停、
财报修订版本、ST 窗口、行业与市值暴露。

注意数据生成过程本身也不含未来函数：个股漂移来自一个固定的潜在质量 q，
而 q 通过「已公告的 EPS」逐步体现在价格里；任何 t 日的价格只依赖 t 日之前的信息。
"""
from dataclasses import replace
from datetime import date
import math
import random

from .calendar import TradingCalendar
from .panel import Bar, Panel, Market
from .pit import PITFundamentals, Report, disclosure_deadline
from .universe import Listing, Universe


def _quarters(start_year: int, end_year: int) -> list[str]:
    out = []
    for y in range(start_year, end_year + 1):
        out += [f"{y}-03-31", f"{y}-06-30", f"{y}-09-30", f"{y}-12-31"]
    return out


def generate_synthetic_market(n_codes: int = 60, n_days: int = 760,
                              seed: int = 20261105, start: str = "2020-01-01",
                              n_industries: int = 8) -> Market:
    rng = random.Random(seed)
    dates = TradingCalendar.business_days(start, n_days)
    cal = TradingCalendar(dates)
    uni, pan, fund = Universe(), Panel(), PITFundamentals()

    # 共同市场因子路径（事前生成，但每只股票只消费到自己那天为止）
    market_shocks = [rng.gauss(0.0002, 0.012) for _ in dates]

    for i in range(n_codes):
        code = f"{600000 + i:06d}.SH"
        quality = rng.gauss(0.0, 1.0)                 # 潜在质量，决定长期漂移
        shares = rng.uniform(2e7, 8e8)                # 总股本
        px = rng.uniform(4.0, 60.0)
        base_amt = rng.uniform(3e6, 9e7)
        industry = f"IND{i % n_industries:02d}"
        board = "GEM" if i % 7 == 0 else "MAIN"
        listed_on = dates[0]
        # 每 11 只安排一只中途退市；退市前的历史必须完整保留
        delisted_on = dates[int(n_days * 0.62)] if i % 11 == 5 else None
        st_windows: tuple[tuple[str, str], ...] = ()
        if i % 9 == 3:
            st_windows = ((dates[int(n_days * 0.3)], dates[int(n_days * 0.38)]),)
        uni.add(Listing(code=code, listed_on=listed_on, delisted_on=delisted_on,
                        board=board, st_windows=st_windows, industry=industry))

        # 财报：每季度一条；每 13 只安排一次「事后修订」，公告日更晚
        start_year = int(dates[0][:4])
        end_year = int(dates[-1][:4])
        for qi, period in enumerate(_quarters(start_year - 1, end_year)):
            ann = disclosure_deadline(period)
            eps = max(-0.5, 0.35 * quality + 0.12 * math.sin(qi / 2.0)
                      + rng.gauss(0, 0.06))
            bps = max(0.5, 4.0 + 0.6 * quality + rng.gauss(0, 0.3))
            fund.add(Report(code=code, period=period, announced_at=ann,
                            eps=eps, bps=bps, roe=eps / bps, version=1))
            if i % 13 == 4 and qi % 6 == 2:
                y, m, d = (int(x) for x in ann.split("-"))
                later = date(y, m, d).toordinal() + 45
                fund.add(Report(code=code, period=period,
                                announced_at=date.fromordinal(later).isoformat(),
                                eps=eps * 0.55, bps=bps, roe=eps * 0.55 / bps,
                                version=2, restated=True))

        # 价格路径：漂移只依赖固定 quality（公开信息），不含未来
        drift = 0.00016 * quality
        prev_close = px
        for j, d in enumerate(dates):
            if delisted_on and d >= delisted_on:
                continue
            beta = rng.uniform(0.6, 1.4)
            shock = beta * market_shocks[j] + rng.gauss(drift, 0.009)
            new_close = max(0.35, prev_close * math.exp(shock))
            op = prev_close * (1 + rng.gauss(0, 0.004))
            hi = max(op, new_close) * (1 + abs(rng.gauss(0, 0.002)))
            lo = min(op, new_close) * (1 - abs(rng.gauss(0, 0.002)))
            amount = base_amt * (0.6 + 0.8 * abs(rng.gauss(0, 1)))
            volume = amount / max(new_close, 0.01)

            suspended = (j > 10 and i % 17 == 2 and j % 97 == 0)
            limit_up = (not suspended and i % 23 == 7 and j % 131 == 0)
            limit_down = (not suspended and i % 29 == 11 and j % 149 == 0)
            if suspended:
                new_close, op, hi, lo, volume, amount = prev_close, prev_close, prev_close, prev_close, 0.0, 0.0
            elif limit_up:
                op = hi = lo = new_close = prev_close * 1.099
            elif limit_down:
                op = hi = lo = new_close = prev_close * 0.901

            pan.add(Bar(date=d, code=code, open=round(op, 4), high=round(hi, 4),
                        low=round(lo, 4), close=round(new_close, 4),
                        amount=round(amount, 2), volume=round(volume, 2),
                        mktcap=round(new_close * shares, 2),
                        suspended=suspended, limit_up=limit_up,
                        limit_down=limit_down))
            prev_close = new_close

    return Market(cal, uni, pan, fund)
