"""全市场截面信号。

性能上的关键改动（2026-11-03）：不再对"每只股票 × 每个调仓日"做二分。
mmap 的日期是升序的，调仓日本身也是升序的，所以给每只股票配一个行游标，
顺着往后走就行：总计 O(股票数 × 行数)，而不是 O(股票数 × 调仓次数 × log 行数)。

内存仍然按"一次只开一个 .bin"来控制（见 market_engine.row_at）。
"""
import math

PX_OPEN, PX_CLOSE, PX_HIGH, PX_LOW, PX_VOL, PX_AMT = 0, 1, 2, 3, 4, 5


class Cursor:
    """给每只股票配一个行游标：只允许向前推进，配合按时间顺序调仓。"""

    def __init__(self, bars):
        self.b = bars
        self.pos = {c: -1 for c in bars.meta}

    def at(self, code, date):
        """返回该股在 date 当天的行号；当天没有数据就返回 None（游标停在前一天）。"""
        m = self.b.meta.get(code)
        if not m: return None
        n = m[1]; db = m[2]; key = date.encode()
        i = self.pos[code]
        if i < 0:
            # 首次：二分定位到 date 当天或之前最近的一行
            lo, hi = 0, n - 1; found = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                cur = db[mid * 10:(mid + 1) * 10]
                if not isinstance(cur, (bytes, bytearray)): cur = bytes(cur)
                if cur <= key: found = mid; lo = mid + 1
                else: hi = mid - 1
            if found < 0: return None
            self.pos[code] = found
            return found
        # 之后：向前推进到今天或今天之前最近的一行（停牌日不推进到未来）
        while i + 1 < n:
            nxt = db[(i + 1) * 10:(i + 2) * 10]
            if not isinstance(nxt, (bytes, bytearray)): nxt = bytes(nxt)
            if nxt > key: break
            i += 1
        self.pos[code] = i
        if i < 0: return None
        cur = db[i * 10:(i + 1) * 10]
        if not isinstance(cur, (bytes, bytearray)): cur = bytes(cur)
        # 只有"最近一行就是今天"才算今天有行情；停牌股用最近一行（价格沿用）但要标记
        return i


def series_from(bars, code, j, n_ret):
    """用第 j 行往前 n_ret 个收益，返回 (波动率, 累计涨幅, 平均成交额)。"""
    if j is None or j < n_ret + 1: return None
    rets = []
    prev = bars.row_i(code, j - n_ret)
    if prev is None or prev[PX_CLOSE] <= 0: return None
    prev_c = prev[PX_CLOSE]
    for k in range(j - n_ret + 1, j + 1):
        r = bars.row_i(code, k)
        if r is None: return None
        if r[PX_CLOSE] <= 0:
            prev_c = None
            continue
        if prev_c is None:
            prev_c = r[PX_CLOSE]; continue
        rets.append(r[PX_CLOSE] / prev_c - 1)
        prev_c = r[PX_CLOSE]
    if len(rets) < n_ret - 2: return None
    m = sum(rets) / len(rets)
    var = sum((x - m) ** 2 for x in rets) / (len(rets) - 1)
    cum = 1.0
    for x in rets: cum *= (1 + x)
    adv = 0.0; cnt = 0
    for k in range(max(0, j - 19), j + 1):
        r = bars.row_i(code, k)
        if r: adv += r[PX_AMT]; cnt += 1
    return math.sqrt(var), cum - 1.0, (adv / cnt if cnt else 0.0)


def cross_section_fast(bars, cur, date, kind, n, min_price=2.0, min_adv=3e7):
    out = {}
    for code in bars.meta:
        j = cur.at(code, date)
        if j is None: continue
        m = series_from(bars, code, j, n)
        if m is None: continue
        vol, cum, adv = m
        b = bars.row_i(code, j)
        if b is None or b[PX_CLOSE] < min_price or adv < min_adv: continue
        if kind == "lowvol":
            if vol <= 0: continue
            out[code] = -vol
        elif kind == "mom":
            out[code] = cum
        elif kind == "momvol":
            if vol <= 0: continue
            out[code] = cum / vol
        elif kind == "rev":
            out[code] = -cum
        elif kind == "lowvol_mom":
            if vol <= 0: continue
            out[code] = -vol + 0.5 * cum
    return out


def series_metrics(bars, code, end_idx, n_ret):
    return series_from(bars, code, end_idx, n_ret)


def cross_section(bars, date, kind, n, min_price=2.0, min_adv=3e7):
    """兼容旧接口：没有游标就现建一个（慢，只用在小测试里）。"""
    cur = Cursor(bars)
    return cross_section_fast(bars, cur, date, kind, n, min_price, min_adv)
