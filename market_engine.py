"""全市场回测引擎：5000+ 只股票、20 年，纯标准库（沙箱里 numpy 大数组会爆）。

与 ETF 版的差别：
  1. 股票池是"当日仍在交易的全部 A 股"，含已退市的（东财返回的就是全量），因此没有幸存者偏差；
     但 ST/停牌靠价格与成交量过滤，没有单独的 ST 标记表。
  2. 涨跌停按 ±10%（主板）/±20%（创业板 300/301、科创板 688）判定。
  3. 分向成本：佣金 2.5bp 最低 5 元 + 过户费 1bp + 卖出印花税 10bp。
  4. 容量：单票不超当日成交额 10%，不超组合的 X%。
"""
import csv, os, math, mmap

PX_OPEN, PX_CLOSE, PX_HIGH, PX_LOW, PX_VOL, PX_AMT = 0, 1, 2, 3, 4, 5


class StockBars:
    """列式存储 + 只读 mmap 访问，内存占用与股票规模无关。

    每个标的写成一个 .bin（float64 数值矩阵）加一个 .idx（日期表）。
    回测时按 (code, date) 定位是二分查找，O(log n)。
    """
    import struct as _struct

    MAGIC = b"MBAR0001"
    COLS = 7   # open, close, high, low, volume, amount, (pad)

    def __init__(self, store_dir):
        self.dir = store_dir
        self.date_bytes = set()
        self.meta = {}       # code -> (mmap, n, date_bytes)
        self.dates = []

    def _load_one(self, code):
        import mmap, struct
        base = os.path.join(self.dir, code)
        fp = open(base + ".bin", "rb")
        mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        n = (len(mm) - len(self.MAGIC)) // (8 * self.COLS)
        with open(base + ".idx", "rb") as f2:
            raw = f2.read()
        # 日期表：前 4 字节是小端长度，之后每 10 字节一个 YYYY-MM-DD
        cnt = struct.unpack("<I", raw[:4])[0]
        dbytes = raw[4:4 + cnt * 10]
        return mm, n, dbytes

    def load_dir(self, codes=None):
        """两种加载方式：

        快路（默认）：store/_all.idx 里所有股票的日期表已合并成一份，
        只 open 一次、只 mmap 一次，然后按记录切片。3000+ 只股票时
        这是唯一可行的方式——逐文件 open 会在几百次之后直接 WinError 8
        （每个文件句柄吃内核页表，进程地址空间耗尽）。

        慢路（_all.idx 不存在时）：老写法，逐文件 open。只适合几十只。
        """
        import struct
        allidx = os.path.join(self.dir, "_all.idx")
        self.meta = {}
        alld = set()
        if os.path.exists(allidx):
            fp = open(allidx, "rb")
            mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            fpos = 0
            nrec = struct.unpack_from("<I", mm, fpos)[0]; fpos += 4
            for _ in range(nrec):
                cnt, ln = struct.unpack_from("<IH", mm, fpos)
                fpos += 6
                code = bytes(mm[fpos:fpos + ln]).decode(); fpos += ln
                body = fpos; fpos += cnt * 10
                if codes and code not in codes: continue
                if cnt < 60: continue
                dbytes = memoryview(mm)[body:body + cnt * 10]
                self.meta[code] = (None, cnt, dbytes)
                alld.update(bytes(dbytes[i * 10:(i + 1) * 10]) for i in range(cnt))
            self._idx_mm = mm
            self._open_bin = {}
        else:
            files = [f[:-4] for f in os.listdir(self.dir) if f.endswith(".bin")]
            if codes: files = [c for c in files if c in codes]
            for code in files:
                mm2, n, dbytes = self._load_one(code)
                if n < 60: continue
                self.date_bytes.add(dbytes)
                self.meta[code] = (mm2, n, dbytes)
                alld.update(dbytes[i*10:(i+1)*10] for i in range(n))
            self._idx_mm = None
        self.dates = [b.decode() for b in sorted(alld)]
        return len(self.meta)

    def finalize(self):
        """把所有日期建成一张字节表，供 O(1) 查找（不再是每只股票一个 dict）。"""
        self.date_index = {d: i for i, d in enumerate(self.dates)}
        self.date_bytes = [d.encode() for d in self.dates]

    def nrows(self, code):
        m = self.meta.get(code)
        return m[1] if m else 0

    def row_i(self, code, i):
        """第 i 行（0 = 最早）。越界返回 None。"""
        m = self.meta.get(code)
        if not m or i < 0 or i >= m[1]: return None
        return self.row_at(code, i)

    def row_at(self, code, i):
        """一行 6 个数，下标固定用这三个常量，别再靠记：
           PX_OPEN=0  PX_CLOSE=1  PX_HIGH=2  PX_LOW=3  PX_VOL=4  PX_AMT=5"""
        import struct
        mm = self.meta[code][0]
        if mm is None:
            # 快路：日期表共用一份 mmap，价格数据按需读一个文件、读完即关，
            # 保证同一时刻打开的句柄不超过 1 个（逐文件 mmap 会耗尽地址空间）。
            base = os.path.join(self.dir, code)
            mm = getattr(self, "_cur", None)
            if mm is None or getattr(self, "_cur_code", None) != code:
                if mm is not None: mm.close()
                fp = open(base + ".bin", "rb")
                mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                fp.close()
                self._cur = mm; self._cur_code = code
        n = self.meta[code][1]
        if i < 0: i += n
        if i < 0 or i >= n: return None
        off = len(self.MAGIC) + i * 8 * self.COLS
        vals = struct.unpack_from("<7d", mm, off)
        return vals[:6]

    def bar(self, code, date):
        i = self._bisect(code, date)
        if i is None: return None
        return self.row_at(code, i)

    def _bisect(self, code, date):
        m = self.meta.get(code)
        if not m: return None
        dbytes = m[2]; n = m[1]; key = date.encode()
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cur = dbytes[mid*10:mid*10+10]
            if not isinstance(cur, (bytes, bytearray)): cur = bytes(cur)
            if cur == key: return mid
            if cur < key: lo = mid + 1
            else: hi = mid - 1
        return None

    def bisect_abs(self, code, date):
        return self._bisect(code, date)

    def row_index(self, code, date):
        return self._bisect(code, date)


def is_limit_up(bars, code, date, pct):
    b = bars.bar(code, date)
    if not b: return None
    j = bars.bisect_abs(code, date)
    if j is None or j == 0: return None
    prev_close = bars.row_i(code, j-1)[PX_CLOSE]
    up = round(prev_close*(1+pct), 2)
    return b[PX_OPEN] >= up - 0.005 and b[PX_HIGH] == b[PX_LOW]     # 开盘顶格且全天无振幅


def is_limit_down(bars, code, date, pct):
    b = bars.bar(code, date)
    if not b: return None
    j = bars.bisect_abs(code, date)
    if j is None or j == 0: return None
    prev_close = bars.row_i(code, j-1)[PX_CLOSE]
    down = round(prev_close*(1-pct), 2)
    return b[PX_OPEN] <= down + 0.005 and b[PX_HIGH] == b[PX_LOW]


def limit_pct(code):
    return 0.20 if code.startswith(("300", "301", "688")) else 0.10


class Acct:
    def __init__(self, cash, comm=0.00025, min_comm=5.0, transfer=0.00001, stamp=0.001, cap_pct=0.10):
        self.cash = cash
        self.comm, self.min_comm, self.transfer, self.stamp, self.cap_pct = comm, min_comm, transfer, stamp, cap_pct
        self.pos = {}          # code -> [qty, cost]
        self.equity = []
        self.fills = 0
        self.fees = 0.0
        self.trace = []          # [(date, code, side, qty, px, cash_after)]
        self.blocked = {"limit_up": 0, "limit_down": 0, "no_bar": 0, "capacity": 0, "cash": 0}

    def mv(self, last_price):
        """持仓市值 = 最后已知收盘价 × 数量。

        关键：退出 last_price 缓存不算 0 元 —— 那等于把持仓当作蒸发了，
        净值会凭空掉一块。停牌就按停牌前的最后价计（这正是现实中券商的做法）。"""
        return sum(q * last_price[c] for c, (q, _) in self.pos.items() if c in last_price)

    def sell(self, bars, code, date, qty, why=""):
        b = bars.bar(code, date)
        if not b: self.blocked["no_bar"] += 1; return 0
        if is_limit_down(bars, code, date, limit_pct(code)): self.blocked["limit_down"] += 1; return 0
        q = self.pos.get(code)
        if not q: return 0
        qty = min(qty, q[0])
        cap = int(b[PX_VOL]*self.cap_pct)
        if qty > cap: qty = cap; self.blocked["capacity"] += 1
        if qty <= 0: return 0
        amt = qty*b[PX_OPEN]        # t+1 开盘成交
        fee = max(amt*self.comm, self.min_comm) + amt*self.transfer + amt*self.stamp
        self.cash += amt - fee
        self.fees += fee
        self.fills += 1
        if len(self.trace) < 400: self.trace.append((date, code, "S", qty, b[PX_OPEN], round(self.cash)))
        q[0] -= qty
        if q[0] <= 0: self.pos.pop(code, None)
        return qty

    def buy(self, bars, code, date, amount, why="", b=None):
        if b is None: b = bars.bar(code, date)
        if not b or b[PX_OPEN] <= 0: self.blocked["no_bar"] += 1; return 0
        if is_limit_up(bars, code, date, limit_pct(code)): self.blocked["limit_up"] += 1; return 0
        if b[PX_VOL] <= 0: self.blocked["no_bar"] += 1; return 0
        px = b[PX_OPEN]
        qty = int(amount/px/100)*100
        cap = int(b[PX_VOL]*self.cap_pct/100)*100
        if qty > cap: qty = cap; self.blocked["capacity"] += 1
        while qty > 0:
            amt = qty*px
            fee = max(amt*self.comm, self.min_comm) + amt*self.transfer
            if amt + fee <= self.cash: break
            qty -= 100
        if qty <= 0:
            self.blocked["cash"] += 1
            return 0
        amt = qty*px
        fee = max(amt*self.comm, self.min_comm) + amt*self.transfer
        self.cash -= amt + fee
        self.fees += fee
        self.fills += 1
        if len(self.trace) < 400: self.trace.append((date, code, "B", qty, px, round(self.cash)))
        p = self.pos.setdefault(code, [0, 0.0])
        p[1] = (p[1]*p[0] + amt + fee)/(p[0] + qty)
        p[0] += qty
        return qty
