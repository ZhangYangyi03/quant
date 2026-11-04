"""compact.py -- 把 4 份 float64 面板压成一份 float32，且提供"只读一行"的访问。

为什么要这个
------------
这台机器（2026-11-04 实测）给子进程的提交上限是 512MB（Windows Job Object，
LimitFlags 0x270c，ProcessMemoryLimit=512MB），而且剩余的提交额度被切碎了：
numpy 一次要 >800KB 的**连续**内存就会 MemoryError，哪怕离上限还有几百 MB。
m_close.bin 等 4 份面板各 370MB、float64、按【股票 x 日期】存，一次 mmap 就 370MB。
这就是以前那些会话"算到一半突然没了"的机械原因，不是模型也不是网络。

做法
----
m_compact.bin：float32、按【日期 x 股票】存，四个字段（close / amt / open / mktcap）
在同一个文件里。370MB -> 185MB/字段，而且某一天就是连续的 21KB：
    p.col("close", j)        -> (NC,) float32     只读 21KB，不 map 整个文件
    p.window("close", a, b)  -> (b-a, NC) float32 一次连续读
不提供整字段映射：185MB 的映射在这台机器上照样可能失败。

用法：
    python compact.py         # 生成 m_compact.bin（一次性，约 740MB 落盘）
    from compact import Panel
"""
import os, json, struct, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "m_compact.bin")
MAGIC = b"CMPC0001"
FIELDS = ["close", "amt", "open", "mktcap"]
SRC = {"close": "m_close.bin", "amt": "amt_flat.bin",
       "open": "m_open.bin", "mktcap": "m_mktcap.bin"}
CHUNK_D = 200


def meta():
    m = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    return m["dates"], m["codes"]


def build():
    dates, codes = meta()
    ND, NC = len(dates), len(codes)
    n = ND * NC
    t0 = time.time()
    with open(OUT, "wb") as out:
        out.write(MAGIC + struct.pack("<II", ND, NC))
        for f in FIELDS:
            p = os.path.join(HERE, SRC[f])
            mm = np.memmap(p, dtype=np.float64, mode="r")
            flat = mm[2:2 + n].reshape(NC, ND)       # 源: 股票 x 日期，跳过 16 字节头
            for d0 in range(0, ND, CHUNK_D):
                d1 = min(ND, d0 + CHUNK_D)
                blk = np.ascontiguousarray(flat[:, d0:d1].T, dtype=np.float32)
                out.write(blk.tobytes())
                del blk
            del flat, mm
            print("[%6.1fs] %-7s done" % (time.time() - t0, f), flush=True)
    print("done %s  %.0f MB" % (OUT, os.path.getsize(OUT) / 1e6))


def _alloc(shape):
    """按块分配：这块机器上 numpy 单次请求超过约 800KB 的连续内存就会失败，
    所以大数组一律"先 mmap 一块匿名内存、再 frombuffer"，不经过 malloc。
    返回 (mmap 句柄, ndarray)；句柄要一直持有，否则内存会被回收。"""
    import mmap
    nbytes = int(np.prod(shape)) * 4
    if nbytes <= 512 * 1024:
        return None, np.empty(shape, dtype=np.float32)
    mm = mmap.mmap(-1, nbytes)
    arr = np.frombuffer(mm, dtype=np.float32, count=int(np.prod(shape))).reshape(shape)
    return mm, arr


_MLOCK = []


def load_rows(path, field, j0, j1, chunk_rows=16, cache_key=None):
    """读 [j0, j1) x 全部股票，返回 (j1-j0, NC) float32。

    两次绕开这台机器的内存限制：
      1. 目标块用 mmap 分配（不走 malloc），大块也能一次要到；
      2. 每次只 f.read() 16 行（16 x 5251 x 4 = 336KB），填进目标块。
         一次 read 超过约 1MB 就会 MemoryError —— 这是本机 512MB 作业上限
         留下的碎片问题，不是磁盘的问题。
    可选 cache_key=["close"] 之类：同一字段同一区间只读一次。
    """
    global _CACHE
    key = (cache_key, field, j0, j1) if cache_key is not None else None
    if key is not None and key in _CACHE:
        return _CACHE[key]
    import mmap
    nrows = j1 - j0
    NC_ = Panel_NC(path)
    n = NC_
    mm = mmap.mmap(-1, nrows * NC_ * 4)
    _MLOCK.append(mm)
    arr = np.frombuffer(mm, dtype=np.float32, count=nrows * NC_).reshape(nrows, NC_)
    with open(path, "rb", buffering=0) as f:
        f.seek(16 + FIELDS.index(field) * Panel_ND(path) * NC_ * 4 + j0 * NC_ * 4)
        done = 0
        while done < nrows:
            k = min(chunk_rows, nrows - done)
            raw = f.read(k * NC_ * 4)
            arr[done:done + k] = np.frombuffer(raw, dtype=np.float32).reshape(k, NC_)
            done += k
    if key is not None:
        _CACHE[key] = arr
    return arr


_CACHE = {}
_HEAD = {}


def Panel_NC(path):
    if path not in _HEAD:
        with open(path, "rb") as f:
            head = f.read(16)
        assert head[:8] == MAGIC, "not a compact panel: %r" % head[:8]
        _HEAD[path] = struct.unpack("<II", head[8:16])
    return _HEAD[path][1]


def Panel_ND(path):
    if path not in _HEAD:
        with open(path, "rb") as f:
            head = f.read(16)
        assert head[:8] == MAGIC, "not a compact panel: %r" % head[:8]
        _HEAD[path] = struct.unpack("<II", head[8:16])
    return _HEAD[path][0]


class Panel:
    def __init__(self, path=OUT):
        self.path = path
        with open(path, "rb") as f:
            head = f.read(16)
        assert head[:8] == MAGIC, "not a compact panel: %r" % head[:8]
        self.ND, self.NC = struct.unpack("<II", head[8:16])
        self.dates, self.codes = meta()

    def window(self, field, j0, j1):
        return load_rows(self.path, field, j0, j1, cache_key=self.path)

    def col(self, field, j):
        return self.window(field, j, j + 1)[0]

    def drop_cache(self):
        _CACHE.clear()


if __name__ == "__main__":
    build()
