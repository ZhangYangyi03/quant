# -*- coding: utf-8 -*-
"""fraw.py -- 读 fpanel.npz 的单个成员，返回 (需要先查 shape 可用 fraw.meta())"""
import os, struct, zipfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "fpanel.npz")


def meta():
    out = {}
    with zipfile.ZipFile(F) as Z:
        for zi in Z.infolist():
            with Z.open(zi.filename) as f:
                head = f.read(128)
            mv = head[6]
            hl = struct.unpack("<H", head[8:10])[0] if mv == 1 else struct.unpack("<I", head[8:12])[0]
            hs = 10 if mv == 1 else 12
            d = eval(head[hs:hs + hl].decode("latin1").strip(), {"False": False, "True": True})
            out[zi.filename[:-4]] = {"shape": d["shape"], "dtype": str(np.dtype(d["descr"])),
                                     "off": zi.header_offset, "size": zi.file_size}
    return out


def rows(name, i0=0, i1=None):
    """读 [i0,i1) 行。逐行小读，不整份 read：这里 f.read(>1MB) 直接 MemoryError。"""
    m = meta()[name]
    W = m["shape"][1]; dt = np.dtype(m["dtype"]); ib = dt.itemsize
    i1 = m["shape"][0] if i1 is None else i1
    out = np.zeros((i1 - i0, W), dtype=dt)
    with zipfile.ZipFile(F) as Z:
        zi = Z.getinfo(name + ".npy")
        with Z.open(zi) as f:
            head = f.read(128)
            mv = head[6]
            hl = struct.unpack("<H", head[8:10])[0] if mv == 1 else struct.unpack("<I", head[8:12])[0]
            hs = 10 if mv == 1 else 12
            base = hs + hl
            for r in range(i0, i1):
                f.seek(base + r * W * ib)
                row = np.frombuffer(f.read(W * ib), dtype=dt)
                np.copyto(out[r - i0], row)
    return out


def row(name, i):
    return rows(name, i, i + 1)[0]
