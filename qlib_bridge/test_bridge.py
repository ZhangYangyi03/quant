# -*- coding: utf-8 -*-
"""test_bridge.py —— qlib 桥接层的验收测试，纯标准库，不需要 pytest。
跑法：.qlibenv/Scripts/python.exe -u test_bridge.py
"""
import sys
import os, json, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "qlib_data")


def test_binary_layout():
    """qlib feature 文件：float32，第 0 个元素是 start_index，文件名不带 $。"""
    meta = json.load(open(os.path.join(ROOT, "px_index_meta.json"), encoding="utf-8"))
    ND, NC = len(meta["dates"]), len(meta["codes"])
    f = os.path.join(DATA, "features", "000001", "close.day.bin")
    a = np.fromfile(f, dtype="<f4")
    assert a[0] >= 0 and a[0] < ND
    assert len(a) - 1 == ND - int(a[0])
    assert not os.path.exists(os.path.join(DATA, "features", "000001", "$close.day.bin"))


def test_calendar_and_instruments():
    cal = open(os.path.join(DATA, "calendars", "day.txt"), encoding="utf-8").read().splitlines()
    meta = json.load(open(os.path.join(ROOT, "px_index_meta.json"), encoding="utf-8"))
    assert cal[:len(meta["dates"])] == meta["dates"]
    inst = [l.split("\t") for l in open(os.path.join(DATA, "instruments", "all.txt"),
                                        encoding="utf-8").read().splitlines() if "\t" in l]
    assert len(inst) >= len(meta["codes"])
    assert any(r[0] == "__bench__" for r in inst)


def test_corrupt_days_are_nan():
    """审计接缝：corrupt/halt 日的 close 必须是 NaN，不能是 0。"""
    meta = json.load(open(os.path.join(ROOT, "px_index_meta.json"), encoding="utf-8"))
    dates, codes = meta["dates"], meta["codes"]
    corrupt = np.memmap(os.path.join(ROOT, "corrupt_mask.bin"), dtype=np.uint8,
                        mode="r", shape=(len(dates), len(codes)))
    checked = 0
    for k in range(min(50, len(codes))):
        js = np.flatnonzero(corrupt[:, k])[:2]
        for j in js:
            f = os.path.join(DATA, "features", codes[k].lower(), "close.day.bin")
            a = np.fromfile(f, dtype="<f4")
            start = int(a[0])
            v = a[1 + int(j) - start]
            assert np.isnan(v), "corrupt day must be NaN, got %s" % v
            checked += 1
    assert checked > 0


def test_reconcile_report():
    r = json.load(open(os.path.join(HERE, "reconcile_report.json"), encoding="utf-8"))
    assert r["price_check"]["ok"] is True
    assert r["price_check"]["max_abs_err"] == 0.0
    assert r["quality"]["ok"] is True
    assert r["curve"]["qlib_days"] > 500


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("PASS", fn.__name__, flush=True)
    print("ALL PASS", len(fns), flush=True)
