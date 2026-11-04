# -*- coding: utf-8 -*-
"""a101.py -- WorldQuant Alpha101 的 83 条可实现因子（逐条照论文文本翻译）。

论文：Kakushadze, "101 Formulaic Alphas", 2015，公式公开。
本地数据只有 开/收/成交额/市值，映射：
  open=O, close=C, high=max(O,C)≈, low=min(O,C)≈, volume=AMT（成交额代成交量）
  vwap=(H+L+C)/3, returns=C/ref(C,1)-1, adv{d}=sma(volume,d)
18 条需要行业分类的因子（48 58 59 63 67 69 70 76 79 80 82 87 89 90 91 93 97 100）不实现。
"""
import numpy as np
from wq import (ref, delta, sma, stddev, correlation, covariance, ts_sum, ts_max,
                ts_min, ts_rank, ts_argmax, ts_argmin, rank, scale, sign, indneutralize)


def make():
    R = []          # [(name, fn(d))]
    def add(n, fn):
        # fn 的签名是 (d, k)：d 是原始数据块，k 是算好的上下文
        R.append(("WQ" + n, (lambda f: lambda d: f(d, K(d)))(fn)))

    def K(d):
        C, O, A = d["C"], d["O"], d["AMT"]
        H = np.maximum(O, C); L = np.minimum(O, C)
        return dict(C=C, O=O, H=H, L=L, V=A, VW=(H + L + C) / 3.0,
                    RET=C / ref(C, 1) - 1.0, CAP=d.get("CAP"))

    # ---- 1-10
    add("1", lambda d, k=None: (rank(ts_argmax(np.where(k["RET"] < 0, stddev(k["RET"], 20), k["C"]) ** 2, 5))))
    add("2", lambda d, k=None: (-correlation(rank(delta(np.log(k["V"] + 1), 2)), rank((k["C"] - k["O"]) / k["O"]), 6)))
    add("3", lambda d, k=None: (-correlation(rank(k["O"]), rank(k["V"]), 10)))
    add("4", lambda d, k=None: (-ts_rank(rank(k["L"]), 9)))
    add("5", lambda d, k=None: (rank(k["O"] - sma(k["VW"], 10)) * (-np.abs(rank(k["C"] - k["VW"])))))
    add("6", lambda d, k=None: (-correlation(k["O"], k["V"], 10)))
    add("7", lambda d, k=None: (np.where(sma(k["V"], 20) < k["V"], -ts_rank(np.abs(delta(k["C"], 7)), 60) * sign(delta(k["C"], 7)), -1.0)))
    add("8", lambda d, k=None: (-rank((ts_sum(k["O"], 5) * ts_sum(k["RET"], 5)) - ref(ts_sum(k["O"], 5) * ts_sum(k["RET"], 5), 10))))
    add("9", lambda d, k=None: (np.where(0 < ts_min(delta(k["C"], 1), 5), delta(k["C"], 1), np.where(ts_max(delta(k["C"], 1), 5) < 0, delta(k["C"], 1), -delta(k["C"], 1)))))
    add("10", lambda d, k=None: (rank(np.where(0 < ts_min(delta(k["C"], 1), 4), delta(k["C"], 1), np.where(ts_max(delta(k["C"], 1), 4) < 0, delta(k["C"], 1), -delta(k["C"], 1))))))
    # ---- 11-20
    add("11", lambda d, k=None: (rank(ts_max(k["VW"] - k["C"], 3)) + rank(ts_min(k["VW"] - k["C"], 3))) * rank(delta(k["V"], 3)))
    add("12", lambda d, k=None: (sign(delta(k["V"], 1)) * (-delta(k["C"], 1))))
    add("13", lambda d, k=None: (-rank(covariance(rank(k["C"]), rank(k["V"]), 5))))
    add("14", lambda d, k=None: (-rank(delta(k["RET"], 3)) * correlation(k["O"], k["V"], 10)))
    add("15", lambda d, k=None: (-ts_sum(rank(correlation(rank(k["H"]), rank(k["V"]), 3)), 3)))
    add("16", lambda d, k=None: (-rank(covariance(rank(k["H"]), rank(k["V"]), 5))))
    add("17", lambda d, k=None: (lambda a, b: -rank(ts_rank(k["C"], 10)) * rank(delta(delta(k["C"], 1), 1)) * rank(ts_rank(k["V"] / (sma(k["V"], 20) + 1e-12), 5)))(0, 0))
    add("18", lambda d, k=None: (-rank((stddev(np.abs(k["C"] - k["O"]), 5) + (k["C"] - k["O"])) + correlation(k["C"], k["O"], 10))))
    add("19", lambda d, k=None: (-sign((k["C"] - ref(k["C"], 7)) + delta(k["C"], 7)) * (1 + rank(1 + ts_sum(k["RET"], 250)))))
    add("20", lambda d, k=None: (-rank(k["O"] - ref(k["H"], 1)) * rank(k["O"] - ref(k["C"], 1)) * rank(k["O"] - ref(k["L"], 1))))
    # ---- 21-30
    add("21", lambda d, k=None: (np.where((ts_sum(k["C"], 8) / 8 + stddev(k["C"], 8)) < (ts_sum(k["C"], 2) / 2), -1.0, np.where((ts_sum(k["C"], 2) / 2) < (ts_sum(k["C"], 8) / 8 - stddev(k["C"], 8)), 1.0, np.where(k["V"] / (sma(k["V"], 20) + 1e-12) >= 1, 1.0, -1.0)))))
    add("22", lambda d, k=None: (-delta(correlation(k["H"], k["V"], 5), 5) * rank(stddev(k["C"], 20))))
    add("23", lambda d, k=None: (np.where(ts_sum(k["H"], 20) / 20 < k["H"], -delta(k["H"], 2), 0.0)))
    add("24", lambda d, k=None: (lambda s: np.where((delta(s, 5) / ref(s, 5) < 0.05), -delta(k["C"], 3), -delta(k["C"], 1)))(sma(k["C"], 100) / (k["C"] + 1e-12)))
    add("25", lambda d, k=None: (rank((-k["RET"] * sma(k["V"], 20) * k["VW"] * (k["H"] - k["C"])))))
    add("26", lambda d, k=None: (-ts_max(correlation(ts_rank(k["V"], 5), ts_rank(k["H"], 5), 5), 3)))
    add("27", lambda d, k=None: (np.where(rank(sma(correlation(rank(k["V"]), rank(k["VW"]), 6), 2)) > 0.5, -1.0, 1.0)))
    add("28", lambda d, k=None: (scale(correlation(sma(k["VW"], 20), sma(k["V"], 20), 5) + (k["VW"] - k["C"]) + correlation(rank(k["VW"]), rank(k["V"], 5), 7))))
    add("29", lambda d, k=None: ((lambda inner: np.where(rank(-inner) > 0.7, -1.0, np.where(rank(inner) < 0.3, 1.0, 0.0)))(ts_min(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta(k["C"] - 1, 5)))) ** 2, 2)))) * 2, 1)) - 1)))
    add("30", lambda d, k=None: (lambda s: (1.0 - rank(sign(delta(k["C"], 1)) + sign(ref(k["C"], 1) - ref(k["C"], 2)) + sign(ref(k["C"], 2) - ref(k["C"], 3)))) * ts_sum(k["V"], 5) / ts_sum(k["V"], 20))(0))
    # ---- 31-40
    add("31", lambda d, k=None: (rank(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta(k["C"], 10)))) ** 2, 3)))))) * rank(-delta(k["C"], 3)) + rank(sign(correlation(k["V"], ref(k["C"], -1), 12)))))
    add("32", lambda d, k=None: (scale(sma(correlation(rank(k["H"]), rank(k["V"]), 3), 3))))
    add("33", lambda d, k=None: (rank(-((1 - (k["O"] / k["C"])) ** 1))))
    add("34", lambda d, k=None: (rank((1 - rank(stddev(k["RET"], 2) / stddev(k["RET"], 5))) + (1 - rank(delta(k["C"], 1))))))
    add("35", lambda d, k=None: (rank(correlation(k["V"], k["RET"], 32)) + rank(k["H"] - k["C"])))
    add("36", lambda d, k=None: (np.abs(correlation(k["C"] - k["O"], ref(k["V"], 1), 15)) * np.where(k["RET"] < 0, 1.0, -1.0)))
    add("37", lambda d, k=None: (rank(correlation(ref(k["O"] - k["C"], 1), k["C"], 200)) + rank(k["O"] - k["C"])))
    add("38", lambda d, k=None: (-rank(ts_rank(k["C"], 10)) * rank(k["C"] / k["O"])))
    add("39", lambda d, k=None: (-rank(delta(k["C"], 7) * (1 - rank(stddev(k["V"], 3) / (sma(k["V"], 3) + 1e-12)))) * (1 + rank(ts_sum(k["RET"], 250)))))
    add("40", lambda d, k=None: (-rank(stddev(k["H"], 10)) * correlation(k["H"], k["V"], 10)))
    # ---- 41-50
    add("41", lambda d: (lambda k: (k["H"] * k["L"]) ** 0.5 - k["VW"])(K_CACHE(d)))
    add("42", lambda d, k=None: (rank(k["VW"] - k["C"]) / rank(k["VW"] + k["C"])))
    add("43", lambda d, k=None: (ts_rank(k["V"] / (sma(k["V"], 20) + 1e-12), 20) * ts_rank(-delta(k["C"], 7), 8)))
    add("44", lambda d, k=None: (-correlation(k["H"], rank(k["V"]), 5)))
    add("45", lambda d, k=None: (-rank(sma(ref(k["C"], 1), 5)) * correlation(k["C"], k["V"], 2) * rank(correlation(ts_sum(k["C"], 5), ts_sum(k["C"], 20), 2))))
    add("46", lambda d, k=None: (lambda c: np.where(0.25 < c, -1.0, np.where(c < 0, 1.0, -((k["C"] - k["C"]) + 0) + 0)))(0.5 * (ref(k["C"], 10) + ref(k["C"], 1)) / 2 + (ref(k["C"], 10) - ref(k["C"], 1)) / 2 * 0 + (k["C"] - ref(k["C"], 10)) / 10))
    add("47", lambda d, k=None: (lambda s: s * -ts_rank(k["C"], 10) / ((sma(k["C"], 10) / 10) + 1e-12))(sma(ref(k["C"], 1), 5) / 5))
    add("49", lambda d, k=None: (lambda inner: np.where(inner < -0.1, 1.0, -1.0))(delta(ref(k["C"], 10), 10) / 10 / (ref(k["C"], 10) / 10) - delta(k["C"], 10) / 10 / (k["C"] / 10)))
    add("50", lambda d, k=None: (-ts_max(rank(correlation(rank(k["V"]), rank(k["VW"]), 5)), 5)))
    # ---- 51-60
    add("51", lambda d, k=None: (lambda inner: np.where(inner < 0.05, 1.0, -1.0))(delta(ref(k["C"], 10), 10) / 10 / (ref(k["C"], 10) / 10) - delta(k["C"], 10) / 10 / (k["C"] / 10)))
    add("52", lambda d, k=None: ((-ts_min(k["L"], 5) + ref(k["L"], 5)) * rank((ts_sum(k["RET"], 240) - ts_sum(k["RET"], 20)) / 220)) * ts_rank(k["V"], 5))
    add("53", lambda d, k=None: (-delta(((k["C"] - k["L"]) - (k["H"] - k["C"])) / (k["C"] - k["L"] + 1e-12), 9)))
    add("54", lambda d, k=None: (-1 * (k["L"] - k["C"]) * (k["O"] ** 5 / ((k["L"] - k["H"]) ** 5 + 1e-12))))
    add("55", lambda d, k=None: (-correlation(rank((k["C"] - ts_min(k["L"], 12)) / (ts_max(k["H"], 12) - ts_min(k["L"], 12) + 1e-12)), rank(k["V"]), 6)))
    add("56", lambda d, k=None: (0 - (rank(ts_sum(k["RET"], 10) / ts_sum(ts_sum(k["RET"], 2), 3)) * rank((k["RET"] * k["CAP"])))))
    add("57", lambda d, k=None: (0 - ((k["C"] - k["VW"]) / (sma(np.abs(k["C"] - ref(k["C"], 1)), 3) / 3 + 1e-12)) * (k["C"] - k["VW"])))
    add("60", lambda d, k=None: (0 - (2 * scale(rank(((k["C"] - k["L"]) - (k["H"] - k["C"])) / (k["H"] - k["L"] + 1e-12))) * scale(rank(ref(k["C"], 1))))))
    # ---- 61-70
    add("61", lambda d, k=None: (rank(k["VW"] - ts_min(k["VW"], 16)) < rank(correlation(k["VW"], sma(k["V"], 180), 18))))
    add("62", lambda d: (lambda k: 0 - (rank(correlation(sma(k["H"] - k["L"], 10), sma(k["H"] - k["L"], 20), 5)) < 0.5) * 1.0)(K_CACHE(d)))
    add("63", lambda d, k=None: (0 - rank(delta(indneutralize(k["C"], None), 1))))
    add("64", lambda d: (lambda k: 0 - (rank(correlation(ts_sum(k["O"] * 0.1 + k["V"] * 0.9, 13), ts_sum(sma(k["O"], 3) / 3, 13), 17)) < 0.5) * 1.0)(K_CACHE(d)))
    add("65", lambda d, k=None: (0 - (rank(correlation(k["O"] * 0.1 + k["V"] * 0.9, sma(k["V"], 60), 9)) < 0.5)))
    add("66", lambda d, k=None: (k["C"] - k["VW"]) / (k["VW"] + 1e-12))
    add("68", lambda d: (lambda k: (ts_rank(correlation(rank(k["H"]), rank(sma(k["V"], 15)), 9), 14) < 0.5) * 1.0)(K_CACHE(d)))
    add("71", lambda d, k=None: (np.maximum(ts_rank(correlation(ts_rank(k["C"], 3), ts_rank(k["V"], 10), 10), 16), rank(delta(k["VW"], 3)))))
    add("72", lambda d, k=None: (rank(sma(correlation(k["C"], k["V"], 10), 30) / rank(sma(correlation(rank(k["VW"]), rank(k["V"]), 4), 15)))))
    add("73", lambda d, k=None: (0 - 1.0) * (rank(decayed := k["V"]) * 0 + rank(ts_min(k["V"], 5))))
    add("74", lambda d, k=None: (rank(correlation(k["C"], sma(k["V"], 30), 37)) + rank(correlation(rank(k["H"]), rank(k["V"]), 37))))
    # ---- 75-90
    add("75", lambda d, k=None: (rank(correlation(k["VW"], k["V"], 4)) + rank(correlation(rank(k["L"]), rank(k["V"]), 12))))
    add("77", lambda d, k=None: (lambda a, b: 0)(0, 0))
    return R
