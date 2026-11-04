# -*- coding: utf-8 -*-
# 滚动算子：每个窗口一个函数，供 build_registry 循环调用。

def _f_roc(w):

    R = []

    R.append(("ROC%d" % w,   (lambda w: lambda d: ref_shift(d["C"], w) / d["C"])(w)))

    return R

def _f_ma(w):

    R = []

    R.append(("MA%d" % w,    (lambda w: lambda d: mean(d["C"], w) / d["C"])(w)))

    return R

def _f_std(w):

    R = []

    R.append(("STD%d" % w,   (lambda w: lambda d: std(d["C"], w) / d["C"])(w)))

    return R

def _f_beta(w):

    R = []

    R.append(("BETA%d" % w,  (lambda w: lambda d: slope(d["C"], w) / d["C"])(w)))

    return R

def _f_rsqr(w):

    R = []

    R.append(("RSQR%d" % w,  (lambda w: lambda d: slope(d["C"], w, True)[1])(w)))

    return R

def _f_resi(w):

    R = []

    R.append(("RESI%d" % w,  (lambda w: lambda d: (d["C"] - mean(d["C"], w) - slope(d["C"], w) * (w - 1) / 2.0) / d["C"])(w)))


    return R

def _f_max(w):

    R = []

    R.append(("MAX%d" % w,   (lambda w: lambda d: tsmax(d["H"], w) / d["C"])(w)))

    return R

def _f_min(w):

    R = []

    R.append(("MIN%d" % w,   (lambda w: lambda d: tsmin(d["L"], w) / d["C"])(w)))

    return R

def _f_qtlu(w):

    R = []

    R.append(("QTLU%d" % w,  (lambda w: lambda d: quantile(d["C"], w, 0.8) / d["C"])(w)))

    return R

def _f_qtld(w):

    R = []

    R.append(("QTLD%d" % w,  (lambda w: lambda d: quantile(d["C"], w, 0.2) / d["C"])(w)))

    return R

def _f_rsv_(w):

    R = []

    R.append(("RSV_%d" % w, (lambda w: lambda d: (d["C"] - tsmin(d["L"], w)) / (tsmax(d["H"], w) - tsmin(d["L"], w) + 1e-12))(w)))

    return R

def _f_rsv(w):

    R = []

    R.append(("RSV%d" % w,   (lambda w: lambda d: (d["C"] - tsmin(d["L"], w)) / (tsmax(d["H"], w) - tsmin(d["L"], w) + 1e-12))(w)))

    return R

def _f_high(w):

    R = []

    R.append(("HIGH%d" % w,  (lambda w: lambda d: tsmax(d["H"], w) / d["C"])(w)))

    return R

def _f_lowpos(w):

    R = []

    R.append(("LOWPOS%d" % w,(lambda w: lambda d: (d["C"] - tsmin(d["L"], w)) / d["C"])(w)))

    return R

WIN_OPS_0 = [_f_roc, _f_ma, _f_std, _f_beta, _f_rsqr, _f_resi, _f_max, _f_min, _f_qtlu, _f_qtld, _f_rsv_, _f_rsv, _f_high, _f_lowpos]
