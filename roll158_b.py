# -*- coding: utf-8 -*-
# 滚动算子：每个窗口一个函数，供 build_registry 循环调用。

def _f_vwapdev(w):

    R = []

    R.append(("VWAPDEV%d" % w,(lambda w: lambda d: (d["C"] - d["VWAP"]) / (d["VWAP"] + 1e-12))(w)))

    return R

def _f_imax(w):

    R = []

    R.append(("IMAX%d" % w,  (lambda w: lambda d: idxmax(d["H"], w))(w)))

    return R

def _f_imin(w):

    R = []

    R.append(("IMIN%d" % w,  (lambda w: lambda d: idxmin(d["L"], w))(w)))

    return R

def _f_imxd(w):

    R = []

    R.append(("IMXD%d" % w,  (lambda w: lambda d: idxmax(d["H"], w) - idxmin(d["L"], w))(w)))

    return R

def _f_corr(w):

    R = []

    R.append(("CORR%d" % w,  (lambda w: lambda d: corr(d["C"], np.log(d["AMT"] + 1), w))(w)))

    return R

def _f_cord(w):

    R = []

    R.append(("CORD%d" % w,  (lambda w: lambda d: corr(d["C"] / ref_shift(d["C"], 1), np.log(d["AMT"] / ref_shift(d["AMT"], 1) + 1), w))(w)))

    return R

def _f_cntp(w):

    R = []

    R.append(("CNTP%d" % w,  (lambda w: lambda d: mean((d["C"] > ref_shift(d["C"], 1)).astype(np.float64), w))(w)))

    return R

def _f_cntn(w):

    R = []

    R.append(("CNTN%d" % w,  (lambda w: lambda d: mean((d["C"] < ref_shift(d["C"], 1)).astype(np.float64), w))(w)))

    return R

def _f_cntd(w):

    R = []

    R.append(("CNTD%d" % w,  (lambda w: lambda d: mean((d["C"] > ref_shift(d["C"], 1)).astype(np.float64), w) - mean((d["C"] < ref_shift(d["C"], 1)).astype(np.float64), w))(w)))

    return R

def _f_sump(w):

    R = []

    R.append(("SUMP%d" % w,  (lambda w: lambda d: _sumpos(d["C"] - ref_shift(d["C"], 1), w) / (_sumabs(d["C"] - ref_shift(d["C"], 1), w) + 1e-12))(w)))

    return R

def _f_sumn(w):

    R = []

    R.append(("SUMN%d" % w,  (lambda w: lambda d: _sumpos(ref_shift(d["C"], 1) - d["C"], w) / (_sumabs(d["C"] - ref_shift(d["C"], 1), w) + 1e-12))(w)))

    return R

def _f_sumd(w):

    R = []

    R.append(("SUMD%d" % w,  (lambda w: lambda d: (_sumpos(d["C"] - ref_shift(d["C"], 1), w) - _sumpos(ref_shift(d["C"], 1) - d["C"], w)) / (_sumabs(d["C"] - ref_shift(d["C"], 1), w) + 1e-12))(w)))

    return R

def _f_vma(w):

    R = []

    R.append(("VMA%d" % w,   (lambda w: lambda d: mean(d["AMT"], w) / (d["AMT"] + 1e-12))(w)))

    return R

def _f_vstd(w):

    R = []

    R.append(("VSTD%d" % w,  (lambda w: lambda d: std(d["AMT"], w) / (d["AMT"] + 1e-12))(w)))

    return R






WIN_OPS_1 = [_f_vwapdev, _f_imax, _f_imin, _f_imxd, _f_corr, _f_cord, _f_cntp, _f_cntn, _f_cntd, _f_sump, _f_sumn, _f_sumd, _f_vma, _f_vstd]
