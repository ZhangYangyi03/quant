"""信号层：每个信号只做一件事——给定历史，吐出 t 日收盘时的横截面排序。

规则：任何信号只能读 date 及以前的数据。engine 会把它变成 t+1 开盘的委托。
"""
import math

def _ret(closes, n):
    if len(closes) <= n: return None
    past = closes[-1-n]
    if not past or past <= 0: return None
    return closes[-1]/past - 1

def momentum(n=20):
    def f(hist, codes, date):
        out = {}
        for c in codes:
            r = _ret(hist[c], n)
            if r is not None: out[c] = r
        return out
    return f

def ma_residual(n1=20, n2=60):
    """相对均线偏离之差：(P/MA_n1 - 1) - (P/MA_n2 - 1)，剔除单边趋势里的水平效应。"""
    def f(hist, codes, date):
        out = {}
        for c in codes:
            s = hist[c]
            if len(s) < n2 or not s[-1]: continue
            m1 = sum(s[-n1:])/n1; m2 = sum(s[-n2:])/n2
            if not m1 or not m2: continue
            out[c] = (s[-1]/m1 - 1) - (s[-1]/m2 - 1)
        return out
    return f

def low_vol(n=60):
    def f(hist, codes, date):
        out = {}
        for c in codes:
            s = hist[c]
            if len(s) < n+1: continue
            rets = [s[j]/s[j-1]-1 for j in range(len(s)-n, len(s)) if s[j-1] > 0]
            if len(rets) < n-2: continue
            m = sum(rets)/len(rets)
            var = sum((r-m)**2 for r in rets)/(len(rets)-1)
            out[c] = -math.sqrt(var)
        return out
    return f

def momentum_lowvol(n=20):
    """动量除以波动：单位风险的动量，避免选出高波垃圾。"""
    def f(hist, codes, date):
        out = {}
        for c in codes:
            s = hist[c]
            if len(s) < max(n, 20)+1: continue
            r = _ret(s, n)
            if r is None: continue
            rets = [s[j]/s[j-1]-1 for j in range(len(s)-20, len(s)) if s[j-1] > 0]
            if len(rets) < 18: continue
            m = sum(rets)/len(rets)
            vol = math.sqrt(sum((x-m)**2 for x in rets)/(len(rets)-1))
            if vol <= 0: continue
            out[c] = r/vol
        return out
    return f

def top_k(signal, k, positive_only=False):
    """把横截面分数变成等权目标持仓。"""
    def f(hist, codes, date):
        sc = signal(hist, codes, date)
        items = sorted(sc.items(), key=lambda x: -x[1])
        if positive_only: items = [x for x in items if x[1] > 0]
        return [c for c, _ in items[:k]]
    return f
