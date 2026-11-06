
# -*- coding: utf-8 -*-
"""etf_oos.py —— 换池子 / 留一法，验证 lowvol60 不是这 8 只 ETF 的偶然。"""
import os,sys,json,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
from audit_etf import load_panel,signal_lowvol
from audit_backtest import audit
dates,codes,P=load_panel();T,N=P["close"].shape

def sub_panel(keep):
    ks=("close","open_","amount")
    return {k:P[k][:,keep] for k in ks}

def verdict(keep,label):
    Q2=sub_panel(keep);S=signal_lowvol(Q2,60)
    sub=[codes[j] for j in keep]
    r=audit(S,Q2["close"],Q2["open_"],Q2["amount"],dates,sub,
            topk=min(2,len(keep)),cost_bp=5.0,
            signal_index=[(str(d),c) for d in dates for c in sub])
    print("[oos] %-22s n=%d verdict=%s fails=%s exec=%.4f lag=%s" %
          (label,len(keep),r["verdict"],r["fails"],r["executable_alpha_nav"],
           r["L2_alignment"]["nav_by_lag"]),flush=True)
    return r["executable_alpha_nav"]

# 留一法：每次去掉一只
for j in range(N):
    keep=[x for x in range(N) if x!=j]
    verdict(keep,"leave-out "+codes[j])
# 随机两半
rng=np.random.default_rng(42)
for i in range(4):
    ix=rng.permutation(N);verdict(list(ix[:4]),"random-half-%d"%i)
