
# -*- coding: utf-8 -*-
"""etf_tail.py —— 极端日剔除后 lowvol60 是否还成立。"""
import os,sys,json,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
from audit_etf import load_panel,signal_lowvol
dates,codes,P=load_panel();T,N=P["close"].shape
S=signal_lowvol(P,60);C,O,A=P["close"],P["open_"],P["amount"]
ROO=np.full((T-1,N),np.nan);ok=np.isfinite(O[:-1])&np.isfinite(O[1:])&(O[:-1]>0);ROO[ok]=O[1:][ok]/O[:-1][ok]-1
GAP=np.full((T,N),np.nan);ok=np.isfinite(C[:-1])&np.isfinite(O[1:])&(C[:-1]>0);GAP[1:][ok]=O[1:][ok]/C[:-1][ok]-1
GAP[(~np.isfinite(A))|(A<=0)]=np.nan
s=np.where(np.isfinite(S),S,-1e9)
def run(rebal=20,cost=5.0,winsor=None):
    eq=[1.0];cur=None;prev=None;days=[]
    for k in range(1,T-1):
        if cur is None or (k-1)%rebal==0:
            sc=s[k-1].copy();bad=~np.isfinite(GAP[k+1])|(GAP[k+1]>0.095);sc[bad]=-1e9
            cur=np.argsort(sc)[-2:]
        r=np.nanmean(ROO[k,cur]);r=0 if not np.isfinite(r) else r
        if winsor is not None:r=float(np.clip(r,-winsor,winsor))
        if prev is not None and not np.array_equal(cur,prev):r-=cost/1e4*len(set(cur)^set(prev))/2
        prev=cur;eq.append(eq[-1]*(1+r));days.append((dates[k],r,"+".join(codes[x] for x in cur)))
    x=np.diff(eq)/np.array(eq[:-1])
    sh=float(x.mean()/x.std(ddof=1)*np.sqrt(244))
    return float(eq[-1]),sh,days
for w in (None,0.05,0.03,0.02,0.01):
    navv,sh,days=run(winsor=w)
    print("[tail] winsor",w,"nav",round(navv,4),"sharpe",round(sh,3),flush=True)
navv,sh,days=run()
top=sorted(days,key=lambda x:-x[1])[:8]
print("[tail] top days:")
for d,r,h in top:print("  ",d,round(r*100,2),"%",h)
# 2024-10-08 附近持仓
for d,r,h in days:
    if "2024-09-25"<=d<="2024-10-15":print("[oct]",d,round(r*100,2),"%",h)
