
import os,sys,csv,json,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
from audit_etf import load_panel,signal_lowvol
dates,codes,P=load_panel();T,N=P["close"].shape
S=signal_lowvol(P,60)
C,O,A=P["close"],P["open_"],P["amount"]
ROO=np.full((T-1,N),np.nan);ok=np.isfinite(O[:-1])&np.isfinite(O[1:])&(O[:-1]>0);ROO[ok]=O[1:][ok]/O[:-1][ok]-1
GAP=np.full((T,N),np.nan);ok=np.isfinite(C[:-1])&np.isfinite(O[1:])&(C[:-1]>0);GAP[1:][ok]=O[1:][ok]/C[:-1][ok]-1
GAP[(~np.isfinite(A))|(A<=0)]=np.nan
s=np.where(np.isfinite(S),S,-1e9)
holds={};eq=[1.0];prev=None
for k in range(1,T-1):
    sc=s[k-1].copy();bad=~np.isfinite(GAP[k+1])|(GAP[k+1]>0.095);sc[bad]=-1e9
    top=np.argsort(sc)[-2:]
    for j in top:holds[codes[j]]=holds.get(codes[j],0)+1
    r=np.nanmean(ROO[k,top]);r=0 if not np.isfinite(r) else r
    eq.append(eq[-1]*(1+r))
print("[hold] strategy nav",round(float(eq[-1]),4))
print("[hold] days held",dict(sorted(holds.items(),key=lambda x:-x[1])))
# 基准：纯持有 511010、等权 8 只、纯持有 510300
for nm,j in (("511010国债",codes.index("511010.SH")),("510300沪深300",codes.index("510300.SH"))):
    x=np.nan_to_num(ROO[:,j],nan=0.0)
    print("[bench]",nm,"nav",round(float(np.prod(1+x)),4))
allr=np.nanmean(np.nan_to_num(ROO,nan=0.0),axis=1)
print("[bench] 8ETF等权 nav",round(float(np.prod(1+allr)),4))
# 逐年
import collections
rets=np.array(eq);rets=np.diff(rets)/rets[:-1]
yr=collections.defaultdict(list)
for i,d in enumerate(dates[1:len(rets)+1]):yr[d[:4]].append(rets[i])
for y in sorted(yr):
    v=np.array(yr[y]);print("[year]",y,"n",len(v),"ret",round(float((1+v).prod()*100-100),2),"%")
