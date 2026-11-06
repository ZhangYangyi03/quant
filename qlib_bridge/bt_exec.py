
import os,numpy as np,pandas as pd,pickle
H=os.path.dirname(os.path.abspath(__file__));DD=os.path.join(H,"qlib_data")
cal=[l.strip() for l in open(os.path.join(DD,"calendars","day.txt"),encoding="utf-8") if l.strip()]
YEAR=int(os.environ["DY_YEAR"])
i0=next(i for i,d in enumerate(cal) if d>=f"{YEAR}-01-01")
_c=[i for i,d in enumerate(cal) if d>f"{YEAR}-12-31"];i1=_c[0] if _c else len(cal)
TD=cal[i0:i1];NT=len(TD)
c=pickle.load(open(os.path.join(H,f"pred_dyn1000_{YEAR}.pkl"),"rb"))
pool=c["pool"];s=pd.Series(np.asarray(c["pred"]).ravel(),index=c["index"],name="score")
s=s[~s.index.get_level_values("instrument").isin(["__bench__"])]
SIG=np.nan_to_num(s.unstack("instrument").reindex(index=pd.to_datetime(TD),columns=pool).to_numpy(),nan=-1e9)
F=os.path.join(DD,"features")
def load(fld):
    P=np.full((NT,len(pool)),np.nan)
    for j,cod in enumerate(pool):
        fp=os.path.join(F,cod,fld+".day.bin")
        if not os.path.exists(fp):continue
        a=np.fromfile(fp,dtype="<f4");st=int(a[0]);v=a[1:].astype(np.float64)
        for k in range(NT):
            g=i0+k-st
            if 0<=g<len(v):P[k,j]=v[g]
    return P
C=load("close");O=load("open")
# 次日 open 买入、再次日 open 卖出 的持仓收益：用 open[k+1] -> open[k+2]
# 但按日频调仓，简化为：d-1 收盘出信号，d 开盘买入，d+1 开盘卖出收益 = O[d+1]/O[d]-1
ROO=np.full((NT-1,len(pool)),np.nan)
ok=np.isfinite(O[:-1])&np.isfinite(O[1:])&(O[:-1]>0)
ROO[ok]=O[1:][ok]/O[:-1][ok]-1
# 收盘到收盘（原口径）
RCC=np.full((NT-1,len(pool)),np.nan)
ok=np.isfinite(C[:-1])&np.isfinite(C[1:])&(C[:-1]>0)
RCC[ok]=C[1:][ok]/C[:-1][ok]-1
# 次日开盘跳空：O[d]/C[d-1]-1，用来判定一字涨停不可买（跳空>9.5%）与停牌
GAP=np.full((NT,len(pool)),np.nan)
ok=np.isfinite(C[:-1])&np.isfinite(O[1:])&(C[:-1]>0)
GAP[1:][ok]=O[1:][ok]/C[:-1][ok]-1
def run(ret,lag=1,cost_bp=3.65,filter_limit=False):
    eq=[1.0];prev=None;blocked=0
    for k in range(1,NT-1):
        si=max(0,k-lag)
        sc=SIG[si].copy()
        if filter_limit:
            g=GAP[k]
            sc[(g>0.095)|(np.isnan(g))]=-1e9   # 一字涨停/无开盘价 不可买
        top=np.argsort(sc)[-50:]
        if filter_limit:blocked+=int(((GAP[k,top]>0.095)|np.isnan(GAP[k,top])).sum())
        r=np.nanmean(ret[k,top]);r=0.0 if not np.isfinite(r) else r
        tv=1.0 if prev is None else len(set(top)^set(prev))/50
        r-=cost_bp/1e4*tv;prev=top
        eq.append(eq[-1]*(1+r))
    return round(float(eq[-1]),4),blocked
for nm,ret in (("C2C",RCC),("O2O",ROO)):
    for fl in (False,True):
        n,b=run(ret,filter_limit=fl)
        print(f"[exec] YEAR={YEAR} {nm} filter_limit={fl} nav {n:.4f} blocked {b}",flush=True)
