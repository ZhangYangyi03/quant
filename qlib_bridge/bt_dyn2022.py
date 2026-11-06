
import os,numpy as np,pandas as pd,pickle
H=os.path.dirname(os.path.abspath(__file__));DD=os.path.join(H,"qlib_data")
cal=[l.strip() for l in open(os.path.join(DD,"calendars","day.txt"),encoding="utf-8") if l.strip()]
i0=next(i for i,d in enumerate(cal) if d>="2022-01-04")
i1=next(i for i,d in enumerate(cal) if d>"2022-12-30")
TD=cal[i0:i1];NT=len(TD)
c=pickle.load(open(os.path.join(H,"pred_dyn1000_2022.pkl"),"rb"))
pool=c["pool"];idx=c["index"]
s=pd.Series(np.asarray(c["pred"]).ravel(),index=idx,name="score")
s=s[~s.index.get_level_values("instrument").isin(["__bench__"])]
SIG=s.unstack("instrument").reindex(index=pd.to_datetime(TD),columns=pool).to_numpy()
SIG=np.nan_to_num(SIG,nan=-1e9)
print("[dyn] SIG",SIG.shape,"finite frac",round(float((SIG>-1e8).mean()),3),flush=True)
F=os.path.join(DD,"features")
P=np.full((NT,len(pool)),np.nan)
for j,cod in enumerate(pool):
    fp=os.path.join(F,cod,"close.day.bin")
    if not os.path.exists(fp):continue
    a=np.fromfile(fp,dtype="<f4");st=int(a[0]);v=a[1:].astype(np.float64)
    for k in range(NT):
        g=i0+k-st
        if 0<=g<len(v):P[k,j]=v[g]
R=np.full((NT-1,len(pool)),np.nan);ok=np.isfinite(P[:-1])&np.isfinite(P[1:])&(P[:-1]>0)
R[ok]=P[1:][ok]/P[:-1][ok]-1
print("[dyn] price finite frac",round(float(np.isfinite(P).mean()),3),"ret days",NT-1,flush=True)
def run(sc,topk=50,cost_bp=3.65):
    eq=[1.0];rets=[];turn=[]
    for k in range(1,NT):
        prev=np.argsort(sc[k-2])[-topk:] if k>=2 else np.argsort(sc[0])[-topk:]
        top=np.argsort(sc[k-1])[-topk:]
        r=np.nanmean(R[k-1,top]);r=0.0 if not np.isfinite(r) else r
        tv=len(set(top)^set(prev))/topk
        r-=cost_bp/1e4*tv
        rets.append(r);turn.append(tv);eq.append(eq[-1]*(1+r))
    return float(np.array(eq)[-1]),round(float(np.mean(turn)),3)
for nm,sc in (("REAL",SIG),):
    for cb in (0.0,3.65):
        n,t=run(sc,cost_bp=cb);print("[dyn] %s cost=%.2fbp nav %.4f turnover %.3f"%(nm,cb,n,t),flush=True)
rng=np.random.default_rng(7);SH=SIG.copy()
for k in range(len(SH)):rng.shuffle(SH[k])
print("[dyn] XS-SHUFFLE",run(SH)[0],flush=True)
