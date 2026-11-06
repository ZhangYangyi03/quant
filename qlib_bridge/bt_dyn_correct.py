
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
F=os.path.join(DD,"features");P=np.full((NT,len(pool)),np.nan)
for j,cod in enumerate(pool):
    a=np.fromfile(os.path.join(F,cod,"close.day.bin"),dtype="<f4");st=int(a[0]);v=a[1:].astype(np.float64)
    for k in range(NT):
        g=i0+k-st
        if 0<=g<len(v):P[k,j]=v[g]
R=np.full((NT-1,len(pool)),np.nan);ok=np.isfinite(P[:-1])&np.isfinite(P[1:])&(P[:-1]>0)
R[ok]=P[1:][ok]/P[:-1][ok]-1
def run(sc,lag,cost_bp=3.65):
    eq=[1.0];prev=None
    for k in range(NT-1):
        si=max(0,k-lag);top=np.argsort(sc[si])[-50:]
        r=np.nanmean(R[k,top]);r=0.0 if not np.isfinite(r) else r
        tv=1.0 if prev is None else len(set(top)^set(prev))/50
        r-=cost_bp/1e4*tv;prev=top
        eq.append(eq[-1]*(1+r))
    return float(eq[-1])
rng=np.random.default_rng(7);SH=SIG.copy()
for k in range(len(SH)):rng.shuffle(SH[k])
for lag in (1,2):
    print(f"[corr] YEAR={YEAR} LAG={lag} REAL nav {run(SIG,lag):.4f} XS-SHUFFLE nav {run(SH,lag):.4f}",flush=True)
