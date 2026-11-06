
import os,sys,numpy as np,pandas as pd,pickle,json
Q=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Q)
from audit_backtest import audit
H=os.path.join(Q,"qlib_bridge");DD=os.path.join(H,"qlib_data")
cal_all=[l.strip() for l in open(os.path.join(DD,"calendars","day.txt"),encoding="utf-8") if l.strip()]
import os as _os
YEAR=int(_os.environ.get("DY_YEAR","2022"));i0=next(i for i,d in enumerate(cal_all) if d>=f"{YEAR}-01-01")
_c=[i for i,d in enumerate(cal_all) if d>f"{YEAR}-12-31"];i1=_c[0] if _c else len(cal_all)
TD=cal_all[i0:i1];NT=len(TD)
c=pickle.load(open(os.path.join(H,f"pred_dyn1000_{YEAR}.pkl"),"rb"));pool=c["pool"]
s=pd.Series(np.asarray(c["pred"]).ravel(),index=c["index"],name="score")
s=s[~s.index.get_level_values("instrument").isin(["__bench__"])]
SIG_FRAME=s.unstack("instrument").reindex(index=pd.to_datetime(TD),columns=pool)
SIG=SIG_FRAME.to_numpy()
# 显式索引：让 L1 能核对信号是否落在 (cal x codes) 网格上
EXPECTED_INDEX=[(str(d)[:10],c) for d in TD for c in pool]
F=os.path.join(DD,"features")
def load(fld):
    P=np.full((NT,len(pool)),np.nan)
    for j,cod in enumerate(pool):
        a=np.fromfile(os.path.join(F,cod,fld+".day.bin"),dtype="<f4");st=int(a[0]);v=a[1:].astype(np.float64)
        for k in range(NT):
            g=i0+k-st
            if 0<=g<len(v):P[k,j]=v[g]
    return P
C,O,A=load("close"),load("open"),load("amount")
r=audit(SIG,C,O,A,TD,pool,signal_index=EXPECTED_INDEX)
print(f"[audit] YEAR={YEAR} verdict={r['verdict']} fails={r['fails']} "
      f"lag_nav={r['L2_alignment']['nav_by_lag']} "
      f"c2c={r['L3_untradable']['c2c_nav']} c2c_f={r['L3_untradable']['c2c_filtered_nav']} "
      f"o2o_f={r['L3_untradable']['o2o_filtered_nav']} "
      f"exec_alpha={r['executable_alpha_nav']}",flush=True)
json.dump(r,open(os.path.join(Q,f"audit_{YEAR}.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
