
import os,sys,json,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
from audit_etf import load_panel,signal_lowvol
from audit_backtest import audit,multi_year_verdict
dates,codes,P=load_panel();T,N=P["close"].shape
S=signal_lowvol(P,60)
reps=[]
for Y in range(2017,2027):
    idx=[i for i,d in enumerate(dates) if d[:4]==str(Y)]
    if len(idx)<100:continue
    a,b=idx[0],idx[-1]+1
    sl=slice(a,b)
    r=audit(S[sl],P["close"][sl],P["open_"][sl],P["amount"][sl],
            dates[a:b],codes,topk=2,cost_bp=5.0,
            signal_index=[(str(d),c) for d in dates[a:b] for c in codes])
    reps.append((Y,r))
    print("[etf-year]",Y,"days",b-a,"verdict",r["verdict"],"fails",r["fails"],
          "exec",r["executable_alpha_nav"],flush=True)
v=multi_year_verdict(reps)
print("[etf-verdict]",json.dumps(v,ensure_ascii=False)[:900],flush=True)
json.dump(v,open(os.path.join(Q,"etf_verdict.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
