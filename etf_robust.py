
# -*- coding: utf-8 -*-
"""etf_robust.py —— lowvol60 的三重压力测试：成本、调仓周期、多重检验。"""
import os,sys,json,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
from audit_etf import load_panel,signal_lowvol,signal_momentum
from audit_backtest import audit

dates,codes,P=load_panel();T,N=P["close"].shape
S=signal_lowvol(P,60)
C,O,A=P["close"],P["open_"],P["amount"]
RCC=np.full((T-1,N),np.nan);ok=np.isfinite(C[:-1])&np.isfinite(C[1:])&(C[:-1]>0);RCC[ok]=C[1:][ok]/C[:-1][ok]-1
ROO=np.full((T-1,N),np.nan);ok=np.isfinite(O[:-1])&np.isfinite(O[1:])&(O[:-1]>0);ROO[ok]=O[1:][ok]/O[:-1][ok]-1
GAP=np.full((T,N),np.nan);ok=np.isfinite(C[:-1])&np.isfinite(O[1:])&(C[:-1]>0);GAP[1:][ok]=O[1:][ok]/C[:-1][ok]-1
GAP[(~np.isfinite(A))|(A<=0)]=np.nan

def nav(sig,ret,cost_bp,rebal=1,block=True,lag=1):
    s=np.where(np.isfinite(sig),sig,-1e9)
    eq=[1.0];cur=None;prev=None;turn=0
    for k in range(1,T-1):
        # 每 rebal 日才换仓
        if cur is None or (k-1)%rebal==0:
            sc=s[k-lag].copy()
            if block:
                bad=~np.isfinite(GAP[k+1])|(GAP[k+1]>0.095);sc[bad]=-1e9
            cur=np.argsort(sc)[-2:]
        r=np.nanmean(ret[k,cur]);r=0 if not np.isfinite(r) else r
        if prev is not None and not np.array_equal(cur,prev):
            tv=len(set(cur)^set(prev))/2;r-=cost_bp/1e4*tv;turn+=tv
        prev=cur
        eq.append(eq[-1]*(1+r))
    return round(float(eq[-1]),4),round(turn,1)

print("[cost] O2O + block, lag=1")
for cb in (0,5,10,20,30,50):
    print("  cost_bp=%2d daily-rebal %.4f  20d-rebal %.4f"%(cb,nav(S,ROO,cb,1)[0],nav(S,ROO,cb,20)[0]),flush=True)
print("[rebal] cost=5bp O2O")
for rb in (1,5,10,20,60):
    print("  rebal=%2d nav %.4f turnover %.1f"%(rb,*nav(S,ROO,5,rb)),flush=True)
print("[benchmarks] cost=5bp rebal=20")
for nm,ret in (("C2C",RCC),("O2O",ROO)):
    print("  ",nm,"nav",nav(S,ret,5,20)[0],flush=True)
# 10 个配置的 PBO（用 run_backtest 的 10 组信号，日频净值）
configs={"lowvol60/k2":(S,2),"lowvol60/k3":(S,3),
         "momentum20/k2":(signal_momentum(P,20),2),"momentum20/k3":(signal_momentum(P,20),3),
         "momentum60/k2":(signal_momentum(P,60),2),"momentum60/k3":(signal_momentum(P,60),3)}
def curve(sig,k):
    s=np.where(np.isfinite(sig),sig,-1e9);eq=[1.0]
    for d in range(1,T-1):
        top=np.argsort(s[d-1])[-k:]
        r=np.nanmean(ROO[d,top]);r=0 if not np.isfinite(r) else r
        r-=5.0/1e4
        eq.append(eq[-1]*(1+r))
    return np.diff(eq)/np.array(eq[:-1])
curves={nm:curve(sg,k) for nm,(sg,k) in configs.items()}
# PBO: 8 段，IS 段选最优，看 OOS 排名
n=min(len(x) for x in curves.values())
M=np.array([x[:n] for x in curves.values()]);seg=n//8
ranks=[]
for i in range(7):
    a,b=i*seg,(i+1)*seg;c,d=(i+1)*seg,(i+2)*seg
    is_sharpe=M[:,a:b].mean(axis=1)/(M[:,a:b].std(axis=1)+1e-9)
    best=int(np.argmax(is_sharpe))
    oos=M[:,c:d].mean(axis=1)/(M[:,c:d].std(axis=1)+1e-9)
    rank=float((oos>oos[best]).mean())
    ranks.append(rank)
pbo=float(np.mean(np.array(ranks)<0.5))
print("[pbo] 6 configs / 7 IS-OOS splits, median_oos_rank",round(float(np.median(ranks)),3),"pbo",round(pbo,3),flush=True)
print("[pbo] ranks",[round(x,2) for x in ranks],flush=True)
