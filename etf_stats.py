
# -*- coding: utf-8 -*-
"""etf_stats.py —— lowvol60 的单策略多重检验校正：DSR / NW-t / PSR。"""
import os,sys,json,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
sys.path.insert(0,os.path.join(Q,"quantlab"))
from audit_etf import load_panel,signal_lowvol
dates,codes,P=load_panel();T,N=P["close"].shape
S=signal_lowvol(P,60)
C,O,A=P["close"],P["open_"],P["amount"]
ROO=np.full((T-1,N),np.nan);ok=np.isfinite(O[:-1])&np.isfinite(O[1:])&(O[:-1]>0);ROO[ok]=O[1:][ok]/O[:-1][ok]-1
GAP=np.full((T,N),np.nan);ok=np.isfinite(C[:-1])&np.isfinite(O[1:])&(C[:-1]>0);GAP[1:][ok]=O[1:][ok]/C[:-1][ok]-1
GAP[(~np.isfinite(A))|(A<=0)]=np.nan
s=np.where(np.isfinite(S),S,-1e9)
def curve(rebal=20,cost_bp=5.0,block=True):
    eq=[1.0];cur=None;prev=None
    for k in range(1,T-1):
        if cur is None or (k-1)%rebal==0:
            sc=s[k-1].copy()
            if block:
                bad=~np.isfinite(GAP[k+1])|(GAP[k+1]>0.095);sc[bad]=-1e9
            cur=np.argsort(sc)[-2:]
        r=np.nanmean(ROO[k,cur]);r=0 if not np.isfinite(r) else r
        if prev is not None and not np.array_equal(cur,prev):
            r-=cost_bp/1e4*len(set(cur)^set(prev))/2
        prev=cur
        eq.append(eq[-1]*(1+r))
    return np.diff(eq)/np.array(eq[:-1])
r=curve()
n=len(r)
mu=float(r.mean());sd=float(r.std(ddof=1))
sharpe=mu/sd*np.sqrt(244)
skew=float(((r-mu)**3).mean()/sd**3);kurt=float(((r-mu)**4).mean()/sd**4)
# Newey-West t on mean return
def nw_t(x,lags=5):
    x=np.asarray(x);n=len(x);e=x-x.mean();g0=(e*e).mean();s=g0
    for l in range(1,lags+1):
        g=(e[l:]*e[:-l]).mean();s+=2*(1-l/(lags+1))*g
    return float(x.mean()/np.sqrt(max(s,1e-18)/n))
t=nw_t(r)
# PSR / DSR (Bailey & Lopez de Prado) 用标准正态近似
from math import sqrt,log,erf
def Phi(z):return 0.5*(1+erf(z/sqrt(2)))
def psr(sr,sr0,n,g3,g4):
    if n<=1:return 0.5
    den=1-g3*sr+(g4-1)/4*sr*sr
    return Phi((sr-sr0)*sqrt(n-1)/sqrt(max(den,1e-9)))
sr=sharpe/sqrt(244)
# 试验次数：results_etf.json 记 10 个配置；DSR 的 sr0 用期望最大夏普近似
trials=10
sr0=sqrt(2*log(trials))*(1-0.5772/(2*log(trials)))/sqrt(n) if n>0 else 0
sr0*=1.0   # 单位：日频夏普基准
print("[stat] days",n,"ann_sharpe",round(float(sharpe),4),"skew",round(skew,3),"kurt",round(kurt,3))
print("[stat] NW-t mean return",round(t,4))
print("[stat] PSR(sr0=0)",round(float(psr(sr,0,n,skew,kurt)),4))
print("[stat] DSR(trials=%d)"%trials,round(float(psr(sr,sr0,n,skew,kurt)),4),"sr0_daily",round(float(sr0),6))
# 逐年 OOS：把 10 年切成 5 个 2 年块，只用历史选参数(无参数)直接看
for rb in (10,20,60):
    x=curve(rebal=rb);print("[stat] rebal",rb,"sharpe",round(float(x.mean()/x.std(ddof=1)*sqrt(244)),3),"NWt",round(nw_t(x),3))
json.dump({"days":n,"annual_sharpe":float(sharpe),"nw_t":t,"skew":skew,"kurt":kurt,
 "psr_vs_zero":float(psr(sr,0,n,skew,kurt)),
 "dsr_trials10":float(psr(sr,sr0,n,skew,kurt)),
 "rebal20_net_nav":float(np.prod(1+curve(20)))},
 open(os.path.join(Q,"etf_stats.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
