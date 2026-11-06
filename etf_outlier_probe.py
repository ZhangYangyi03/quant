
import os,sys,numpy as np
Q=os.path.dirname(os.path.abspath(__file__));sys.path.insert(0,Q)
from audit_etf import load_panel,signal_lowvol
dates,codes,P=load_panel();T,N=P["close"].shape
C,O=P["close"],P["open_"]
ROO=np.full((T-1,N),np.nan);ok=np.isfinite(O[:-1])&np.isfinite(O[1:])&(O[:-1]>0);ROO[ok]=O[1:][ok]/O[:-1][ok]-1
# 找所有 |ret|>0.15 的格子
bad=np.argwhere(np.isfinite(ROO)&(np.abs(ROO)>0.15))
print("cells |O2O ret|>15%",len(bad))
for i,j in bad[:30]:
    print(" ",dates[i+1],codes[j],"O",O[i,j],O[i+1,j],"C",C[i,j],C[i+1,j],"ret",round(float(ROO[i,j]),4))
# 全市场每天最大 |ret| 分布
mx=np.nanmax(np.abs(np.nan_to_num(ROO,nan=0)),axis=1)
print("daily max |ret| >10% days",int((mx>0.10).sum()),">15%",int((mx>0.15).sum()),">20%",int((mx>0.20).sum()))
