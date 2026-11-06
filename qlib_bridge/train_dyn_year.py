
import os,sys,time,pickle
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
import numpy as np,pandas as pd,qlib
from qlib.config import C
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
H=os.path.dirname(os.path.abspath(__file__));OUT=os.path.join(H,"qlib_data")
qlib.init(provider_uri=OUT,region="cn",expression_cache=None,dataset_cache=None)
C["kernels"]=1;C["joblib_backend"]="threading"
# 2022 年可交易股票，且上市满 250 日（2019 前上市），避免次新
lines=[l.strip().split("\t") for l in open(os.path.join(OUT,"instruments","all.txt"),encoding="utf-8") if "\t" in l]
YEAR=int(os.environ.get("DY_YEAR","2022"));d0=f"{YEAR}-01-04"
import random
_cut=f"{YEAR-3}-12-31"
_all=[x[0] for x in lines if len(x)>=3 and x[1]<=_cut and x[2]>=d0]
random.Random(42).shuffle(_all)
pool=_all[:1000]
print(f"[dyn] {YEAR} universe",len(_all),"sample",len(pool),flush=True)
inst=os.path.join(OUT,"instruments",f"dyn{YEAR}.txt")
with open(inst,"w",encoding="utf-8") as f:
    f.write("\n".join("%s\t%s-01-01\t%s-12-31"%(c,YEAR-3,YEAR) for c in pool)+"\n")
feature=["Log($mktcap + 1)","Mean($amount, 20)/($amount + 1)","$close/Ref($close, 20) - 1","Ref($close, 5)/$close - 1"]
label=["Ref($close, -5)/$close - 1"]
loader={"class":"QlibDataLoader","kwargs":{"config":{"feature":feature,"label":label},"freq":"day"}}
t=time.time()
hd=DataHandlerLP(instruments=f"dyn{YEAR}",start_time=f"{YEAR-3}-01-01",end_time=f"{YEAR}-12-31",data_loader=loader,
  infer_processors=[{"class":"DropnaProcessor"},{"class":"CSZScoreNorm","kwargs":{"fields_group":"feature"}}],
  learn_processors=[{"class":"DropnaLabel"}])
print("[dyn] handler %.0fs rows %d"%(time.time()-t,len(hd._infer)),flush=True)
ds=DatasetH(handler=hd,segments={"train":(f"{YEAR-3}-01-01",f"{YEAR-1}-12-31"),"test":(f"{YEAR}-01-01",f"{YEAR}-12-31")})
m=LGBModel(loss="mse",num_boost_round=80,early_stopping_rounds=15,max_depth=4,learning_rate=0.05,num_leaves=16,verbose=-1)
m.fit(ds)
pred=m.predict(ds)
idx=getattr(pred,"index",None)
print("[dyn] pred",len(pred),"idx",None if idx is None else (idx.min(),idx.max()),flush=True)
pickle.dump({"pool":pool,"pred":np.asarray(pred,dtype=np.float64),"index":idx,
             "te_start":d0,"te_end":f"{YEAR}-12-31"},
            open(os.path.join(H,f"pred_dyn1000_{YEAR}.pkl"),"wb"))
print("[dyn] DONE %.0fs"%(time.time()-t),flush=True)
