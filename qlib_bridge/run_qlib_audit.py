
import os,sys,json,pickle
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
Q=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Q)
import numpy as np,pandas as pd,qlib
from qlib.config import C
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from qlib_audit import qlib_audit_dataset
H=os.path.join(Q,"qlib_bridge");OUT=os.path.join(H,"qlib_data")
qlib.init(provider_uri=OUT,region="cn",expression_cache=None,dataset_cache=None)
C["kernels"]=1;C["joblib_backend"]="threading"
pool=pickle.load(open(os.path.join(H,"pred_dyn1000_2022.pkl"),"rb"))["pool"]
feature=["Log($mktcap + 1)","Mean($amount, 20)/($amount + 1)","$close/Ref($close, 20) - 1","Ref($close, 5)/$close - 1"]
label=["Ref($close, -5)/$close - 1"]
loader={"class":"QlibDataLoader","kwargs":{"config":{"feature":feature,"label":label},"freq":"day"}}
hd=DataHandlerLP(instruments="dyn2022",start_time="2019-01-01",end_time="2022-12-31",data_loader=loader,
  infer_processors=[{"class":"DropnaProcessor"},{"class":"CSZScoreNorm","kwargs":{"fields_group":"feature"}}],
  learn_processors=[{"class":"DropnaLabel"}])
ds=DatasetH(handler=hd,segments={"train":("2019-01-01","2021-12-31"),"test":("2022-01-01","2022-12-31")})
PRED=os.path.join(H,"_audit_pred.pkl")
import pickle as _pk
if os.path.exists(PRED):
    _raw=_pk.load(open(PRED,"rb"))
    class _M:
        def predict(self,d):return _raw
    m=_M()
else:
    m=LGBModel(loss="mse",num_boost_round=80,early_stopping_rounds=15,max_depth=4,learning_rate=0.05,num_leaves=16,verbose=-1)
    m.fit(ds)
    _pk.dump(m.predict(ds),open(PRED,"wb"))
import time as _t
_t0=_t.time()
r=qlib_audit_dataset(ds,m,("2022-01-04","2022-12-30"),topk=50,cost_bp=3.65,run_qlib_backtest=False)
print("[timing] qlib_audit_dataset %.1fs"%(_t.time()-_t0),flush=True)
print("[qlib_audit] T",r["T"],"N",r["N"],"signal_pairs",r["signal_pairs"],"expected",r["expected_pairs"])
print("[qlib_audit] qlib_nav",r.get("qlib_nav"),"nav_gap",r.get("nav_gap"),"err",r.get("qlib_nav_error"))
a=r["audit"]
print("[qlib_audit] verdict",a["verdict"],"fails",a["fails"])
print("[qlib_audit] L1",json.dumps(a["L1_signal_index"],ensure_ascii=False)[:500])
print("[qlib_audit] L2",json.dumps(a["L2_alignment"],ensure_ascii=False)[:400])
print("[qlib_audit] exec_alpha",a["executable_alpha_nav"])
json.dump(r,open(os.path.join(Q,"qlib_audit_2022.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1,default=str)
