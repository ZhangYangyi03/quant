
import os,sys,pickle,json,time as _time
os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="1";os.environ["MKL_NUM_THREADS"]="1"
Q=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,Q)
import numpy as np,pandas as pd,qlib
from qlib.config import C
from qlib.backtest import backtest
from qlib_audit import _qlib_native_nav
H=os.path.join(Q,"qlib_bridge");OUT=os.path.join(H,"qlib_data")
qlib.init(provider_uri=OUT,region="cn",expression_cache=None,dataset_cache=None)
C["kernels"]=1;C["joblib_backend"]="threading"
raw=pickle.load(open(os.path.join(H,"_audit_pred.pkl"),"rb"))
if isinstance(raw,pd.DataFrame):raw=raw.iloc[:,0]
pool=pickle.load(open(os.path.join(H,"pred_dyn1000_2022.pkl"),"rb"))["pool"]
t=_time.time()
nav=_qlib_native_nav(raw,("2022-01-04","2022-12-30"),pool,50,1_000_000,None)
print("[native] qlib_nav",nav,"%.1fs"%(_time.time()-t),flush=True)
json.dump({"qlib_nav":nav},open(os.path.join(Q,"qlib_native_2022.json"),"w",encoding="utf-8"))
