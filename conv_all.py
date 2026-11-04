# -*- coding: utf-8 -*-
"""conv_all.py -- 把所有还没算完的段一口气跑完（外部计时器按 N 秒调它）。"""
import os, subprocess, sys, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
LOGF = os.path.join(HERE, "conv_all.log")
WANT = 154          # 每段应该有多少个因子文件（RESI/MOMR 的重复项已删，现库共 154 个）

def log(msg):
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%H:%M:%S"), msg))
    print(msg, flush=True)

def counts():
    A = os.path.join(HERE, "alpha_parts")
    c = {}
    for f in os.listdir(A):
        if f.endswith(".npy") and not f.startswith("meta_"):
            sid = int(f.rsplit("_", 1)[1][:5])
            c[sid] = c.get(sid, 0) + 1
    return c

def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    t0 = time.time()
    log("start budget=%.0fs" % budget)
    while time.time() - t0 < budget:
        c = counts()
        todo = [s for s in range(25) if c.get(s, 0) < WANT]
        if not todo:
            log("ALLDONE")
            return
        sid = todo[0]
        r = subprocess.run([sys.executable, "-B", "fcalc.py", "--one", "seg", str(sid), "0", "0", "90"],
                           cwd=HERE, capture_output=True, text=True)
        log("seg %d rc=%d %s" % (sid, r.returncode, (r.stdout or "").strip()[:80]))
        if r.returncode:
            log("ERR " + (r.stderr or "")[-200:])
    log("done %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
