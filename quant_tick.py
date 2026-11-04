# -*- coding: utf-8 -*-
"""quant_tick.py -- 无人工干预地推进 quant 项目：因子落盘 + 财报表下载。

由系统计划任务每 30 分钟调用一次。每次只跑固定预算，跑不完下次接着来（两者都是断点续传）。
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "quant_tick.log")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def counts():
    c = {}
    A = os.path.join(HERE, "alpha_parts")
    for f in os.listdir(A):
        if f.endswith(".npy") and not f.startswith("meta_"):
            sid = int(f.rsplit("_", 1)[1][:5])
            c[sid] = c.get(sid, 0) + 1
    return c


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    t0 = time.time()
    log("tick start budget=%.0fs" % budget)
    # 1) 财务（慢，先做）
    r = subprocess.run([sys.executable, "-B", "fundpull.py", "%.0f" % (budget * 0.55)],
                       cwd=HERE, capture_output=True, text=True)
    log("fund rc=%d %s" % (r.returncode, (r.stdout or "").strip().split("\n")[-1][:80]))
    # 2) 因子（快，后做）
    calls = 0
    while time.time() - t0 < budget:
        c = counts()
        todo = [s for s in range(25) if c.get(s, 0) < 154]
        if not todo:
            log("alpha ALLDONE")
            break
        rr = subprocess.run([sys.executable, "-B", "fcalc.py", "--one", "seg", str(todo[0]), "0", "0", "90"],
                            cwd=HERE, capture_output=True, text=True)
        calls += 1
        if rr.returncode:
            log("alpha ERR %s" % (rr.stderr or "")[-150:])
    c = counts()
    done = sum(1 for v in c.values() if v >= 154)
    log("tick done %.0fs alpha=%d/25 calls=%d" % (time.time() - t0, done, calls))


if __name__ == "__main__":
    main()
