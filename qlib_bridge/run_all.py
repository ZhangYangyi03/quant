# -*- coding: utf-8 -*-
"""run_all.py —— 一键跑完 导出 → 训练 → 回测 → 对账。
每个子进程独立，避开本机 512MB 提交上限与单次调用时长限制。
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
# 用当前解释器；本机若保留 .qlibenv 且当前不是它，也可通过 PYTHON 环境变量覆盖。
VENV = os.environ.get("PYTHON") or sys.executable
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
           MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
ALL = ["export.py", "bridge_train.py", "bridge_run.py", "reconcile.py"]
# 单次调用窗口约 90s，支持 --part 1/2 分两轮跑
part = sys.argv[1] if len(sys.argv) > 1 else ""
steps = ALL[:2] if part == "1" else (ALL[2:] if part == "2" else ALL)
for s in steps:
    t = time.time()
    print("[all] === %s ===" % s, flush=True)
    r = subprocess.run([VENV, "-u", s], cwd=HERE, env=env)
    print("[all] %s rc=%d %.0fs" % (s, r.returncode, time.time() - t), flush=True)
    if r.returncode:
        sys.exit(r.returncode)
print("[all] done", flush=True)
