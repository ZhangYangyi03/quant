# -*- coding: utf-8 -*-
"""check_env.py —— 部署自检：clone 后先跑这个。

只读，不改文件。逐项报告核心依赖、数据文件、可选 qlib 层、硬编码路径，
并给出「哪些脚本现在就能跑」。退出码 0 = 核心层可用。
"""
import os, sys, importlib.util, glob

Q = os.path.dirname(os.path.abspath(__file__))


def has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def main():
    print("== quant 部署自检 ==")
    print("python", sys.version.split()[0], sys.executable)

    print("\n[1] 核心依赖")
    core = {"numpy": True, "scipy": True, "pandas": True}
    ok_core = True
    for m in core:
        v = has(m)
        core[m] = v
        ok_core &= v
        print("  %-10s %s" % (m, "OK" if v else "MISSING -> pip install -r requirements.txt"))

    print("\n[2] 数据文件（核心 ETF 审计所需，已入库）")
    data = sorted(glob.glob(os.path.join(Q, "data", "*.csv")))
    extra = sorted(glob.glob(os.path.join(Q, "data_extra", "*.csv")))
    ok_data = len(data) >= 8 and len(extra) >= 20
    print("  data/*.csv        %d 只 (需 >=8)  %s" % (len(data), "OK" if len(data) >= 8 else "MISSING"))
    print("  data_extra/*.csv  %d 只 (需 >=20) %s" % (len(extra), "OK" if len(extra) >= 20 else "MISSING"))

    print("\n[3] 大数据（个股 / qlib 线，未入库，需本机重建）")
    for name, need in (("stockdata/", 1000), ("store/", 1000),
                       ("qlib_bridge/qlib_data/features/", 1000), ("m_compact.bin", 1)):
        p = os.path.join(Q, name)
        n = len(os.listdir(p)) if os.path.isdir(p) else (1 if os.path.exists(p) else 0)
        print("  %-38s %s" % (name, "OK %d" % n if n >= need else "MISSING (个股/qlib 线不可用，核心层不受影响)"))

    print("\n[4] 可选 qlib 层")
    q_ok = has("qlib")
    print("  qlib %s" % ("OK" if q_ok else "未安装 -> pip install -r requirements-qlib.txt（Linux 需源码编译）"))

    print("\n[5] 绝对路径残留（应只剩文档/检测目标）")
    import subprocess
    bad = []
    try:
        tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=Q,
                                 capture_output=True, text=True).stdout.splitlines()
    except Exception:
        tracked = []
    for rel in tracked:
        if rel == "check_env.py" or rel.endswith("test_regressions.py"):
            continue  # 前者是检测器本身，后者把路径当检测目标
        fp = os.path.join(Q, rel)
        if not os.path.exists(fp):
            continue
        t = open(fp, encoding="utf-8", errors="replace").read()
        if "D:\\Users\\china" in t or '".qlibenv", "Scripts"' in t:
            bad.append(rel)
    print("  ", bad if bad else "无")

    print("\n[6] 现在能跑什么")
    can = ["verdict_all.py", "audit_backtest.py 库", "test_audit_backtest.py",
           "test_etf_alpha.py", "tests/ (110，缺大数据时 2 个 skip)", "etf_year_audit.py",
           "etf_holdout.py", "etf_capacity.py", "etf_universe_expanded.py"]
    if ok_core and ok_data:
        for c in can:
            print("   OK  ", c)
    else:
        print("   核心层未就绪，先装 requirements.txt 并确认 data/*.csv 存在")
    print("   需大数据  audit_all.py / qlib_bridge/*.py / qlib_audit.py")

    print("\n== 核心层 %s ==" % ("可用" if (ok_core and ok_data) else "不可用"))
    return 0 if (ok_core and ok_data) else 1


if __name__ == "__main__":
    sys.exit(main())
