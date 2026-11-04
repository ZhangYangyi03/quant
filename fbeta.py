"""fbeta.py -- 分位数组合回测，把因子换成"能不能赚钱"的答案。

为什么还要这一步
----------------
IC 只回答"排序有没有信息"，不回答"扣掉成本还剩不剩钱"，而 A 股换手惩罚很重，
很多公开因子的纸面 IC 赚不到钱。这里做最朴素也最可信的一版：
  每个调仓日按因子值分 5 组等权，看多空（Top-Bottom）与多头（Top 对全市场均值）
  的月均收益、t 值、胜率、换手、扣费后净值。

成本
----
  买入 0.025%、卖出 0.075%（佣金 + 印花税 + 滑点的保守合并），按每期实际换手计。

输入
----
  ic_fast.npz（由 fastpanel.py 写出：{factor: (月份 x 股票) 因子值}）
  没有就报错，不猜。

用法：
  python fbeta.py                 # 全部因子
  python fbeta.py --split 2016-01-01   # 2016 年前后分开报，看样本外
"""
import os, sys, json, math, time
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from compact import Panel

FEE_BUY, FEE_SELL = 0.00025, 0.00075
Q = 5


def quantile_groups(fv, alive, q=Q):
    """按因子值把活着的股票切成 q 组，返回每个股票所在的组号（0 最低，q-1 最高）。"""
    idx = np.flatnonzero(alive & np.isfinite(fv))
    if len(idx) < 200:
        return None, None
    order = idx[np.argsort(fv[idx], kind="stable")]
    bounds = np.linspace(0, len(order), q + 1).astype(int)
    g = np.full(fv.shape, -1, dtype=np.int8)
    for k in range(q):
        g[order[bounds[k]:bounds[k + 1]]] = k
    return g, order


def main(argv):
    split = None
    if "--split" in argv:
        split = argv[argv.index("--split") + 1]
    z = os.path.join(HERE, "ic_fast.npz")
    if not os.path.exists(z):
        print("缺 ic_fast.npz：先跑 fastpanel.py export")
        return
    D = np.load(z, allow_pickle=False)
    factors = [k for k in D.files if k not in ("dates", "alive", "fwd")]
    p = Panel()
    dates = [str(x) for x in D["dates"]]
    alive = D["alive"]
    fwd = D["fwd"]
    print("因子 %d 个，%d 个月：%s ~ %s" % (len(factors), alive.shape[0], dates[0], dates[-1]))

    rows = []
    for f in factors:
        F = D[f]
        top_ret, bot_ret, mkt_ret, turn, net = [], [], [], [], []
        prev_top = None
        for k in range(alive.shape[0] - 1):
            fv = F[k].astype(np.float64)
            a = alive[k] & np.isfinite(fv) & np.isfinite(fwd[k])
            g, order = quantile_groups(fv, a)
            if g is None:
                continue
            r = fwd[k]
            top = g == Q - 1
            bot = g == 0
            top_ret.append(float(r[top].mean()))
            bot_ret.append(float(r[bot].mean()))
            mkt_ret.append(float(r[a].mean()))
            cur = set(np.flatnonzero(top).tolist())
            if prev_top is not None:
                t_ = len(cur - prev_top) / max(1, len(cur))
            else:
                t_ = 1.0
            prev_top = cur
            turn.append(t_)
            cost = t_ * (FEE_BUY + FEE_SELL) + (1 - t_) * 0.0
            net.append(top_ret[-1] - cost)
        if len(top_ret) < 12:
            continue
        ls = np.array(top_ret) - np.array(bot_ret)
        lt = np.array(net) - np.array(mkt_ret)
        def stat(x):
            m, s = x.mean(), x.std(ddof=1)
            return m, (m / (s / math.sqrt(len(x))) if s > 0 else float("nan"))
        ls_m, ls_t = stat(ls)
        lt_m, lt_t = stat(lt)
        rec = {"factor": f, "n": len(ls),
               "ls_m": round(ls_m * 100, 3), "ls_t": round(ls_t, 2),
               "top_m": round(float(np.mean(top_ret)) * 100, 3),
               "excess_m": round(lt_m * 100, 3), "excess_t": round(lt_t, 2),
               "win": round(float((lt > 0).mean()), 3),
               "turn": round(float(np.mean(turn)) * 100, 1)}
        rows.append(rec)
        if split:
            cut = next(i for i, d in enumerate(dates) if d >= split)
            for part, sl in (("in", slice(0, cut)), ("oos", slice(cut, None))):
                x = lt[sl]
                if len(x) > 6:
                    m_, t_ = stat(x)
                    rec[part + "_excess"] = round(m_ * 100, 3)
                    rec[part + "_t"] = round(t_, 2)
    rows.sort(key=lambda r: -(r["excess_t"] or -9))
    hdr = "%-10s %4s %8s %7s %8s %8s %6s %6s" % ("factor", "n", "LS月均%", "LS_t", "超额月均%", "超额_t", "胜率", "月换手%")
    if split:
        hdr += " %9s %9s" % ("样本内超额", "样本外超额")
    print(hdr)
    for r in rows:
        line = "%-10s %4d %8.3f %7.2f %8.3f %8.2f %6.3f %6.1f" % (
            r["factor"], r["n"], r["ls_m"], r["ls_t"], r["excess_m"], r["excess_t"], r["win"], r["turn"])
        if split and "oos_excess" in r:
            line += " %9.3f %9.3f" % (r["in_excess"], r["oos_excess"])
        print(line)
    json.dump(rows, open(os.path.join(HERE, "fbeta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved fbeta.json")


if __name__ == "__main__":
    main(sys.argv)
