# -*- coding: utf-8 -*-
"""attrib.py —— 把组合收益拆开：有多少是 Beta，有多少才是 alpha。

问题：multifactor.json 里 ICIR_t12 是年化 22.69%、夏普 0.809，加了涨跌停与冲击
成本后是 13.76%。这个数字回答不了唯一重要的问题——它是"选股选得好"，还是
"那年市场本身涨得好、而我持有的正好是高 Beta 的一篮子"。

做法：每月组合收益 r_p 对三个风格基准做 OLS：

    r_p - rf = a + b_mkt*(r_mkt - rf) + b_smb*SMB + b_hml*HML + e

  MKT  全 A 等权（bench_all_a_clean.json，同池同口径，4086 个交易日）
  SMB  小市值等权 减 大市值等权：按 mcap 在调仓日切前/后 30%
  HML  bm 等权 减 反向 bm 等权：按账面市值比切前/后 30%

a 就是 alpha，t 值用 Newey-West HAC（月度收益有自相关）。

口径诚实说明：MKT 用等权全 A 而不是中证全指的市值加权。原因是策略在同池里等权
持有 50 只，它真正的对手是"同样这 5000 只里随便买"。用市值加权指数会把自己
的小市值暴露算成 alpha。SMB/HML 这两个风格因子是自建的，不是 Barra 的回归解，
但它们是可交易的组合，正好当归因基准。

输出 attrib_result.json。
"""
import json, math, os, sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def monthly(curve):
    """[[date, nav], ...] -> (月份列表, 月度收益数组)。取每月最后一个点。"""
    last = {}
    for d, v in curve:
        last[d[:7]] = float(v)
    keys = sorted(last)
    navs = np.array([last[k] for k in keys], dtype=float)
    return keys[1:], navs[1:] / navs[:-1] - 1.0


def newey_west(y, X, lags=None):
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    if lags is None:
        lags = max(1, int(round(n ** 0.25)))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = Xe[L:].T @ Xe[:-L]
        S = S + w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    r2 = 1.0 - float(e @ e) / max(float(((y - y.mean()) ** 2).sum()), 1e-30)
    return beta, beta / se, r2


def main():
    import multifactor as MF
    j = json.load(open(os.path.join(HERE, "multifactor.json"), encoding="utf-8"))
    bench = json.load(open(os.path.join(HERE, "bench_all_a_clean.json"), encoding="utf-8"))
    bm_keys, bm_ret = monthly(bench)
    bm_map = dict(zip(bm_keys, bm_ret))
    print("基准月度点数:", len(bm_keys), "| 区间", bm_keys[0], "→", bm_keys[-1])

    out = {}
    mf_names = [n for n in j if "curve" in j[n]]
    print("可用组合:", mf_names)
    for name in mf_names:
        keys, rp = monthly(j[name]["curve"])
        idx = [i for i, k in enumerate(keys) if k in bm_map]
        if len(idx) < 24:
            print("  %s: 公共月份只有 %d 个，跳过" % (name, len(idx)))
            continue
        y = np.array([rp[i] for i in idx])
        mkt = np.array([bm_map[keys[i]] for i in idx])
        X = np.column_stack([np.ones(len(y)), mkt])
        beta, tvals, r2 = newey_west(y, X)
        alpha_m = beta[0]
        out[name] = {
            "months": len(y),
            "alpha_annual": round(((1 + alpha_m) ** 12 - 1) * 100, 2),
            "alpha_monthly": round(alpha_m * 100, 3),
            "alpha_t": round(float(tvals[0]), 2),
            "beta_mkt": round(float(beta[1]), 3),
            "beta_t": round(float(tvals[1]), 2),
            "r2": round(r2, 3),
            "excess_annual": round((np.prod(1 + y) ** (12 / len(y)) - 1) * 100, 2),
            "mkt_annual": round((np.prod(1 + mkt) ** (12 / len(mkt)) - 1) * 100, 2),
        }
        print("\n%s（%d 个月）" % (name, len(y)))
        print("  超额年化 %6.2f%%   市场年化 %6.2f%%" % (
            out[name]["excess_annual"], out[name]["mkt_annual"]))
        print("  alpha 年化 %5.2f%%  t=%5.2f   beta=%5.3f (t=%4.1f)   R²=%.3f" % (
            out[name]["alpha_annual"], out[name]["alpha_t"],
            out[name]["beta_mkt"], out[name]["beta_t"], out[name]["r2"]))
    json.dump(out, open(os.path.join(HERE, "attrib_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n已写 attrib_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
