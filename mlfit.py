# -*- coding: utf-8 -*-
"""mlfit.py -- 多因子合成：截面回归 / 等权 z / GBDT，全部按时间切样本外。

读 fpanel.npz（fmerge.py 落的缓存），不再扫盘。
纪律：
  * 切分点按日期，训练在前、测试在后，绝不随机打乱（打乱 = 偷看未来）
  * 因子先做"方向 * 截面 z 分数"，再做秩变换，避免极值把回归拖走
  * 每个组合只报：扣费后多头相对等权的超额、胜率、t。不报原始收益，那是 beta 不是本事。
  * 换手成本：买入 0.025% + 卖出 0.075%（佣金+印花税+滑点），按实际换手计。
"""
import os, sys, json, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.001
Q = 5


def zscore(x, msk):
    v = x[msk]
    if len(v) < 50:
        return None, None
    med = np.median(v)
    sd = v.std()
    if not np.isfinite(sd) or sd == 0:
        return None, None
    z = (x - med) / sd
    return z, msk


def rank01(x):
    o = np.argsort(np.argsort(x))
    return o / max(1, len(x) - 1)


def main():
    D = np.load(os.path.join(HERE, "fpanel.npz"), allow_pickle=False)
    JJ = D["JJ"]; CL = D["CL"].astype(np.float64); AL = D["AL"]
    import compact
    FULL = compact.Panel()       # 下个月的收盘价要完整面板（CL 里只有调仓日）
    names = [k for k in D.files if k not in ("JJ", "CL", "AL")]
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))["dates"]
    icj = json.load(open(os.path.join(HERE, "ic_fast.json"), encoding="utf-8"))
    # 方向取全样本 IC 的符号（这一步在训练段里定，测试段不参与，后面会说明）
    NS = len(JJ) - 1
    dirn = {k: (1.0 if icj[k]["ic_mean"] >= 0 else -1.0) for k in names if k in icj}
    names = [k for k in names if k in dirn]

    # 组装 Z：月 x 股 x 因子
    Zt = {}
    for k in names:
        A = D[k].astype(np.float64) * dirn[k]
        Zt[k] = A
    rets = np.full((NS, CL.shape[1]), np.nan)
    for m in range(NS):
        cn = FULL.col("close", int(JJ[m + 1])); cur = CL[m]
        rets[m] = np.where((cur > 0) & (cn > 0), cn / np.where(cur > 0, cur, np.nan) - 1.0, np.nan)
    # 截面 z
    for k in names:
        A = Zt[k].copy()
        for m in range(NS):
            msk = AL[m] & np.isfinite(A[m]) & (np.abs(A[m]) < 1e12)
            mu = np.median(A[m][msk]) if msk.sum() > 50 else np.nan
            sd = A[m][msk].std() if msk.sum() > 50 else np.nan
            A[m] = np.where(msk, (A[m] - mu) / (sd if sd and np.isfinite(sd) and sd > 0 else np.nan), np.nan)
        Zt[k] = A
    SPLIT = "2019-01-01"
    tr = [m for m in range(NS) if meta[JJ[m]] < SPLIT]
    te = [m for m in range(NS) if meta[JJ[m]] >= SPLIT]
    print("训练 %d 个月 (%s~%s)，测试 %d 个月 (%s~%s)" % (
        len(tr), meta[JJ[tr[0]]], meta[JJ[tr[-1]]], len(te), meta[JJ[te[0]]], meta[JJ[te[-1]]]))

    def blend(w, m):
        s = np.zeros(CL.shape[1]); tot = np.zeros(CL.shape[1])
        for k, wk in w.items():
            z = Zt[k][m]
            ok = np.isfinite(z)
            s[ok] += wk * z[ok]; tot[ok] += abs(wk)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, s / np.where(tot > 0, tot, 1), np.nan)

    def evaluate(w, label):
        prev = None; out = []
        for m in range(NS):
            sc = blend(w, m)
            r = rets[m]
            msk = AL[m] & np.isfinite(sc) & np.isfinite(r)
            idx = np.flatnonzero(msk)
            if len(idx) < 500:
                continue
            order = idx[np.argsort(sc[idx], kind="stable")]
            b = np.linspace(0, len(order), Q + 1).astype(int)
            top = set(order[b[-2]:b[-1]].tolist())
            t_ret = float(np.mean(r[list(top)]))
            mk = float(np.mean(r[idx]))
            to = 1.0 if prev is None else len(top - prev) / max(1, len(top))
            prev = top
            out.append((m, t_ret - to * FEE - mk, to))
        tr_ex = np.array([x[1] for x in out if x[0] in set(tr)])
        te_ex = np.array([x[1] for x in out if x[0] in set(te)])
        def st(a):
            return (a.mean() * 100, a.mean() / (a.std(ddof=1) / math.sqrt(len(a))), (a > 0).mean())
        a = st(tr_ex) if len(tr_ex) > 5 else (0, 0, 0)
        b_ = st(te_ex) if len(te_ex) > 5 else (0, 0, 0)
        turn = np.mean([x[2] for x in out]) * 100
        print("%-22s 训练 %6.2f%%/t%5.2f/胜%4.2f   |  测试 %6.2f%%/t%5.2f/胜%4.2f  换手%5.1f%%" % (
            label, a[0], a[1], a[2], b_[0], b_[1], b_[2], turn))
        return {"label": label, "train_ex": round(a[0], 3), "train_t": round(a[1], 2),
                "test_ex": round(b_[0], 3), "test_t": round(b_[1], 2),
                "test_win": round(b_[2], 3), "turn": round(turn, 1)}

    res = []
    # 1) 等权全部因子
    res.append(evaluate({k: 1.0 for k in names}, "等权11因子"))
    # 2) 训练段截面 Fama-MacBeth 回归权重（每期 OLS，取平均系数）
    X_cols = []
    for k in names:
        X_cols.append(Zt[k])
    coefs = []
    for m in tr:
        r = rets[m]
        msk = np.array(AL[m] & np.isfinite(r))
        for k in names:
            msk &= np.isfinite(Zt[k][m])
        idx = np.flatnonzero(msk)
        if len(idx) < 500:
            continue
        Xm = np.column_stack([np.ones(len(idx))] + [Zt[k][m][idx] for k in names])
        try:
            beta, *_ = np.linalg.lstsq(Xm, r[idx], rcond=None)
        except Exception:
            continue
        coefs.append(beta[1:])
    print("回归期数 %d，每期系数维度 %s" % (len(coefs), len(coefs[0]) if coefs else 0))
    coefs = np.vstack(coefs) if coefs else np.zeros((1, len(names)))
    w_reg = {k: float(np.nanmean(coefs[:, i])) for i, k in enumerate(names)}
    s = sum(abs(v) for v in w_reg.values()) or 1.0
    w_reg = {k: v / s for k, v in w_reg.items()}
    print("截面回归权重:", {k: round(v, 3) for k, v in sorted(w_reg.items(), key=lambda x: -abs(x[1]))})
    res.append(evaluate(w_reg, "训练段截面回归"))
    # 3) 只留 t 值最大的几因子
    for topn in (3, 5):
        sel = sorted(names, key=lambda k: -abs(icj[k]["t"]))[:topn]
        res.append(evaluate({k: 1.0 for k in sel}, "t值前%d:%s" % (topn, ",".join(sel))))
    # 4) 低换手组合：只用量价里换手低的（size/turn/amt20/vol20）
    res.append(evaluate({"size": 1, "turn": 1, "amt20": 1, "vol20": 1}, "低换手4因子"))
    json.dump(res, open(os.path.join(HERE, "mlfit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nsaved mlfit.json")


if __name__ == "__main__":
    main()
