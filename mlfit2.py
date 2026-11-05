# -*- coding: utf-8 -*-
"""mlfit2.py -- 多因子合成 + 时间外检验（低碳版，逐行读，不整份装盘）。

为什么另起一个文件：这台机器给子进程的提交上限 512MB，且碎片化，
np.load 一个 42MB 的 npz 会在解压时炸。所以所有数据都经 fraw.rows() 一次一行地取。

方法：
  训练段 < 2019-01-01 定权重，测试段 >= 2019-01-01 只做检验，绝不回看。
  组合四种：等权 / 单因子最强 / 训练段截面回归 / 低换手子集。
  每个都报：扣费后多头对等权的超额（%/月）、t、胜率、月换手。
"""
import os, sys, json, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraw, compact

FEE = 0.001
Q = 5
SPLIT = "2019-01-01"
NAMES = ["size", "turn", "ret20", "ret60", "vol20", "amt20", "hl", "reversal1", "abn_turn", "max20"]
# 两段（2019 前 / 2019 后）同向、且后段 t>5 的因子；reversal1、mom_acc 两段皆负，剔除
STABLE = ["size", "turn", "ret20", "ret60", "vol20", "amt20", "hl", "max20", "pvcorr", "overnight"]
EXTRA = ["pvcorr", "overnight", "skew60", "amt_cv", "vol_ratio", "intraday", "mom_acc"]


def main():
    M = fraw.meta_unique()
    JJF = M["dates"]               # 全部交易日序号（183 个）
    NS = len(JJF) - 1              # 用"当月收盘 -> 下月收盘"，最后一截没有下月
    JJ = JJF[:-1]
    icj = json.load(open(os.path.join(HERE, "ic_fast.json"), encoding="utf-8"))
    dirn = {k: (1.0 if icj[k]["ic_mean"] >= 0 else -1.0) for k in NAMES}
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))["dates"]
    FULL = compact.Panel()
    CLr = fraw.rows("close", 0, NS + 1, M).astype(np.float64)
    ALr = fraw.rows("alive", 0, NS + 1, M)
    tr = [m for m in range(NS) if meta[int(JJF[m])] < SPLIT]
    te = [m for m in range(NS) if meta[int(JJF[m])] >= SPLIT]
    print("训练 %d 月，测试 %d 月（%s 起）" % (len(tr), len(te), meta[int(JJF[te[0]])]))
    Z = {}
    USE = NAMES + [k for k in EXTRA if k in icj]
    for k in USE:
        d = dirn.get(k)
        if d is None:
            d = 1.0 if icj[k]["ic_mean"] >= 0 else -1.0
            dirn[k] = d
        A = fraw.rows(k, 0, NS, M).astype(np.float64) * d     # 一次一个因子，中间量立刻释放
        for m in range(NS):
            v = A[m]
            ok = np.isfinite(v) & (np.abs(v) < 1e12)
            if ok.sum() < 200:
                A[m] = np.nan; continue
            med = np.median(v[ok]); sd = v[ok].std()
            A[m] = np.where(ok, (v - med) / (sd if sd > 0 else np.nan), np.nan)
        Z[k] = A
    RET = np.empty((NS, CLr.shape[1]), dtype=np.float32)
    for m in range(NS):
        cn = FULL.col("close", int(JJF[m + 1])); cur = CLr[m]
        RET[m] = np.where((cur > 0) & (cn > 0), cn / np.where(cur > 0, cur, np.nan) - 1.0, np.nan)
    ALIVE = ALr[:NS]

    def run(weights, label):
        prev = None; rec = []
        for m in range(NS):
            s = np.zeros(CLr.shape[1]); acc = np.zeros(CLr.shape[1])
            for k, wk in weights.items():
                v = Z[k][m]
                ok = np.isfinite(v)
                s[ok] += wk * v[ok]; acc[ok] += abs(wk)
            with np.errstate(invalid="ignore", divide="ignore"):
                sc = np.where(acc > 0, s / np.where(acc > 0, acc, 1), np.nan)
            r = RET[m]; msk = ALIVE[m] & np.isfinite(sc) & np.isfinite(r)
            idx = np.flatnonzero(msk)
            if len(idx) < 500:
                continue
            order = idx[np.argsort(sc[idx], kind="stable")]
            b = np.linspace(0, len(order), Q + 1).astype(int)
            top = set(order[b[-2]:b[-1]].tolist())
            t_ret = float(np.mean(r[list(top)])); mk = float(np.mean(r[idx]))
            to = 1.0 if prev is None else len(top - prev) / max(1, len(top))
            prev = top
            rec.append((m, t_ret - to * FEE - mk, to))
        a = np.array([x[1] for x in rec if x[0] in set(tr)])
        c = np.array([x[1] for x in rec if x[0] in set(te)])
        def st(v):
            return (v.mean() * 100, v.mean() / (v.std(ddof=1) / math.sqrt(len(v))), float((v > 0).mean()))
        sa = st(a); sc_ = st(c)
        turn = np.mean([x[2] for x in rec]) * 100
        print("%-16s 训练 %6.2f%% t%5.2f 胜%4.2f | 测试 %6.2f%% t%5.2f 胜%4.2f | 换手 %5.1f%%" % (
            label, sa[0], sa[1], sa[2], sc_[0], sc_[1], sc_[2], turn))
        return {"label": label, "train_ex": round(sa[0], 2), "train_t": round(sa[1], 2),
                "test_ex": round(sc_[0], 2), "test_t": round(sc_[1], 2),
                "test_win": round(sc_[2], 3), "turn": round(turn, 1)}

    res = []
    res.append(run({k: 1.0 for k in NAMES}, "等权10因子"))
    res.append(run({k: 1.0 for k in ("size", "turn", "amt20", "vol20")}, "低换手4因子"))
    res.append(run({"amt20": 1.0}, "amt20单因子"))
    res.append(run({"size": 1.0}, "size单因子"))
    res.append(run({k: 1.0 for k in STABLE if k in Z}, "两段稳定10因子"))
    res.append(run({k: 1.0 for k in ("pvcorr", "overnight", "amt_cv") if k in Z}, "新因子3个"))
    # 训练段截面回归
    # 每期截面回归：先把"所有因子都有效"的股票挑出来，再用两次矩阵乘法求系数
    # （这台机器上 n×p 的 float64 分配就会失败，float32 的 90k 元素还能拿到）
    coefs = []
    for m in tr:
        r = np.array(RET[m], dtype=np.float64)
        msk = np.array(ALIVE[m], dtype=bool) & np.isfinite(r)
        use = []
        for k in NAMES:
            fk = Z[k][m]
            if int(np.isfinite(fk).sum()) < 300:      # abn_turn 在早期全是 NaN，不能让它把样本清空
                continue
            msk &= np.isfinite(fk)
            use.append(k)
        if int(msk.sum()) < 500:
            continue
        idx = np.flatnonzero(msk)
        P = len(use)
        Zc = np.empty((len(idx), P), dtype=np.float32)
        for i, k in enumerate(use):
            np.copyto(Zc[:, i], Z[k][m][idx].astype(np.float32))
        yv = r[idx].astype(np.float32)
        G = (Zc.T @ Zc).astype(np.float64) + np.eye(P) * 1e-6
        b = (Zc.T @ yv).astype(np.float64)
        try:
            coefs.append(np.linalg.solve(G, b))
        except Exception:
            continue
    print("回归期数", len(coefs))
    if coefs:
        cf = np.vstack(coefs)
        w = {k: float(np.mean(cf[:, i])) for i, k in enumerate(use)}
        sc = sum(abs(v) for v in w.values()) or 1.0
        w = {k: v / sc for k, v in w.items()}
        print("训练段回归权重:", {k: round(v, 3) for k, v in sorted(w.items(), key=lambda x: -abs(x[1]))})
        res.append(run(w, "训练段截面回归"))
    # 关键对照：回归权重原封不动拿到测试段用，和重估权重的结果比
    if coefs:
        res.append(run(w, "回归权重复用"))
        res.append(run({"size": 1.0, "amt20": 1.0, "turn": 1.0, "vol20": 1.0, "max20": 1.0}, "五因子等权"))
        res.append(run({"size": 1.0, "amt20": 1.0}, "size+amt20"))
        res.append(run({"size": 1.0, "amt20": 1.0, "pvcorr": 1.0}, "size+amt20+pvcorr"))
    json.dump(res, open(os.path.join(HERE, "mlfit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved mlfit.json")


if __name__ == "__main__":
    main()
