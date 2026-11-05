# -*- coding: utf-8 -*-
"""tca.py —— 机构口径交易成本分析 (Transaction Cost Analysis)

问题
----
multifactor.json 里 "ICIR_t12" 年化 22.69%，"ICIR+涨跌停+冲击" 年化 13.76%。
attrib.py 把这个 8.93pp 的差笼统叫做"成本吃掉 7.9 个点"，但那 8.93pp 里
其实混了三件性质完全不同的东西：

  1. 显式成本  佣金 / 印花税 / 过户费（真金白银，可分项记账）
  2. 冲击成本  .sqrt(订单占 ADV) 模型（只在买入侧扣了，卖出侧没扣）
  3. 实现缺口  limit_check 把涨停买不进的票剔除，改变了选股集合本身

更要紧的一条：multifactor.run() 的净值用 t 日收盘 C[:,j] -> C[:,jn] 结算，
等于 t 日收盘看到信号就按 t 日收盘价成交。engine.py 明明规定 t+1 开盘成交，
这里没遵守。本文件把"严格执行"单列成一个情景，量出这条前视值多少钱。

做法
----
一次性重建 197 个调仓日的因子打分与 ICIR 权重（与 multifactor.py 完全同口径，
先用 --calibrate 逐点校准净值曲线），然后在内存里按不同成本/执行假设重放，
每期落一条分项账：

  commission_buy / commission_sell / stamp_sell / transfer /
  impact_buy / impact_sell / turnover / npos

Wall Street 口径的两条硬规矩：
  - 冲击必须两侧对称（买卖都要付）
  - 成本按当期净值记 bps，且要把"规模随复利膨胀"与"单位成本假设"分开

输出 tca_result.json / tca_curves.json；不改动任何既有产物。
"""
import os, json, struct, mmap, time, datetime, sys, math

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import anomaly_race as AR
import ic_walkforward as IW

# 与 multifactor.py 完全相同，校准不过就说明复刻有偏差
FEE_BUY, FEE_SELL = 0.00025, 0.00075
COMM, STAMP, TRANSFER = 0.00025, 0.00050, 0.00001
NAMES = ["size", "ep", "bm", "volatility", "max", "turnover",
         "abnormal_turnover", "illiquidity"]
FIELDS = {"ep": "eps", "bm": "bps"}


def zscore(x):
    m = np.isfinite(x)
    if m.sum() < 30:
        return np.full_like(x, np.nan, dtype=np.float64)
    v = x[m].astype(np.float64)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    if mad <= 0:
        return np.full_like(x, np.nan, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[m] = (v - med) / mad
    return np.clip(out, -3, 3)


def monthly(curve):
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


class Panel:
    """一次建好，后面所有情景共享，避免重复读 1.5GB 面板。"""

    def __init__(self, start="2010-06-01", k=50):
        t0 = time.time()
        self.k = k
        self.C, ND, NC = AR.load_panel("m_close.bin")
        self.O, _, _ = AR.load_panel("m_open.bin")
        self.A, _, _ = AR.load_panel("amt_flat.bin")
        self.MK, _, _ = AR.load_panel("m_mktcap.bin")
        meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
        self.dates, self.codes = meta["dates"], meta["codes"]
        ci_of = {c: i for i, c in enumerate(self.codes)}
        self.ND, self.NC = ND, NC
        si = next(i for i, d in enumerate(self.dates) if d >= start)

        # 价格跳变清理（同 multifactor）
        bad = np.zeros((NC, ND), dtype=bool)
        for r0 in range(0, NC, 500):
            blk = self.C[r0:r0 + 500]
            prev, cur = blk[:, :-1], blk[:, 1:]
            good = (prev > 0) & (cur > 0)
            r = np.zeros(prev.shape, dtype=np.float32)
            np.divide(cur, np.where(good, prev, np.float32(1)), out=r, where=good)
            r -= 1
            jump = good & ((r > 0.21) | (r < -0.21))
            iso = jump.copy()
            iso[:, 1:] &= ~jump[:, :-1]
            iso[:, :-1] &= ~jump[:, 1:]
            bad[r0:r0 + 500, 1:] = iso
            del r, jump, iso, good, prev, cur
        self.badin = bad.copy()
        self.badin[:, 1:] |= bad[:, :-1]
        self.badin[:, :-1] |= bad[:, 1:]
        del bad
        import gc
        gc.collect()
        first_day = np.full(NC, ND, dtype=np.int32)
        for ci in range(NC):
            nz = np.where((self.C[ci] > 0) | (self.A[ci] > 0))[0]
            if len(nz):
                first_day[ci] = nz[0]
        self.first_day = first_day
        self.MIND = 250
        self.rebal = [j for j in range(si + 1, ND) if self.dates[j][:7] != self.dates[j - 1][:7]]
        print("[%5.1fs] 面板就绪：%d 期调仓 %s ~ %s" % (
            time.time() - t0, len(self.rebal), self.dates[self.rebal[0]], self.dates[self.rebal[-1]]), flush=True)
        self._build_factors(ci_of, t0)
        self._build_ic(t0)

    def live_at(self, j):
        return ((self.C[:, j] > 2.0) & (self.A[:, j] > 0) &
                ~self.badin[:, j] & (self.first_day + self.MIND <= j))

    def ret_slice(self, j0, j1):
        prev, cur = self.C[:, j0:j1 - 1], self.C[:, j0 + 1:j1]
        good = prev > 0
        out = np.full((self.NC, j1 - j0 - 1), np.nan, dtype=np.float32)
        np.divide(cur, np.where(good, prev, np.float32(1)), out=out, where=good)
        out[~good] = np.nan
        out -= 1
        return out

    def _build_factors(self, ci_of, t0):
        C, O, A, MK = self.C, self.O, self.A, self.MK
        NC, dates = self.NC, self.dates
        FV = {n: [] for n in NAMES}
        FWD = []
        FM = {}
        cur = {"f": None}

        def get(fld):
            if cur["f"] != fld:
                FM.clear()
                import gc
                gc.collect()
                M = AR.fund_matrix(self.codes, ci_of, dates, field=fld)
                FM[fld] = np.ascontiguousarray(M, dtype=np.float32)
                if M is not FM[fld]:
                    del M
                gc.collect()
                cur["f"] = fld
            return FM[fld]

        def factor_at(name, j):
            if name == "size":
                return -np.log(np.maximum(MK[:, j], 1))
            if name == "ep":
                return np.where(C[:, j] > 0, get("eps")[:, j] / np.where(C[:, j] > 0, C[:, j], np.nan), np.nan)
            if name == "bm":
                return np.where(C[:, j] > 0, get("bps")[:, j] / np.where(C[:, j] > 0, C[:, j], np.nan), np.nan)
            if name == "volatility":
                return -np.nanstd(self.ret_slice(max(0, j - 20), j + 1), axis=1).astype(np.float64)
            if name == "max":
                return -np.nanmax(self.ret_slice(max(0, j - 20), j + 1), axis=1).astype(np.float64)
            if name == "turnover":
                j0 = max(0, j - 20); m = MK[:, j0:j + 1]
                return -np.nanmean(A[:, j0:j + 1] / np.where(m > 0, m, np.nan), axis=1)
            if name == "abnormal_turnover":
                j0a = max(0, j - 20); ma = MK[:, j0a:j + 1]
                ta = np.nanmean(A[:, j0a:j + 1] / np.where(ma > 0, ma, np.nan), axis=1)
                j0b = max(0, j - 250); mb = MK[:, j0b:j + 1]
                tb = np.nanmean(A[:, j0b:j + 1] / np.where(mb > 0, mb, np.nan), axis=1)
                return -np.where(tb > 0, ta / np.where(tb > 0, tb, np.nan), np.nan)
            if name == "illiquidity":
                j0 = max(0, j - 19)
                r = self.ret_slice(j0, j + 1); amt = A[:, j0 + 1:j + 1]
                return np.nanmean(np.abs(r) / np.maximum(amt / 1e8, 1e-9), axis=1)
            raise KeyError(name)

        for n in NAMES:
            if n in FIELDS:
                cur["f"] = None; get(FIELDS[n])
            for j in self.rebal:
                FV[n].append(zscore(factor_at(n, j)).astype(np.float32))
            print("[%5.1fs]   %s 算完" % (time.time() - t0, n), flush=True)
        self.FV = FV

        rebal = self.rebal
        for ri in range(len(rebal)):
            jn = rebal[ri]
            j2 = rebal[ri + 1] if ri + 1 < len(rebal) else ND_LAST
            if j2 <= jn:
                FWD.append(np.full(NC, np.nan)); continue
            ok = (O[:, jn] > 0) & (O[:, j2] > 0)
            f = np.full(NC, np.nan)
            f[ok] = O[ok, j2] / O[ok, jn] - 1
            FWD.append(f)
        self.FWD = FWD
        self.ICmat = np.full((len(NAMES), len(rebal)), np.nan)
        for ni, n in enumerate(NAMES):
            for ri, j in enumerate(rebal):
                f, y = FV[n][ri], FWD[ri]
                valid = self.live_at(j) & np.isfinite(y) & np.isfinite(f)
                if valid.sum() < 200:
                    continue
                self.ICmat[ni, ri] = IW.spearman(np.where(valid, f, np.nan), np.where(valid, y, np.nan))
        print("[%5.1fs] IC 矩阵算完" % (time.time() - t0), flush=True)

    def _build_ic(self, t0):
        # 占位：IC 在 _build_factors 里算完，保留接口清晰
        pass

    def weights_at(self, ri, win=12):
        h = self.ICmat[:, max(0, ri - win):ri]
        w = np.zeros(len(NAMES))
        for ni in range(len(NAMES)):
            v = h[ni][np.isfinite(h[ni])]
            if len(v) < 6:
                continue
            sd = v.std(ddof=1)
            if sd > 0:
                w[ni] = v.mean() / sd
        w = np.where(w > 0, w, 0)
        s = w.sum()
        return w / s if s > 0 else np.ones(len(NAMES)) / len(NAMES)

    def scores(self, ri):
        w = self.weights_at(ri)
        score = np.zeros(self.NC)
        cnt = np.zeros(self.NC)
        for ni, n in enumerate(NAMES):
            if w[ni] == 0:
                continue
            f = self.FV[n][ri]
            ok = np.isfinite(f)
            score[ok] += w[ni] * f[ok]
            cnt[ok] += w[ni]
        return np.where(cnt > 0, score / np.maximum(cnt, 1e-12), np.nan), w

    def target(self, ri, limit_check=False):
        j = self.rebal[ri]
        score, w = self.scores(ri)
        valid = self.live_at(j) & np.isfinite(score)
        if valid.sum() < self.k:
            return np.array([], dtype=int), w
        idx = np.where(valid)[0]
        order = idx[np.argsort(-score[idx])]
        if limit_check and j > 0:
            pc = self.C[:, j - 1]
            chg = np.where(pc > 0, self.C[:, j] / np.where(pc > 0, pc, 1) - 1, 0)
            tradable = (chg < 0.095) & (chg > -0.095)
            order = np.array([c for c in order if tradable[c]], dtype=int)
        return order, w

    def run(self, *, fee_on=True, impact=0.0, sym_impact=False, limit_check=False,
            exec_open=False, buffer=0.0, hold=1, fixed_capital=False, label=""):
        """重放一次，返回净值曲线与逐期分项账。"""
        C, O, A = self.C, self.O, self.A
        k, rebal, dates = self.k, self.rebal, self.dates
        ND = self.ND
        nav, start_nav = 1.0e6, 1.0e6
        curve, ledger = [], []
        pos = set()
        for ri in range(len(rebal)):
            j = rebal[ri]
            jn = rebal[ri + 1] if ri + 1 < len(rebal) else min(j + 21, ND - 1)

            if ri % hold == 0:
                order, w = self.target(ri, limit_check=limit_check)
                if buffer > 0 and len(order) and pos:
                    band = int(k * (1 + buffer))
                    keep_zone = set(order[:band].tolist())
                    keep = [c for c in pos if c in keep_zone][:k]
                    have = set(keep)
                    fill = [int(c) for c in order if int(c) not in have][:k - len(keep)]
                    newpos = set(keep + fill)
                else:
                    newpos = set(order[:k].tolist())
            else:
                newpos = set(pos)

            sells = pos - newpos
            buys = newpos - pos
            n_new, n_old = max(len(newpos), 1), max(len(pos), 1)
            stake_b = nav / n_new
            stake_s = nav / n_old
            rec = {"date": dates[j], "npos": len(newpos),
                   "turnover": round((len(buys) + len(sells)) / (2.0 * k), 4),
                   "nav_before": round(nav, 2)}
            c_buy = c_sell = stamp = transfer = imp_b = imp_s = 0.0
            if fee_on:
                for c in buys:
                    c_buy += stake_b * COMM
                    transfer += stake_b * TRANSFER
                for c in sells:
                    c_sell += stake_s * COMM
                    stamp += stake_s * STAMP
                    transfer += stake_s * TRANSFER
            if impact > 0:
                base = start_nav if fixed_capital else nav
                for c in buys:
                    adv = A[c, j] if A[c, j] > 0 else 1e6
                    stake = base / n_new
                    imp_b += stake * impact * math.sqrt(min(stake / adv, 0.3))
                if sym_impact:
                    for c in sells:
                        adv = A[c, j] if A[c, j] > 0 else 1e6
                        stake = base / n_old
                        imp_s += stake * impact * math.sqrt(min(stake / adv, 0.3))
            fees = c_buy + c_sell + stamp + transfer + imp_b + imp_s
            nav -= fees
            rec.update({"commission_buy": round(c_buy, 2), "commission_sell": round(c_sell, 2),
                        "stamp": round(stamp, 2), "transfer": round(transfer, 2),
                        "impact_buy": round(imp_b, 2), "impact_sell": round(imp_s, 2),
                        "fees": round(fees, 2),
                        "cost_bps": round(fees / max(nav + fees, 1) * 1e4, 2)})
            pos = newpos
            if jn > j and len(pos):
                if exec_open:
                    # t 日收盘出信号 -> t+1 开盘买 -> 下个调仓日的 t+1 开盘卖
                    b = min(j + 1, ND - 1); s = min(jn + 1, ND - 1)
                    if s > b:
                        rr = np.array([O[c, s] / O[c, b] - 1 if O[c, b] > 0 and O[c, s] > 0 else 0.0
                                       for c in pos])
                    else:
                        rr = np.zeros(len(pos))
                else:
                    rr = np.array([C[c, jn] / C[c, j] - 1 if C[c, j] > 0 and C[c, jn] > 0 else 0.0
                                   for c in pos])
                nav *= (1 + rr.mean())
            curve.append([dates[j], round(float(nav), 2)])
            rec["nav_after"] = round(float(nav), 2)
            ledger.append(rec)

        sv = np.array([x[1] for x in curve])
        d0 = datetime.date(*map(int, dates[rebal[0]].split("-")))
        d1 = datetime.date(*map(int, dates[rebal[-1]].split("-")))
        yrs = max((d1 - d0).days / 365.25, 1 / 365)
        cagr = (sv[-1] / sv[0]) ** (1 / yrs) - 1
        dr = sv[1:] / sv[:-1] - 1
        sharpe = dr.mean() / dr.std(ddof=1) * math.sqrt(12) if dr.std() > 0 else 0
        peak = np.maximum.accumulate(sv); mdd = float((sv / peak - 1).min())
        tot_fees = sum(x["fees"] for x in ledger)
        return {"label": label, "cagr": round(cagr * 100, 2), "sharpe": round(float(sharpe), 3),
                "mdd": round(mdd * 100, 1), "total": round(float(sv[-1] / sv[0]), 2),
                "curve": curve, "ledger": ledger,
                "fees_total": round(tot_fees, 2),
                "fees_by": {kk: round(sum(x[kk] for x in ledger), 2)
                            for kk in ("commission_buy", "commission_sell", "stamp",
                                       "transfer", "impact_buy", "impact_sell")},
                "turnover_avg": round(float(np.mean([x["turnover"] for x in ledger])), 4),
                "cost_bps_avg": round(float(np.mean([x["cost_bps"] for x in ledger])), 2)}


    def run_fixed(self, *, aum=1.0e6, impact=0.1, cap_ratio=0.0, limit_check=True,
                  exec_open=True, label=""):
        """机构口径：AUM 外生给定（不随 PnL 复利膨胀），每期做固定名义再平衡。

        与 run() 的区别是这个问题的答案：一个策略从 100 万赚到 3400 万，
        管理人不会把 3400 万全按 50 只等权砸进去。真实世界要么把利润赎回
        （AUM 固定），要么受单票 ADV 容量上限约束（多余的钱留现金）。

        每期：
          目标名义 = min(aum / k, cap_ratio * ADV_c)   （cap_ratio=0 表示不限）
          未投出去的部分留现金，零收益
          收益率 = sum(目标名义 * 个股收益) / aum - 费用 / aum
          净值 = aum + 累计盈亏（算术，不是复利）
        这样"策略越赚，冲击越贵"的伪反馈被切断，冲击系数只由参与率决定。
        """
        C, O, A = self.C, self.O, self.A
        k, rebal, dates = self.k, self.rebal, self.dates
        ND = self.ND
        nav = aum
        curve, ledger = [], []
        pos = set()
        prev_notional = {}
        for ri in range(len(rebal)):
            j = rebal[ri]
            jn = rebal[ri + 1] if ri + 1 < len(rebal) else min(j + 21, ND - 1)
            order, w = self.target(ri, limit_check=limit_check)
            newpos = set(order[:k].tolist())
            notional = {}
            for c in newpos:
                adv = A[c, j] if A[c, j] > 0 else 0.0
                v = aum / max(k, 1)
                if cap_ratio > 0 and adv > 0:
                    v = min(v, cap_ratio * adv)
                notional[c] = v
            buys = newpos - pos
            sells = pos - newpos
            c_buy = c_sell = stamp = transfer = imp_b = imp_s = 0.0
            for c in buys:
                v = notional[c]
                c_buy += v * COMM
                transfer += v * TRANSFER
                adv = A[c, j] if A[c, j] > 0 else 1e6
                imp_b += v * impact * math.sqrt(min(v / adv, 0.3))
            for c in sells:
                v = prev_notional.get(c, aum / max(k, 1))
                c_sell += v * COMM
                stamp += v * STAMP
                transfer += v * TRANSFER
                adv = A[c, j] if A[c, j] > 0 else 1e6
                imp_s += v * impact * math.sqrt(min(v / adv, 0.3))
            fees = c_buy + c_sell + stamp + transfer + imp_b + imp_s
            pnl = 0.0
            if jn > j and newpos:
                if exec_open:
                    b = min(j + 1, ND - 1); s2 = min(jn + 1, ND - 1)
                    for c in newpos:
                        if s2 > b and O[c, b] > 0 and O[c, s2] > 0:
                            pnl += notional[c] * (O[c, s2] / O[c, b] - 1)
                else:
                    for c in newpos:
                        if C[c, j] > 0 and C[c, jn] > 0:
                            pnl += notional[c] * (C[c, jn] / C[c, j] - 1)
            invested = sum(notional.values())
            ret = (pnl - fees) / aum
            nav += pnl - fees
            curve.append([dates[j], round(float(nav), 2)])
            ledger.append({"date": dates[j], "npos": len(newpos),
                           "invested": round(invested, 2),
                           "capacity_util": round(invested / aum, 4),
                           "fees": round(fees, 2),
                           "cost_bps": round(fees / aum * 1e4, 2),
                           "pnl": round(pnl, 2), "ret": round(ret, 6),
                           "turnover": round((len(buys) + len(sells)) / (2.0 * k), 4),
                           "nav": round(float(nav), 2),
                           "by": {"commission_buy": round(c_buy, 2),
                                  "commission_sell": round(c_sell, 2),
                                  "stamp": round(stamp, 2), "transfer": round(transfer, 2),
                                  "impact_buy": round(imp_b, 2), "impact_sell": round(imp_s, 2)}})
            pos = newpos
            prev_notional = notional

        sv = np.array([x[1] for x in curve])
        r = np.diff(sv) / aum
        d0 = datetime.date(*map(int, dates[rebal[0]].split("-")))
        d1 = datetime.date(*map(int, dates[rebal[-1]].split("-")))
        yrs = max((d1 - d0).days / 365.25, 1 / 365)
        # 算术净值（AUM 固定 + 利润赎回）不是复利基数，负值时 (end/start)**(1/yrs) 无定义。
        # 这里用"月均算术收益 / AUM"年化，口径与净值曲线一致；复利口径见 total_multiple。
        ann_arith = float(r.mean() * 12)
        sharpe = r.mean() / r.std(ddof=1) * math.sqrt(12) if r.std() > 0 else 0
        return {"label": label, "aum": aum, "impact": impact, "cap_ratio": cap_ratio,
                "cagr": round(ann_arith * 100, 2), "sharpe": round(float(sharpe), 3),
                "total": round(float(sv[-1] / sv[0]), 2),
                "cagr_basis": "arithmetic_monthly_on_fixed_AUM",
                "curve": curve, "ledger": ledger,
                "fees_total": round(sum(x["fees"] for x in ledger), 2),
                "fees_by": {kk: round(sum(x["by"][kk] for x in ledger), 2)
                            for kk in ("commission_buy", "commission_sell", "stamp",
                                       "transfer", "impact_buy", "impact_sell")},
                "capacity_util_avg": round(float(np.mean([x["capacity_util"] for x in ledger])), 4),
                "cost_bps_avg": round(float(np.mean([x["cost_bps"] for x in ledger])), 2),
                "turnover_avg": round(float(np.mean([x["turnover"] for x in ledger])), 4)}


# 模块级常量，供 _build_factors 的 FWD 末端使用
ND_LAST = 8815


def alpha_of(curve, bm_map, fixed_aum=None):
    """OLS 归因。fixed_aum 非 None 时，curve 是"固定本金 + 累计盈亏"的算术净值，
    月度收益必须用 差分/aum，不能再用比例；同时把基准也换成同口径的月度差分，
    否则拿一个比例序列去回归一个算术序列，beta 和 alpha 都没有意义。"""
    if fixed_aum is not None:
        # 取每月最后一个净值点
        last = {}
        for d, v in curve:
            last[d[:7]] = float(v)
        keys = sorted(last)
        vals = np.array([last[k] for k in keys], dtype=float)
        r_arith = np.diff(vals) / float(fixed_aum)
        mk = sorted(bm_map)
        # 基准用同窗口的月度比例收益；对固定本金账户，等价的名义收益就是 mkt * aum
        y = np.array([])
        mkt = np.array([])
        ky = []
        for i, kk in enumerate(keys[1:]):
            if kk in bm_map:
                ky.append(i)
        y = r_arith[ky]
        mkt = np.array([bm_map[keys[1:][i]] for i in ky])
        if len(y) < 24:
            return {}
        X = np.column_stack([np.ones(len(y)), mkt])
        beta, tvals, r2 = newey_west(y, X)
        a = float(beta[0])
        return {"months": len(y),
                "alpha_annual": round(a * 12 * 100, 2),
                "alpha_t": round(float(tvals[0]), 2),
                "beta": round(float(beta[1]), 3), "r2": round(r2, 3),
                "excess_annual": round(float(y.mean() * 12 * 100), 2),
                "basis": "arithmetic_on_fixed_AUM"}
    keys, rp = monthly(curve)
    idx = [i for i, kk in enumerate(keys) if kk in bm_map]
    if len(idx) < 24:
        return {}
    y = np.array([rp[i] for i in idx])
    mkt = np.array([bm_map[keys[i]] for i in idx])
    X = np.column_stack([np.ones(len(y)), mkt])
    beta, tvals, r2 = newey_west(y, X)
    a = float(beta[0])
    return {"months": len(y),
            "alpha_annual": round(((1 + a) ** 12 - 1) * 100, 2),
            "alpha_t": round(float(tvals[0]), 2),
            "beta": round(float(beta[1]), 3), "r2": round(r2, 3),
            "excess_annual": round((np.prod(1 + y) ** (12 / len(y)) - 1) * 100, 2)}


def main():
    cal = "--calibrate" in sys.argv
    P = Panel()
    bench = json.load(open(os.path.join(HERE, "bench_all_a_clean.json"), encoding="utf-8"))
    bk, br = monthly(bench)
    bm_map = dict(zip(bk, br))
    mf = json.load(open(os.path.join(HERE, "multifactor.json"), encoding="utf-8"))

    scen = [
        ("S0_gross",          dict(fee_on=False, impact=0.0, label="毛收益：零成本、收盘成交")),
        ("S1_explicit",       dict(fee_on=True, impact=0.0, label="仅佣金+印花税+过户费")),
        ("S2_buy_impact",     dict(fee_on=True, impact=0.1, label="+买入侧冲击(原口径)")),
        ("S3_limit_impact",   dict(fee_on=True, impact=0.1, limit_check=True, label="+涨跌停过滤(复刻原13.76%)")),
        ("S4_sym_impact",     dict(fee_on=True, impact=0.1, sym_impact=True, limit_check=True, label="+卖出侧也付冲击(机构口径)")),
        ("S5_exec_open",      dict(fee_on=True, impact=0.1, sym_impact=True, limit_check=True, exec_open=True, label="+t+1开盘成交(去掉前视)")),
        ("S6_fixed_capital",  dict(fee_on=True, impact=0.1, sym_impact=True, limit_check=True, exec_open=True, fixed_capital=True, label="+按初始规模计冲击(隔离复利膨胀)")),
    ]
    if cal:
        scen = [("CAL_base", dict(fee_on=True, impact=0.0, label="校准：复刻 ICIR_t12")),
                ("CAL_limit", dict(fee_on=True, impact=0.1, limit_check=True, label="校准：复刻 ICIR+涨跌停+冲击"))]

    out, curves, ledgers = {}, {}, {}
    for name, kw in scen:
        t0 = time.time()
        r = P.run(**kw)
        r["alpha"] = alpha_of(r["curve"], bm_map)
        out[name] = {kk: vv for kk, vv in r.items() if kk not in ("curve", "ledger")}
        curves[name] = r["curve"]
        ledgers[name] = r["ledger"]
        a = r["alpha"]
        print("%-18s CAGR %6.2f%%  alpha %6.2f%% t=%5.2f  换手 %.3f  费 %.0f  [%.1fs]" % (
            name, r["cagr"], a.get("alpha_annual", 0), a.get("alpha_t", 0),
            r["turnover_avg"], r["fees_total"], time.time() - t0), flush=True)

    # 校准：与既有产物逐点核对
    if cal:
        for name, ref in (("CAL_base", "ICIR_t12"), ("CAL_limit", "ICIR+涨跌停+冲击")):
            a = [x[1] for x in curves[name]]
            b = [x[1] for x in mf[ref]["curve"]]
            n = min(len(a), len(b))
            dif = [abs(a[i] - b[i]) / max(b[i], 1) for i in range(n)]
            print("CAL %-10s vs %-20s  n=%d  最大相对误差 %.6f  末值 %.2f vs %.2f  CAGR %.2f vs %.2f" % (
                name, ref, n, max(dif), a[-1], b[-1], out[name]["cagr"], mf[ref]["cagr"]), flush=True)
        return 0

    # 降换手前沿：在最现实的情景 S5 上叠加 no-trade band / 拉长持有期
    frontier = []
    for buf, hold, tag in [(0.0, 1, "baseline"), (0.25, 1, "band25"), (0.5, 1, "band50"),
                           (1.0, 1, "band100"), (0.5, 2, "band50_h2"),
                           (0.5, 3, "band50_h3"), (1.0, 3, "band100_h3"),
                           (1.0, 6, "band100_h6")]:
        r = P.run(fee_on=True, impact=0.1, sym_impact=True, limit_check=True,
                  exec_open=True, buffer=buf, hold=hold, label=tag)
        al = alpha_of(r["curve"], bm_map)
        frontier.append({"tag": tag, "buffer": buf, "hold_months": hold,
                         "cagr": r["cagr"], "alpha_annual": al.get("alpha_annual"),
                         "alpha_t": al.get("alpha_t"), "turnover": r["turnover_avg"],
                         "cost_bps": r["cost_bps_avg"], "fees_by": r["fees_by"]})
        print("前沿 %-14s CAGR %6.2f%%  alpha %6.2f%% t=%5.2f  换手 %.3f" % (
            tag, r["cagr"], al.get("alpha_annual", 0), al.get("alpha_t", 0), r["turnover_avg"]), flush=True)
        curves["F_" + tag] = r["curve"]

    # 冲击系数敏感性 + AUM 固定 / 单票容量约束（机构真实口径）
    sens, fixed = [], {}
    for eta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5):
        r = P.run_fixed(aum=1.0e6, impact=eta, label="eta_%.2f" % eta)
        al = alpha_of(r["curve"], bm_map, fixed_aum=1.0e6)
        sens.append({"eta": eta, "cagr": r["cagr"], "sharpe": r["sharpe"],
                     "alpha_annual": al.get("alpha_annual"), "alpha_t": al.get("alpha_t"),
                     "cost_bps": r["cost_bps_avg"], "fees_by": r["fees_by"]})
        print("敏感 eta %.2f  CAGR %6.2f%%  alpha %6.2f%% t=%5.2f  bps %.2f" % (
            eta, r["cagr"], al.get("alpha_annual", 0), al.get("alpha_t", 0), r["cost_bps_avg"]), flush=True)
        curves["ETA_%.2f" % eta] = r["curve"]
        fixed["eta_%.2f" % eta] = r
    for aum in (1.0e6, 5.0e6, 2.0e7, 1.0e8):
        for cap in (0.0, 0.10, 0.02):
            tag = "aum%d_cap%02d" % (int(aum / 1e6), int(cap * 100))
            r = P.run_fixed(aum=aum, impact=0.1, cap_ratio=cap, label=tag)
            al = alpha_of(r["curve"], bm_map, fixed_aum=aum)
            sens.append({"kind": "aum", "tag": tag, "aum": aum, "cap_ratio": cap,
                         "cagr": r["cagr"], "alpha_annual": al.get("alpha_annual"),
                         "alpha_t": al.get("alpha_t"), "cost_bps": r["cost_bps_avg"],
                         "capacity_util": r["capacity_util_avg"]})
            print("容量 %-16s CAGR %6.2f%%  alpha %6.2f%% t=%5.2f  投入率 %.3f  bps %.2f" % (
                tag, r["cagr"], al.get("alpha_annual", 0), al.get("alpha_t", 0),
                r["capacity_util_avg"], r["cost_bps_avg"]), flush=True)
            curves["FIX_" + tag] = r["curve"]
            fixed[tag] = r

    json.dump({"scenarios": out, "frontier": frontier, "sensitivity": sens,
               "fixed_aum": {kk: {x: y for x, y in vv.items() if x not in ("curve", "ledger")}
                             for kk, vv in fixed.items()},
               "generated": datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(HERE, "tca_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(curves, open(os.path.join(HERE, "tca_curves.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(ledgers, open(os.path.join(HERE, "tca_ledger.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print("\n已写 tca_result.json / tca_curves.json / tca_ledger.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
