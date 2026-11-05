# -*- coding: utf-8 -*-
"""strategy_layer.py —— 把两条收益叙事合并到同一套真实面板、成本与归因上。

历史问题：README 的 EP 叙事（CAGR 24.14%/超额 17.45pp）、tca 的
22.7% -> alpha 2.3% t=0.70、mlfit2 的 size+amt20 月度 1.27% t4.66，
三者用的是不同面板、不同成本、不同基准、不同执行日，互相无法比较。
本文件把它们放进**同一条管线**：

  面板    m_compact.bin（5251 只 × 8816 日，close/open/amt/mktcap，真实市值）
  截面    fscan_parts 的 166 个唯一月末（fraw.meta_unique，已修重复落盘）
  信号    size+amt20 双因子 / EP(PIT 财务) / 全池等权基准 / 分位锚点
  执行    决策日 S 收盘出信号 -> 下一交易日 E 开盘成交；持有到下一 E 开盘换仓
          逐日 buy-and-hold 计价，停牌沿用最后价，退市归零
  成本    quantlab.model.cost.compute_cost（佣金/过户/印花税分段/校准冲击）
          冲击系数取自 execution_cal.json 的 5 分钟线校准，与 tca 同口径
  风控    quantlab.execution.PreTradeRisk（单票/行业/参与率/涨跌停/现金/整手/T+1）
  撮合    quantlab.execution.PaperBroker + reconcile 逐笔对账
  归因    MKT/SMB/HML 自建可交易组合 + Newey-West HAC t 值

输出 unified_result.json / order_plan_<cfg>.jsonl / fills_<cfg>.jsonl / STRATEGY_LAYER.md
"""
from __future__ import annotations
import os, sys, json, math, struct, bisect, datetime
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np
np.seterr(all="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fraw
from quantlab.config import Config, CostConfig, ExecutionConfig, BacktestConfig
from quantlab.model.cost import compute_cost, max_notional
from quantlab.execution import OrderPlan, PreTradeRisk, OrderRecord, FillRecord, reconcile
from quantlab.execution.broker import PaperBroker, fill_is_bps
from quantlab.execution.oms import order_plan_fingerprint
from quantlab.data.pit import disclosure_deadline
from dataclasses import replace, asdict

np.random.seed(20261105)


class RealPanel:
    """m_compact.bin 的按 [日期 x 股票] 顺序读，不缓存、不 mmap 整文件。"""

    FIELDS = ("close", "amt", "open", "mktcap")

    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, "m_compact.bin")
        self.f = open(self.path, "rb", buffering=0)
        head = self.f.read(16)
        assert head[:8] == b"CMPC0001", "不是 compact 面板"
        self.ND, self.NC = struct.unpack("<II", head[8:16])
        meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
        self.dates, self.codes = meta["dates"], meta["codes"]
        assert len(self.dates) == self.ND and len(self.codes) == self.NC
        self.code_idx = {c: i for i, c in enumerate(self.codes)}
        self.date_idx = {d: i for i, d in enumerate(self.dates)}
        self.bad = np.memmap(os.path.join(HERE, "corrupt_mask.bin"), dtype=np.uint8,
                            mode="r", shape=(self.ND, self.NC))
        self.halt = np.memmap(os.path.join(HERE, "halt_mask.bin"), dtype=np.uint8,
                             mode="r", shape=(self.ND, self.NC))

    def rows(self, field, j0, j1):
        n = (j1 - j0) * self.NC
        off = 16 + self.FIELDS.index(field) * self.ND * self.NC * 4 + j0 * self.NC * 4
        self.f.seek(off)
        raw = self.f.read(n * 4)
        assert len(raw) == n * 4, (field, j0, j1, len(raw))
        return np.frombuffer(raw, dtype="<f4", count=n).reshape(j1 - j0, self.NC).astype(np.float64)

    def col(self, field, j):
        return self.rows(field, j, j + 1)[0]


def zscore_row(v):
    ok = np.isfinite(v) & (np.abs(v) < 1e12)
    if ok.sum() < 200:
        return np.full(v.shape, np.nan)
    med = float(np.median(v[ok])); sd = float(v[ok].std())
    return np.where(ok, (v - med) / (sd if sd > 0 else np.nan), np.nan)


def nw_t(y, X, lags=None):
    """OLS + Newey-West HAC，返回 (beta, t, r2)。"""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    L = lags or max(1, int(round(4 * (n / 100.0) ** (2.0 / 9.0))))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    r2 = 1.0 - float(e @ e) / max(float(((y - y.mean()) ** 2).sum()), 1e-30)
    return beta, beta / se, r2


def stats_of(curve):
    """curve: [(date, nav)] -> 年化/夏普/最大回撤（日频）。"""
    if len(curve) < 3:
        return {}
    nav = np.array([v for _, v in curve], dtype=float)
    r = nav[1:] / nav[:-1] - 1.0
    days = (datetime.date.fromisoformat(curve[-1][0]) - datetime.date.fromisoformat(curve[0][0])).days
    yrs = max(days / 365.25, 1e-9)
    cagr = (nav[-1] / nav[0]) ** (1.0 / yrs) - 1.0
    ann_vol = float(r.std(ddof=1)) * math.sqrt(250) if len(r) > 2 else 0.0
    sharpe = float(r.mean()) / float(r.std(ddof=1)) * math.sqrt(250) if len(r) > 2 and r.std(ddof=1) > 0 else None
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    return {"cagr_pct": round(cagr * 100, 2), "sharpe": round(sharpe, 3) if sharpe is not None else None,
            "ann_vol_pct": round(ann_vol * 100, 2),
            "mdd_pct": round(mdd * 100, 2), "total_mult": round(float(nav[-1] / nav[0]), 3),
            "n_days": len(curve)}


def monthly_returns(curve):
    last = {}
    for d, v in curve:
        last[d[:7]] = float(v)
    ks = sorted(last)
    v = np.array([last[k] for k in ks])
    return ks[1:], v[1:] / v[:-1] - 1.0


def load_context():
    rp = RealPanel()
    M = fraw.meta_unique()
    JJ = np.array([int(x) for x in M["dates"]], dtype=np.int64)   # 决策日 S
    NS = len(JJ) - 1
    EX = JJ[:NS] + 1                                             # 执行日 E = S+1
    assert np.all(EX[1:] > EX[:-1])
    icj = json.load(open(os.path.join(HERE, "ic_fast.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(HERE, "execution_cal.json"), encoding="utf-8"))
    return rp, M, JJ, EX, NS, icj, cal


def build_signals(rp, M, JJ, NS, icj):
    """返回 {name: (NS, NC) 打分矩阵}，越大越好；NaN = 不可选。"""
    NC = rp.NC
    out = {}
    Z = {}
    for k in ("size", "amt20"):
        A = fraw.rows(k, 0, NS, M).astype(np.float64)
        d = -1.0 if icj[k]["ic_mean"] < 0 else 1.0
        for m in range(NS):
            row = A[m] * d
            row[~rp.clean[int(JJ[m])]] = np.nan   # 质量闸门：坏样本不参与打分
            A[m] = zscore_row(row)
        Z[k] = A
    out["size_amt20"] = (Z["size"] + Z["amt20"]) / 2.0
    out["size"] = Z["size"]
    out["amt20"] = Z["amt20"]

    # EP：PIT 财务，公告可见性用 quantlab.data.pit.disclosure_deadline
    fund = json.load(open(os.path.join(HERE, "fundamentals.json"), encoding="utf-8"))
    epi = rp.code_idx
    EP = np.full((NS, NC), np.nan)
    per_code = {}
    for code, rows in fund.items():
        ci = epi.get(code)
        if ci is None:
            continue
        rec = []
        for r in rows:
            try:
                ann = disclosure_deadline(r["d"])
            except (ValueError, KeyError):
                continue
            if r.get("eps") is None or r.get("bps") is None:
                continue
            rec.append((ann, r["d"], float(r["eps"]), float(r["bps"])))
        rec.sort()
        per_code[ci] = rec
    for m in range(NS):
        d = rp.dates[int(JJ[m])]
        close = rp.col("close", int(JJ[m]))
        for ci, rec in per_code.items():
            if not rec or close[ci] <= 0:
                continue
            vis = [x for x in rec if x[0] <= d]
            if not vis:
                continue
            ann, period, eps, bps = vis[-1]
            byperiod = {x[1]: x for x in vis}
            y = int(period[:4]); mmdd = period[5:]
            if period.endswith("-12-31"):
                ttm = eps
            else:
                pa, ps = byperiod.get(f"{y-1}-12-31"), byperiod.get(f"{y-1}-{mmdd}")
                ttm = eps + pa[2] - ps[2] if pa and ps else eps
            EP[m, ci] = ttm / close[ci]
    for m in range(NS):
        EP[m][~rp.clean[int(JJ[m])]] = np.nan
        EP[m] = zscore_row(EP[m])
    out["ep"] = EP
    return out


def flags_at(rp, S, E, idxs, limit_main=0.10, limit_gem=0.20):
    """执行日开盘的涨跌停/停牌近似，以及决策价。没有 high/low，用开盘触板近似，偏保守。
    idxs 是股票的整数列号。"""
    cs = rp.col("close", S); op = rp.col("open", E); am = rp.col("amt", E)
    out = {}
    for i in idxs:
        i = int(i)
        lim = limit_gem if (rp.codes[i][:3] in ("300", "301", "688")) else limit_main
        px_prev = float(cs[i]) if np.isfinite(cs[i]) else 0.0
        px = float(op[i]) if np.isfinite(op[i]) else 0.0
        ratio = px / px_prev - 1.0 if px_prev > 0 and px > 0 else 0.0
        bad = bool(rp.bad[E, i]) or (not rp.clean[S][i])
        out[i] = {"decision_price": px_prev, "suspended": bool(px <= 0 or am[i] <= 0 or bad),
                  "limit_up": bool(ratio >= lim - 0.005), "limit_down": bool(ratio <= -lim + 0.005)}
    return out


def market_and_style_factors(rp, M, JJ, NS):
    """同池、同决策日、同执行日的可交易风格基准（不依赖任何策略净值）。

    MKT  全池等权，E 开盘 -> 下一 E 开盘
    SMB  小市值 30% 等权 - 大市值 30% 等权
    HML  高 E/P 30% 等权 - 低 E/P 30% 等权（价值的代理，明确标注）
    """
    fund = json.load(open(os.path.join(HERE, "fundamentals.json"), encoding="utf-8"))
    epi = rp.code_idx
    recs = {}
    for code, rows in fund.items():
        ci = epi.get(code)
        if ci is None:
            continue
        rs = []
        for r in rows:
            try:
                ann = disclosure_deadline(r["d"])
            except (ValueError, KeyError):
                continue
            if r.get("eps") is not None:
                rs.append((ann, float(r["eps"])))
        rs.sort()
        recs[ci] = rs
    ep_ann = {ci: [x[0] for x in rr] for ci, rr in recs.items()}
    keys, mk, smb, hml = [], [], [], []
    for m in range(NS):
        S, E = int(JJ[m]), int(JJ[m]) + 1
        E2 = int(JJ[m + 1]) + 1 if m + 1 < NS else min(S + 21, rp.ND - 1)
        if E2 <= E:
            continue
        op0 = rp.col("open", E); op1 = rp.col("open", E2)
        cS = rp.col("close", S); cap = rp.col("mktcap", S)
        alive = np.array(fraw.row("alive", m, M), dtype=bool)
        clean = rp.clean[S]
        ok = alive & clean & np.isfinite(op0) & (op0 > 0) & np.isfinite(op1) & (op1 > 0) & (cS > 2)
        good = ok & np.isfinite(cap) & (cap > 0)
        if good.sum() < 200:
            continue
        ret = op1[good] / op0[good] - 1.0
        ret = np.clip(ret, -0.5, 0.5)
        keys.append(rp.dates[E][:7])
        gidx = np.flatnonzero(good)
        mk.append(float(ret.mean()))
        c = cap[good]
        q = np.quantile(c, [0.3, 0.7])
        smb.append(float(ret[c <= q[0]].mean() - ret[c >= q[1]].mean()))
        # E/P：可见最新一期 EPS / 决策日收盘（bisect 定位，避免逐股列表推导）
        ep = np.full(len(gidx), np.nan)
        d = rp.dates[S]
        for t, ci in enumerate(gidx):
            rr = recs.get(int(ci), ())
            if not rr or cS[ci] <= 0:
                continue
            lo = bisect.bisect_right(ep_ann[int(ci)], d) - 1
            if lo >= 0:
                ep[t] = rr[lo][1] / cS[ci]
        v = np.isfinite(ep)
        if v.sum() > 100:
            qq = np.quantile(ep[v], [0.3, 0.7])
            hml.append(float(ret[v][ep[v] >= qq[1]].mean() - ret[v][ep[v] <= qq[0]].mean()))
        else:
            hml.append(0.0)
    return keys, np.array(mk), np.array(smb), np.array(hml)


def run_config(name, score, rp, M, JJ, EX, NS, cfg, *, aum, top_k=50,
               quintile=False, cap_w=0.04, cap_ind=0.20, industries=None,
               risk_on=True, write_ledger=False):
    """一条完整管线：信号 -> 目标 -> 订单 -> 风控 -> 撮合 -> 逐日净值 -> 对账。"""
    ind = industries or {}
    NC = rp.NC
    cal_eta = 0.00975
    cost_cfg = replace(cfg.cost, use_impact_model=True, impact_eta=1.0, impact_sigma=cal_eta,
                       adv_cap=0.10)
    cfg = replace(cfg, cost=cost_cfg, backtest=replace(cfg.backtest, cash=float(aum)))
    risk = PreTradeRisk(cfg, max_weight=cap_w, max_industry_weight=cap_ind,
                        max_participation=0.10)
    broker = PaperBroker()
    cash = float(aum)
    shares: dict[int, int] = {}
    last_px: dict[int, float] = {}
    curve = []
    all_plan, all_fills, rejects = [], [], {}
    cost_sum = {"commission": 0.0, "transfer": 0.0, "stamp": 0.0,
                "slippage": 0.0, "impact": 0.0, "total": 0.0}
    turn_sum = 0.0
    is_list = []
    recon_bad = 0
    delisted = 0
    dead: set[int] = set()          # 一旦数据不可信，永久按 0 计价，不许复活

    for m in range(NS):
        S, E = int(JJ[m]), int(EX[m])
        E2 = int(EX[m + 1]) if m + 1 < NS else min(S + 21, rp.ND - 1)
        cS = rp.col("close", S); mcap = rp.col("mktcap", S)
        amtS = rp.col("amt", S)
        alive = np.array(fraw.row("alive", m, M), dtype=bool)
        clean = rp.clean[S]
        pool = alive & clean & np.isfinite(cS) & (cS > cfg.execution.min_price) & np.isfinite(mcap) & (mcap > 0) & (amtS > 0)
        sc = np.where(pool, score[m], np.nan)
        idx = np.flatnonzero(np.isfinite(sc))
        if len(idx) < 100:
            continue
        if quintile:
            k = max(20, int(len(idx) * 0.2))
        else:
            k = min(top_k, len(idx))
        order = idx[np.argsort(-sc[idx], kind="stable")][:k]
        target = {int(i): 1.0 / len(order) for i in order}

        prices = {i: float(cS[i]) for i in set(shares) | set(target)}
        # ADV：决策日前 20 日日均成交额
        am = rp.rows("amt", max(0, S - 20), S)
        with np.errstate(all="ignore"):
            adv_arr = np.nanmean(np.where(am > 0, am, np.nan), axis=0) if len(am) else np.zeros(NC)
        adv_arr = np.nan_to_num(adv_arr, nan=0.0)
        adv = {i: float(adv_arr[i]) if np.isfinite(adv_arr[i]) else 0.0
               for i in set(shares) | set(target)}
        idxs = list(set(shares) | set(target))
        fl = flags_at(rp, S, E, idxs)
        flags = {int(i): fl[int(i)] for i in idxs}

        op = rp.col("open", E)
        equity_now = cash + sum(n * (op[i] if np.isfinite(op[i]) and op[i] > 0 else last_px.get(i, 0.0))
                               for i, n in shares.items())
        plan = OrderPlan(equity_now, lot=cfg.execution.lot, max_weight=cap_w)
        orders = plan.build({rp.codes[i]: w for i, w in target.items()},
                            {rp.codes[i]: n for i, n in shares.items()},
                            {rp.codes[i]: n for i, n in shares.items()},  # 月度调仓，T+1 自然满足
                            {rp.codes[i]: v for i, v in prices.items()},
                            rp.dates[S], rp.dates[E])
        # 补 decision_price 与目标权重映射（OrderPlan 用代码，风控用序号）
        code_of = {rp.codes[i]: i for i in set(shares) | set(target)}
        orders = [replace(o, decision_price=float(prices[code_of[o.code]]),
                          target_weight=float(target.get(code_of[o.code], 0.0))) for o in orders]
        if risk_on:
            approved, rejected = risk.check(
                orders, prices={o.code: prices[code_of[o.code]] for o in orders},
                adv={o.code: adv[code_of[o.code]] for o in orders},
                cash=cash, equity=equity_now,
                industries={o.code: ind.get(o.code, "__unknown__") for o in orders},
                bar_flags={o.code: flags[code_of[o.code]] for o in orders},
                positions={o.code: shares.get(code_of[o.code], 0) for o in orders})
        else:
            approved = [replace(o, status="APPROVED") for o in orders]
            rejected = []
        for o in rejected:
            rejects[o.reject_reason] = rejects.get(o.reject_reason, 0) + 1
        fills = broker.send(approved, prices={rp.codes[i]: (float(op[i]) if np.isfinite(op[i]) else 0.0)
                                              for i in set(shares) | set(target)},
                            adv={rp.codes[i]: adv[i] for i in set(shares) | set(target)},
                            date=rp.dates[E], cfg=cfg, cash=cash,
                            bar_flags={rp.codes[i]: flags[i] for i in set(shares) | set(target)})
        rec = reconcile(approved + rejected, fills, set(broker.last_rejects))
        if not rec["clean"]:
            recon_bad += 1
        fill_by_id = {}
        for f in fills:
            fill_by_id.setdefault(f.order_id, []).append(f)
        for o in approved:
            fs_ = fill_by_id.get(o.order_id, [])
            if not fs_:
                continue
            f = fs_[0]; i = code_of[o.code]
            amt = f.shares * f.price
            fee = float(f.fee.get("total", 0.0))
            for kk in cost_sum:
                cost_sum[kk] += float(f.fee.get(kk, 0.0))
            turn_sum += amt
            if o.side == "BUY":
                cash -= amt + fee
                shares[i] = shares.get(i, 0) + f.shares
            else:
                cash += amt - fee
                left = shares.get(i, 0) - f.shares
                if left <= 0:
                    shares.pop(i, None)
                else:
                    shares[i] = left
            is_list.append(fill_is_bps(o, f))
        # 逐日 buy-and-hold 计价：E 当日收盘 -> E2-1 收盘
        if E2 > E:
            cl = rp.rows("close", E, E2)
            cr = rp.bad[E:E2].astype(bool)
            hr = rp.halt[E:E2].astype(bool)
            held = list(shares.items())
            for t in range(E, E2):
                row = cl[t - E]
                mv = 0.0
                for i, n in held:
                    if i in dead:
                        continue                      # 已按 0 记市值，永久死亡
                    px = row[i]
                    if cr[t - E, i]:
                        # 前复权数据损坏：永久按 0 记市值，不生成现金、不"卖出"
                        delisted += 1
                        dead.add(i)
                        last_px[i] = 0.0
                    elif hr[t - E, i] or not (np.isfinite(px) and px > 0):
                        # 停牌 / 未上市：沿用最后已知价，不动现金也不清零
                        mv += n * last_px.get(i, 0.0)
                        continue
                    else:
                        last_px[i] = float(px)
                        mv += n * float(px)
                curve.append((rp.dates[t], cash + mv))
            if dead:
                for i in list(shares):
                    if i in dead:
                        shares.pop(i, None)
                        last_px[i] = 0.0
        all_plan.extend([o.as_dict() for o in approved + rejected])
        all_fills.extend([f.as_dict() for f in fills])

    if write_ledger:
        with open(os.path.join(HERE, "order_plan_%s.jsonl" % name), "w", encoding="utf-8") as f:
            for r in all_plan:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(HERE, "fills_%s.jsonl" % name), "w", encoding="utf-8") as f:
            for r in all_fills:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    st = stats_of(curve)
    return {"name": name, "aum": aum, "top_k": top_k, "quintile": quintile,
            "stats": st, "cost": {k: round(v, 2) for k, v in cost_sum.items()},
            "turnover_notional": round(turn_sum, 2),
            "is_bps_mean": round(float(np.mean(is_list)), 2) if is_list else None,
            "rejects": rejects, "recon_failures": recon_bad,
            "n_orders": len(all_plan), "n_fills": len(all_fills),
            "plan_fingerprint": order_plan_fingerprint([OrderRecord(**r) for r in all_plan]) if all_plan else "",
            "delisted_mark_events": delisted,
            "curve": curve}, curve


def main():
    rp, M, JJ, EX, NS, icj, cal = load_context()
    industries = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    print("[load] %d 个唯一月末 %s ~ %s，%d 只股票，%d 个交易日" % (
        len(JJ), rp.dates[int(JJ[0])], rp.dates[int(JJ[-1])], rp.NC, rp.ND), flush=True)
    # 质量闸门预计算：坏样本前后各 60 日整段剔除，一次扫描代替每月重复扫描
    # clean[S] = 截至 S 从未出现过前复权数据损坏（只用过去，无未来函数）
    # 逐行累计，避免 8816x5251 的 int64 临时数组撞本机 512MB 提交上限
    clean = np.empty((rp.ND, rp.NC), dtype=bool)
    running = np.zeros(rp.NC, dtype=bool)
    for j in range(rp.ND):
        running |= rp.bad[j].astype(bool)
        clean[j] = ~running
    rp.clean = clean
    sig = build_signals(rp, M, JJ, NS, icj)
    # 宽松口径：不因「历史上出现过前复权数据损坏」永久剔除股票，只在使用当日
    # corrupt 时跳过计价。两种口径并列输出，用来量化幸存者偏差。
    rp.clean = np.ones((rp.ND, rp.NC), dtype=bool)
    sig_loose = build_signals(rp, M, JJ, NS, icj)
    rp.clean = clean
    print("[signals] strict", {k: int(np.isfinite(v).sum()) for k, v in sig.items()},
          "loose", {k: int(np.isfinite(v).sum()) for k, v in sig_loose.items()}, flush=True)
    cfg = Config()
    # 同池、同 k、同频率、同成本的可交易随机基准（固定哈希，可复现）
    rnd = np.full((NS, rp.NC), np.nan)
    for m in range(NS):
        h = np.array([int.from_bytes(__import__("hashlib").sha256(
            ("20261105|%s|%s" % (rp.dates[int(JJ[m])], c)).encode()).digest()[:8], "big")
            for c in rp.codes], dtype=np.float64) / 2 ** 64
        rnd[m] = h
    results, curves = {}, {}
    jobs = [
        ("size_amt20_top50", sig["size_amt20"], 1e6, 50, False, True),
        ("size_amt20_quintile_anchor", sig["size_amt20"], 1e6, 50, True, True),
        ("ep_top50", sig["ep"], 1e6, 50, False, True),
        ("size_amt20_aum1e8", sig["size_amt20"], 1e8, 50, False, True),
        ("size_amt20_no_risk", sig["size_amt20"], 1e6, 50, False, False),
        ("loose_size_amt20_top50", sig_loose["size_amt20"], 1e6, 50, False, True),
    ]
    for name, score, aum, k, quin, rk in jobs:
        r, c = run_config(name, score, rp, M, JJ, EX, NS, cfg, aum=aum, top_k=k,
                          quintile=quin, industries=industries, risk_on=rk,
                          write_ledger=name in ("size_amt20_top50", "size_amt20_aum1e8",
                                                "loose_size_amt20_top50"))
        results[name] = {k2: v for k2, v in r.items() if k2 != "curve"}
        curves[name] = c
        print("%-30s CAGR %6s%%  夏普 %5s  MDD %7s%%  成本 %.0f元  IS %.1fbps  拒单 %s  对账失败 %d" % (
            name, r["stats"].get("cagr_pct"), r["stats"].get("sharpe"), r["stats"].get("mdd_pct"),
            r["cost"]["total"], r["is_bps_mean"] or 0, sum(r["rejects"].values()), r["recon_failures"]), flush=True)

    # 可执行基准：同池、同频率、同成本的全池等权（不排序、不抽样，修掉历史基准 bug）
    # 基准也用宽松口径（不引入 corrupt 剔除带来的选股偏差）
    rp_clean_before = rp.clean
    rp.clean = np.ones((rp.ND, rp.NC), dtype=bool)
    base, base_curve = run_config("random_k_equal", rnd, rp, M, JJ, EX, NS, cfg,
                                  aum=1e6, top_k=50, industries=industries,
                                  risk_on=True, write_ledger=False)
    results["random_k_equal"] = {k: v for k, v in base.items() if k != "curve"}
    curves["random_k_equal"] = base_curve
    rp.clean = rp_clean_before
    print("%-30s CAGR %6s%%  夏普 %5s" % ("random_k_equal",
          base["stats"].get("cagr_pct"), base["stats"].get("sharpe")), flush=True)

    # 归因：月度超额对 MKT/SMB/HML 回归 + Newey-West HAC
    fkeys, F_MKT, F_SMB, F_HML = market_and_style_factors(rp, M, JJ, NS)
    fmap = {k: (F_MKT[i], F_SMB[i], F_HML[i]) for i, k in enumerate(fkeys)}
    attrib = {}
    for name, c in curves.items():
        ks, rs = monthly_returns(c)
        common = [i for i, k in enumerate(ks) if k in fmap]
        if len(common) < 24:
            continue
        y = np.array([rs[i] for i in common])
        X = np.column_stack([np.ones(len(y))] +
                            [np.array([fmap[ks[i]][j] for i in common]) for j in range(3)])
        beta, t, r2 = nw_t(y, X)
        excess = y - np.array([fmap[ks[i]][0] for i in common])
        attrib[name] = {"months": len(y),
                        "alpha_annual_pct": round(((1 + beta[0]) ** 12 - 1) * 100, 2),
                        "alpha_t_nw": round(float(t[0]), 2),
                        "beta_mkt": round(float(beta[1]), 3),
                        "beta_smb": round(float(beta[2]), 3),
                        "beta_hml": round(float(beta[3]), 3),
                        "r2": round(float(r2), 3),
                        "excess_annual_pct": round(float(excess.mean()) * 12 * 100, 2)}
    print("[attribution] 因子月数 %d, 归因组合 %d" % (len(fkeys), len(attrib)), flush=True)

    out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "panel": {"file": "m_compact.bin", "stocks": rp.NC, "days": rp.ND,
                     "unique_months": len(JJ), "range": [rp.dates[int(JJ[0])], rp.dates[int(JJ[-1])]],
                     "signal_day": "S=月末收盘", "exec_day": "E=S+1 开盘",
                     "cost": "quantlab.model.cost（佣金/过户/印花分段/校准冲击 eta=%.5f）" % 0.00975,
                     "risk": "单票4%/行业20%/ADV参与率10%/涨跌停/现金/整手/T+1",
                     "known_gaps": ["无 high/low，涨跌停用开盘触板近似（偏保守）",
                                    "未剔除 ST（无历史 ST 名单）",
                                    "财务公告日用法定披露截止日兜底，非真实公告日",
                                    "持有期停牌沿用最后已知价"]},
           "results": results, "attribution": attrib,
           "curves": {k: v for k, v in curves.items()}}
    json.dump(out, open(os.path.join(HERE, "unified_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved unified_result.json", flush=True)
    return out


if __name__ == "__main__":
    main()
