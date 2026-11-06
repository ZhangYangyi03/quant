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
import os, sys, json, math, struct, bisect, datetime, time
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


def visible_date(rp, code: str, period: str) -> str:
    """真实公告日优先，缺失时退回法定披露截止日。

    真实公告覆盖率 72.66%（2026-11-06 补齐年报后重算，见 ANNUAL_PIT.md）：
    一季 75.5% / 半年 74.6% / 三季 75.9% / 年报 65.8%。
    年报分片里 REPORT_DATE 有值的是更正/追溯稿，不是首次披露，清洗时一律丢弃；
    详见 build_annual_ann.py。
    """
    nd = rp.ann(code, period)
    if nd:
        return nd
    return disclosure_deadline(period)


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
        _hl = np.load(os.path.join(HERE, "hl_exec.npz"))
        self.hl_date_idx = {str(d): i for i, d in enumerate(_hl["dates"])}
        self.hl_high, self.hl_low = _hl["high"], _hl["low"]
        # 公告日历 18 万条若常驻会再吃掉 ~40MB，叠加大面板刚好撞本机 512MB
        # 提交上限；改成按需加载，用完即释放。
        self._ann_cache = None

    def ann(self, code, period):
        if self._ann_cache is None:
            self._ann_cache = json.load(open(os.path.join(HERE, "report_ann_dates.json"),
                                             encoding="utf-8"))
        return self._ann_cache.get(code, {}).get(period)

    def drop_ann(self):
        self._ann_cache = None

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
                ann = visible_date(rp, code, r["d"])
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
    """执行日 E 的涨跌停/停牌/一字板，以及决策价。

    用 stockdata/*.csv 的 high/low 判一字板（high==low 且触及涨跌停），
    比"开盘触板"精确：开盘触板但盘中有成交的票仍可买卖。
    """
    cs = rp.col("close", S); op = rp.col("open", E); am = rp.col("amt", E)
    k = rp.hl_date_idx.get(rp.dates[E])
    H = rp.hl_high[k] if k is not None else None
    L = rp.hl_low[k] if k is not None else None
    out = {}
    for i in idxs:
        i = int(i)
        lim = limit_gem if (rp.codes[i][:3] in ("300", "301", "688")) else limit_main
        px_prev = float(cs[i]) if np.isfinite(cs[i]) else 0.0
        px = float(op[i]) if np.isfinite(op[i]) else 0.0
        up_px = px_prev * (1 + lim) if px_prev > 0 else 0.0
        dn_px = px_prev * (1 - lim) if px_prev > 0 else 0.0
        ratio = px / px_prev - 1.0 if px_prev > 0 and px > 0 else 0.0
        # 一字涨停：最高=最低=涨停价；一字跌停：最高=最低=跌停价
        one_up = bool(H is not None and np.isfinite(H[i]) and np.isfinite(L[i])
                      and up_px > 0 and abs(H[i] - L[i]) < 1e-6
                      and abs(H[i] - up_px) <= max(0.011, up_px * 0.005))
        one_dn = bool(H is not None and np.isfinite(H[i]) and np.isfinite(L[i])
                      and dn_px > 0 and abs(H[i] - L[i]) < 1e-6
                      and abs(H[i] - dn_px) <= max(0.011, dn_px * 0.005))
        bad = bool(rp.bad[E, i]) or (not rp.clean[S][i])
        out[i] = {"decision_price": px_prev,
                  "suspended": bool(px <= 0 or am[i] <= 0 or bad),
                  "limit_up": bool(ratio >= lim - 0.005 or one_up),
                  "limit_down": bool(ratio <= -lim + 0.005 or one_dn),
                  "one_word_board": bool(one_up or one_dn)}
    return out


def market_and_style_factors(rp, M, JJ, NS):
    """同池、同决策日、同执行日的可交易风格基准（不依赖任何策略净值）。

    MKT  全池等权，E 开盘 -> 下一 E 开盘
    SMB  小市值 30% 等权 - 大市值 30% 等权
    HML  高 BM（BPS/收盘价）30% 等权 - 低 BM 30% 等权，Fama-French 标准口径
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
                ann = visible_date(rp, code, r["d"])
            except (ValueError, KeyError):
                continue
            if r.get("bps") is not None:
                rs.append((ann, float(r["bps"])))
        rs.sort()
        recs[ci] = rs
    bps_ann = {ci: [x[0] for x in rr] for ci, rr in recs.items()}
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
        # BM：可见最新一期 BPS / 决策日收盘（bisect 定位，避免逐股列表推导）
        bm = np.full(len(gidx), np.nan)
        d = rp.dates[S]
        for t, ci in enumerate(gidx):
            rr = recs.get(int(ci), ())
            if not rr or cS[ci] <= 0:
                continue
            lo = bisect.bisect_right(bps_ann[int(ci)], d) - 1
            if lo >= 0:
                bm[t] = rr[lo][1] / cS[ci]
        v = np.isfinite(bm) & (bm > 0)
        if v.sum() > 100:
            qq = np.quantile(bm[v], [0.3, 0.7])
            hml.append(float(ret[v][bm[v] >= qq[1]].mean() - ret[v][bm[v] <= qq[0]].mean()))
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
    """纯调度：面板与配置都不在主进程展开，规避本机 512MB 提交上限。"""
    import subprocess
    results, curves = {}, {}
    JOBS = [
        ("size_amt20_top50", "size_amt20", "strict", 1e6, 50, False, True),
        ("size_amt20_quintile_anchor", "size_amt20", "strict", 1e6, 50, True, True),
        ("ep_top50", "ep", "strict", 1e6, 50, False, True),
        ("size_amt20_aum1e8", "size_amt20", "strict", 1e8, 50, False, True),
        ("size_amt20_no_risk", "size_amt20", "strict", 1e6, 50, False, False),
        ("loose_size_amt20_top50", "size_amt20", "loose", 1e6, 50, False, True),
        ("pool_loose_size_amt20_top50", "size_amt20", "strict", 1e6, 50, False, True),
        ("random_k_equal", "random", "loose", 1e6, 50, False, True),
    ]
    CACHE = os.path.join(HERE, "_config_cache")
    os.makedirs(CACHE, exist_ok=True)
    for name, which, uni, aum, k, quin, rk in JOBS:
        cf = os.path.join(CACHE, name + ".json")
        if os.path.exists(cf):
            got = json.load(open(cf, encoding="utf-8"))
            results[got["name"]] = {k2: v for k2, v in got.items() if k2 != "curve"}
            curves[got["name"]] = [tuple(x) for x in got["curve"]]
            print("%-30s (cached)" % name, flush=True)
            continue
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        args = [sys.executable, "-u", os.path.abspath(__file__), "--config", name,
                "--signal", which, "--universe", uni, "--aum", repr(aum),
                "--topk", str(k), "--quintile", "1" if quin else "0",
                "--risk", "1" if rk else "0", "--pool",
                "loose" if name.startswith("pool_loose_") else "strict"]
        t0 = time.time()
        r = subprocess.run(args, cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0:
            print("[fail] %s rc=%d\n%s" % (name, r.returncode, r.stderr[-1500:]), flush=True)
            continue
        got = json.loads(r.stdout.strip().splitlines()[-1])
        json.dump(got, open(cf, "w", encoding="utf-8"), ensure_ascii=False)
        results[got["name"]] = {k2: v for k2, v in got.items() if k2 != "curve"}
        curves[got["name"]] = [tuple(x) for x in got["curve"]]
        print("%-30s CAGR %6s%%  夏普 %5s  MDD %7s%%  成本 %.0f元  IS %.1fbps  拒单 %s  对账失败 %d  (%.1fs)" % (
            name, got["stats"].get("cagr_pct"), got["stats"].get("sharpe"),
            got["stats"].get("mdd_pct"), got["cost"]["total"], got["is_bps_mean"] or 0,
            sum(got["rejects"].values()), got["recon_failures"], time.time() - t0), flush=True)

    # 风格因子也交给子进程
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    ff = os.path.join(CACHE, "_factors.json")
    if os.path.exists(ff):
        fac = json.load(open(ff, encoding="utf-8"))
    else:
        r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--factors", "1"],
                           cwd=HERE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0:
            raise RuntimeError("因子子进程失败: " + r.stderr[-1500:])
        fac = json.loads(r.stdout.strip().splitlines()[-1])
        json.dump(fac, open(ff, "w", encoding="utf-8"), ensure_ascii=False)
    fkeys, F_MKT, F_SMB, F_HML = (fac["keys"], np.array(fac["mkt"]),
                                  np.array(fac["smb"]), np.array(fac["hml"]))
    print("[load] %d 个配置，%d 个因子月" % (len(curves), len(fkeys)), flush=True)

    # 统计检验：把多重检验/过拟合/样本外三道关接到真实月度收益上
    from quantlab.validate.stats import (deflated_sharpe, pbo,
                                        bootstrap_max_sharpe_p)
    from quantlab.validate.walkforward import run_walkforward
    monthly_by_cfg = {}
    for name, c in curves.items():
        ks, rs = monthly_returns(c)
        monthly_by_cfg[name] = [float(x) for x in rs]
    # 只用真正"试过的策略配置"计数：主线 5 条 + 宽松口径 + 随机对照 + 基准
    n_trials = 8
    validation = {"n_trials": n_trials, "per_config": {}}
    for name, rs in monthly_by_cfg.items():
        if len(rs) < 24:
            continue
        n = len(rs)
        mu = float(np.mean(rs)); sd = float(np.std(rs, ddof=1))
        sr_m = mu / sd if sd > 0 else 0.0
        skew = float(np.mean(((np.array(rs) - mu) / sd) ** 3)) if sd > 0 else 0.0
        kurt = float(np.mean(((np.array(rs) - mu) / sd) ** 4)) if sd > 0 else 3.0
        # DSR 的 sr_daily 用日频 Sharpe（月度 Sharpe / sqrt(21)）
        sr_d = sr_m / math.sqrt(21.0)
        dsr = deflated_sharpe(sr_d, n_trials=n_trials, n_obs=n * 21,
                              skew=skew, kurtosis=kurt)
        validation["per_config"][name] = {
            "n_months": n, "sharpe_monthly": round(sr_m, 3),
            "dsr_prob": round(dsr, 4),
            "dsr_pass": bool(dsr > 0.95)}
    # PBO / WRC：把所有配置摆在同一张收益矩阵上
    validation["pbo"] = round(pbo(monthly_by_cfg, slices=8), 4)
    validation["wrc_p"] = round(bootstrap_max_sharpe_p(monthly_by_cfg, draws=100, seed=7), 4)
    # Walk-forward：每月用过去 60 个月选夏普最高的配置，下 6 个月样本外结算
    def _choose(hist):
        return max(hist, key=lambda k: (np.mean(hist[k]) / np.std(hist[k], ddof=1)
                                        if len(hist[k]) > 1 and np.std(hist[k], ddof=1) > 0 else -9))
    wf = run_walkforward(monthly_by_cfg, _choose, min_history=60, oos_len=6, strict=True)
    if wf:
        oos = np.array([x.oos_return for x in wf])
        validation["walkforward"] = {"periods": len(wf),
                                    "oos_months": int(len(wf) * 6),
                                    "oos_sum_ret_pct": round(float(oos.sum()) * 100, 2),
                                    "oos_mean_monthly_pct": round(float(oos.mean()) / 6 * 100, 3),
                                    "oos_positive_rate": round(float((oos > 0).mean()), 3),
                                    "chosen_counts": {k: sum(1 for x in wf if x.chosen == k)
                                                      for k in sorted(set(x.chosen for x in wf))}}
    print("[validation] done", flush=True)

    # 归因：月度超额对 MKT/SMB/HML 回归 + Newey-West HAC
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
           "panel": {"file": "m_compact.bin", "stocks": 5251, "days": 8816,
                     "unique_months": 166,
                     "range": [fac["range"][0], fac["range"][1]],
                     "signal_day": "S=月末收盘", "exec_day": "E=S+1 开盘",
                     "cost": "quantlab.model.cost（佣金/过户/印花分段/校准冲击 eta=0.00975）",
                     "risk": "单票4%/行业20%/ADV参与率10%/涨跌停/现金/整手/T+1",
                     "resolved": ["真实 high/low 一字板判定（hl_exec.npz）",
                                  "HML 改用 BPS/收盘价 BM，Fama-French 标准口径",
                                  "72.66% 财报用真实 NOTICE_DATE（年报已补齐），其余法定截止日兜底",
                                  "DSR / PBO / White Reality Check / walk-forward 检验"],
                     "known_gaps": ["历史 ST 名单：fund_parts 的 SECURITY_NAME_ABBR 是当前简称回填（202/202 只全期同名），仍需外部源",
                                    "真实公告日覆盖 72.66%（年报 65.8% 最低），其余仍用法定截止日兜底",
                                    "价格源为前复权，未用不复权价重拉",
                                    "持有期停牌沿用最后已知价",
                                    "未做真正 OOS（walk-forward 仅在同一批配置上滚动选参）"]},
           "results": results, "attribution": attrib, "validation": validation,
           "curves": {k: [list(x) for x in v] for k, v in curves.items()}}
    json.dump(out, open(os.path.join(HERE, "unified_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved unified_result.json", flush=True)
    return out


def _random_score(rp, M, JJ, NS):
    import hashlib
    rnd = np.full((NS, rp.NC), np.nan)
    for m in range(NS):
        rnd[m] = np.array([int.from_bytes(hashlib.sha256(
            ("20261105|%s|%s" % (rp.dates[int(JJ[m])], c)).encode()).digest()[:8], "big")
            for c in rp.codes], dtype=np.float64) / 2 ** 64
    return rnd


def _clean_from_bad(rp):
    """clean[S] = 截至 S 从未出现过前复权数据损坏（只用过去，无未来函数）。"""
    clean = np.empty((rp.ND, rp.NC), dtype=bool)
    running = np.zeros(rp.NC, dtype=bool)
    for j in range(rp.ND):
        running |= rp.bad[j].astype(bool)
        clean[j] = ~running
    return clean


def run_one(args):
    """单条配置的独立进程入口：跑完把结果 JSON 打到 stdout 最后一行。"""
    import gc
    rp, M, JJ, EX, NS, icj, cal = load_context()
    industries = json.load(open(os.path.join(HERE, "stock_industry.json"), encoding="utf-8"))
    clean = _clean_from_bad(rp)
    # signal_universe：打分时是否把 corrupt 股票排除（宽松信号）
    # pool_universe：持仓池是否排除 corrupt 股票（严格口径）
    sig_clean = clean if args.universe == "strict" else np.ones((rp.ND, rp.NC), dtype=bool)
    pool_clean = clean if args.pool != "loose" else np.ones((rp.ND, rp.NC), dtype=bool)
    rp.clean = sig_clean
    if args.signal == "random":
        score = _random_score(rp, M, JJ, NS)
    else:
        sig = build_signals(rp, M, JJ, NS, icj)
        score = sig[args.signal]
        del sig
    rp.drop_ann(); rp.clean = pool_clean; gc.collect()
    cfg = Config()
    r, c = run_config(args.config, score, rp, M, JJ, EX, NS, cfg, aum=args.aum,
                      top_k=args.topk, quintile=bool(int(args.quintile)),
                      industries=industries, risk_on=bool(int(args.risk)),
                      write_ledger=args.config in ("size_amt20_top50", "size_amt20_aum1e8",
                                                   "loose_size_amt20_top50"))
    r["curve"] = [[d, float(v)] for d, v in c]
    print(json.dumps(r, ensure_ascii=False), flush=True)


def _parse_one(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--signal", required=True)
    ap.add_argument("--universe", default="strict")
    ap.add_argument("--pool", default="strict")
    ap.add_argument("--aum", type=float, default=1e6)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--quintile", default="0")
    ap.add_argument("--risk", default="1")
    return ap.parse_args(argv)


if __name__ == "__main__" and "--config" in sys.argv:
    run_one(_parse_one(sys.argv[1:]))
    sys.exit(0)


def run_factors():
    rp, M, JJ, EX, NS, icj, cal = load_context()
    rp.clean = _clean_from_bad(rp)
    k, mk, smb, hml = market_and_style_factors(rp, M, JJ, NS)
    rp.drop_ann()
    print(json.dumps({"keys": k, "range": [rp.dates[int(JJ[0])], rp.dates[int(JJ[-1])]], "mkt": [float(x) for x in mk],
                      "smb": [float(x) for x in smb], "hml": [float(x) for x in hml]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__" and "--factors" in sys.argv:
    run_factors()
    sys.exit(0)


if __name__ == "__main__":
    main()
