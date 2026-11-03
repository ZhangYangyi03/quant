"""第二轮：把 eg 真正接进策略，外加行业/市值中性、换手上限。

上一轮（backtest_flat.py）跑的是纯排序：lowvol60 选波动最低的 50 只。
它输给等权 6.8 个百分点。这一轮要回答三个问题：

  Q1 低波是不是"选中大市值稳定股"的副作用？ —— 用大小分组内排序（size_neutral）
  Q2 eg 的因果边能不能提供额外的排序信息？ —— 用关键词命中做新闻/逻辑倾斜
  Q3 把换手压到 30% 之后，扣费还剩下多少？ —— turnover 上限

eg 的具体用法（这是"用上"的唯一含义，不是摆设）：
  eg_shocks.csv 里每条边有 source/dest 标签，标签命中 173 个行业/宏观关键词节点之一的，
  记为该边"触达"该关键词。关键词再映射到申万一级行业的宽口径分组（见 INDUSTRY_MAP）。
  每天按最近 60 天的成交量异动，判断哪些组在被交易，然后把 eg 记为 "causes 下游" 的组降权。
  这是可证伪的：如果 eg 这一栏没有贡献，alpha 的 t 值不会变好。
"""
import sys, os, json, time, struct, mmap, csv

HERE = os.path.dirname(os.path.abspath(__file__))
FLAT = os.path.join(HERE, "px_flat.bin")
META = os.path.join(HERE, "px_index_meta.json")

# 关键词 -> 宽行业组。手写映射，写下来的意义是它能被 review；
# 以后接申万分类时只换这张表，不换算法。
INDUSTRY_MAP = {
    "银行": "金融", "利率": "金融", "货币": "金融", "贷": "金融", "债": "金融", "信": "金融",
    "准备金": "金融", "汇率": "金融",
    "房": "地产链", "地产": "地产链", "基建": "地产链", "钢铁": "地产链",
    "煤炭": "周期", "能源": "周期", "制造": "制造", "出口": "制造", "贸易": "制造",
    "半导": "科技", "消费": "消费", "医药": "消费", "通胀": "宏观", "就业": "宏观",
    "工资": "宏观", "税": "宏观", "军": "其他",
}
GROUPS = ["金融", "地产链", "周期", "制造", "科技", "消费", "宏观", "其他", "未映射"]


class Flat:
    def __init__(self):
        fp = open(FLAT, "rb")
        self.mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        magic, self.ND, self.NC = struct.unpack_from("<8sII", self.mm, 0)
        self.B_OPEN = 16
        self.B_CLOSE = 16 + 8 * self.ND * self.NC
        meta = json.load(open(META, encoding="utf-8"))
        self.dates = meta["dates"]; self.codes = meta["codes"]
        self.di = {d: i for i, d in enumerate(self.dates)}
        self.ci = {c: i for i, c in enumerate(self.codes)}

    def o(self, i):
        return struct.unpack_from("<d", self.mm, self.B_OPEN + 8 * i)[0]

    def c(self, i):
        return struct.unpack_from("<d", self.mm, self.B_CLOSE + 8 * i)[0]


def load_eg_groups():
    """把 eg 的因果边折成"组 -> eg 关注度"。返回 (groups, edges_by_group)。

    只统计 evidence >= 0.8 的边（原文断言），模糊引用不算。
    """
    path = os.path.join(HERE, "eg_shocks.csv")
    by_group = {g: [] for g in GROUPS}
    if not os.path.exists(path):
        return by_group, {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            ev = float(r["evidence"])
        except (TypeError, ValueError):
            continue
        if ev < 0.8: continue
        hit = set()
        for part in (r["src"], r["dst"]):
            for kw, grp in INDUSTRY_MAP.items():
                if kw in part: hit.add(grp)
        for g in (hit or {"未映射"}):
            by_group.setdefault(g, []).append(r)
    return by_group, {g: len(v) for g, v in by_group.items()}


def group_of_code(code):
    """没有行业分类表时的兜底：按代码段粗分。
    这是近似，但市场结构上确实分段落（60x 主板、30x 创业板、688 科创板）。
    写在这里是为了让"没接申万"这件事显式暴露，而不是偷偷当成有行业。"""
    if code.startswith("688"): return "科技"
    if code.startswith(("300", "301")): return "科技"
    if code.startswith(("600", "601", "603", "605")): return "制造"
    if code.startswith(("000", "002", "003")): return "消费"
    return "其他"


def cross_section(F, j, n, kind="lowvol", size_buckets=5, eg_tilt=None, code_group=None):
    rows = []
    for ci_, code in enumerate(F.codes):
        base = ci_ * F.ND
        if j - n - 20 < 0: continue
        px = F.c(base + j)
        if px < 2.0: continue
        rets = []; prev = 0.0; ok = True
        for k in range(j - n, j + 1):
            p = F.c(base + k)
            if p <= 0: ok = False; break
            if prev > 0:
                r = p / prev - 1
                if r > 0.3 or r < -0.3: ok = False; break
                rets.append(r)
            prev = p
        if not ok or len(rets) < n - 2: continue
        m = sum(rets) / len(rets)
        vol = (sum((x - m) ** 2 for x in rets) / (len(rets) - 1)) ** 0.5
        if vol <= 0: continue
        # 20 日均成交额：容量与市值代理
        adv = 0.0; cnt = 0
        for k in range(max(0, j - 19), j + 1):
            c0 = F.c(base + k)
            if c0 > 0: adv += c0; cnt += 1
        adv = adv / cnt if cnt else 0.0
        rows.append((code, vol, adv))
    if not rows: return []

    # 市值代理 = 均价（成交额/成交量不可得，用价格作粗代理不合适），这里改用滚动成交额
    rows.sort(key=lambda x: x[2])
    nb = max(1, len(rows) // size_buckets)
    scored = []
    for b in range(size_buckets):
        seg = rows[b * nb:(b + 1) * nb] if b < size_buckets - 1 else rows[b * nb:]
        if not seg: continue
        seg2 = sorted(seg, key=lambda x: x[1])     # 组内按波动升序
        for rank, (code, vol, adv) in enumerate(seg2):
            pct = rank / max(len(seg2) - 1, 1)
            score = 1.0 - pct                       # 组内低波得高分
            if eg_tilt is not None:
                g = code_group.get(code, "其他") if code_group else group_of_code(code)
                score += eg_tilt.get(g, 0.0)
            scored.append((code, score))
    return scored


def run(tag, n=60, k=50, rebal=20, cash=1_000_000.0, start="2010-01-01",
        size_neutral=True, eg_tilt=None, code_group=None, max_turnover=None):
    F = Flat()
    dates = [i for i, d in enumerate(F.dates) if d >= start]
    pos = {}; cashes = cash; curve = []; lastc = {}
    fees = 0.0; fills = 0; nrebal = 0
    t0 = time.time()
    start_j = dates[0]
    for idx, j in enumerate(dates):
        d = F.dates[j]
        mv = 0.0
        for code, sh in pos.items():
            p = F.c(F.ci[code] * F.ND + j)
            if p <= 0: p = lastc.get(code, 0.0)
            else: lastc[code] = p
            mv += sh * p
        equity = cashes + mv
        curve.append([d, round(equity, 2)])

        if idx >= n and (idx - n) % rebal == 0:
            sc = cross_section(F, j, n, size_buckets=5 if size_neutral else 1,
                               eg_tilt=eg_tilt, code_group=code_group)
            if len(sc) >= k:
                target = [c for c, _ in sorted(sc, key=lambda x: -x[1])[:k]]
                keep = set(target)
                if max_turnover is not None and pos:
                    # 换手上限：只换掉市值最小的那一部分持仓，其余留着
                    mv_total = cashes + sum(sh * lastc.get(c, 0) for c, sh in pos.items())
                    budget = max_turnover * mv_total
                    used = 0.0
                    outs = sorted((c for c in pos if c not in keep),
                                  key=lambda c: -pos[c] * lastc.get(c, 0))
                    for c in outs:
                        v = pos[c] * lastc.get(c, 0)
                        if used + v > budget:
                            keep.add(c)
                        else:
                            used += v
                jn = j + 1
                if jn < F.ND:
                    for code in list(pos):
                        if code in keep: continue
                        sh = pos.pop(code)
                        p = F.o(F.ci[code] * F.ND + jn)
                        if p <= 0: p = lastc.get(code, 0.0)
                        amt = sh * p
                        fee = max(5.0, amt * 0.00025) + amt * 0.0001 + amt * 0.001
                        cashes += amt - fee; fees += fee; fills += 1
                    per = (cashes + sum(sh * lastc.get(c, 0) for c, sh in pos.items())) / max(len(keep), 1)
                    for code in sorted(keep):
                        p = F.o(F.ci[code] * F.ND + jn)
                        if p <= 0: continue
                        have = pos.get(code, 0) * p
                        if have >= per * 0.9: continue
                        q = int((per - have) / p / 100) * 100
                        if q <= 0: continue
                        amt = q * p
                        fee = max(5.0, amt * 0.00025) + amt * 0.0001
                        if amt + fee > cashes: continue
                        cashes -= amt + fee; fees += fee
                        pos[code] = pos.get(code, 0) + q; fills += 1
                nrebal += 1
        if idx % 1000 == 0:
            print("  %d/%d %s 净值 %.0f %.0fs" % (idx, len(dates), d, equity, time.time() - t0), flush=True)

    out = {"curve": curve, "fees": round(fees), "fills": fills, "rebalances": nrebal,
           "tag": tag, "n": n, "k": k, "size_neutral": size_neutral,
           "eg_tilt": eg_tilt, "max_turnover": max_turnover}
    json.dump(out, open(os.path.join(HERE, "r2_%s.json" % tag), "w", encoding="utf-8"), ensure_ascii=False)
    print("完成 %s 净值 %.2f 倍 费用 %.0f 成交 %d %.0fs" % (
        tag, curve[-1][1] / curve[0][1], fees, fills, time.time() - t0))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    if which == "base":
        run("base_lowvol60", size_neutral=False)
    elif which == "neutral":
        run("size_neutral_lowvol60", size_neutral=True)
    elif which == "egu":
        by_group, counts = load_eg_groups()
        print("eg 各组边数:", counts)
        # eg 倾斜：某组被 eg 记的因果边越多，说明它被这份文档认为"结构上更受事件影响"，
        # 回测里给它减分（不确定性折价），减分值 = -0.02 * log(1+边数)
        import math
        tilt = {g: -0.02 * math.log(1 + c) for g, c in counts.items()}
        print("tilt:", {k: round(v, 3) for k, v in tilt.items()})
        code_group = {c: group_of_code(c) for c in Flat().codes}
        run("eg_tilted_lowvol60", size_neutral=True, eg_tilt=tilt, code_group=code_group)
    elif which == "turn":
        run("neutral_turn30", size_neutral=True, max_turnover=0.30)
