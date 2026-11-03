"""检验事件流有没有预测力：公告事件 -> 个股 -> 行业/全市场。

用的数据：events.jsonl（真实公告）+ 个股日线 + 行业指数。
方法：事件日之后 N 日，事件股的收益 vs 同期全市场等权。
关键是安慰剂：同样数量的随机日期、随机股票，看能不能复现。
"""
import os, json, struct, mmap, math, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load_prices():
    f = open(os.path.join(HERE, "m_close.bin"), "rb")
    magic, ND, NC = struct.unpack_from("<8sII", f.read(16))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    meta = json.load(open(os.path.join(HERE, "px_index_meta.json"), encoding="utf-8"))
    return mm, ND, NC, meta["codes"], meta["dates"]


def px(mm, ND, ci, j):
    return struct.unpack_from("<d", mm, 16 + 8 * (ci * ND + j))[0]


def main():
    mm, ND, NC, codes, dates = load_prices()
    ci_of = {c: i for i, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(dates)}

    evs = []
    seen = set()
    for line in open(os.path.join(HERE, "events.jsonl"), encoding="utf-8"):
        try: e = json.loads(line)
        except: continue
        c = e.get("code")
        if c not in ci_of or e["date"] not in di: continue
        k = (e["date"], c, e["kind"])
        if k in seen: continue
        seen.add(k)
        evs.append(e)
    print("可用事件 %d 条（去重后），覆盖 %d 只股票" % (
        len(evs), len(set(e["code"] for e in evs))))

    # 预计算每日全市场等权收益（只算一次，避免每次遍历 5000 只）
    print("预计算全市场等权日收益...", flush=True)
    mk = [0.0] * ND
    for j in range(1, ND):
        s = 0.0; n = 0
        for ci in range(NC):
            a = px(mm, ND, ci, j - 1); b = px(mm, ND, ci, j)
            if a > 0 and b > 0:
                s += b / a - 1; n += 1
        mk[j] = s / n if n else 0.0
    cum = [0.0] * ND
    for j in range(1, ND):
        cum[j] = cum[j - 1] + math.log1p(mk[j])

    def mkt_ret(j0, j1):
        return math.expm1(cum[min(j1, ND - 1)] - cum[j0])

    def stock_ret(ci, j0, j1):
        a = px(mm, ND, ci, j0); b = px(mm, ND, ci, j1)
        return (b / a - 1) if (a > 0 and b > 0) else None

    print("\n%-10s %5s %8s %10s %10s %8s" % ("事件", "n", "5日超额", "对照均值", "分位", "判定"))
    for kind in sorted(set(e["kind"] for e in evs)):
        rowset = [e for e in evs if e["kind"] == kind]
        for H in (5, 20):
            real = []
            js = []
            for e in rowset:
                j = di[e["date"]]
                if j + H >= ND: continue
                r = stock_ret(ci_of[e["code"]], j, j + H)
                if r is None: continue
                real.append(r - mkt_ret(j, j + H))
                js.append((ci_of[e["code"]], j))
            if len(real) < 15: continue
            mu = sum(real) / len(real)
            # 安慰剂：同样数量，随机日期 + 随机股票
            sims = []
            random.seed(7)
            for _ in range(60):
                s = []
                for _ in range(len(real)):
                    ci = random.randrange(NC)
                    j = random.randrange(0, ND - H)
                    r = stock_ret(ci, j, j + H)
                    if r is not None: s.append(r - mkt_ret(j, j + H))
                if s: sims.append(sum(s) / len(s))
            if not sims: continue
            sims.sort()
            pct = sum(1 for x in sims if x < mu) / len(sims) * 100
            verdict = "真" if (pct >= 95 or pct <= 5) else "噪声"
            if H == 5:
                print("%-10s %5d %7.2f%% %9.2f%% %7.0f%% %8s" % (
                    kind, len(real), mu * 100, sum(sims) / len(sims) * 100, pct, verdict), flush=True)


if __name__ == "__main__":
    main()
