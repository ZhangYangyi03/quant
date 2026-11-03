"""安慰剂检验：如果降准日的行业效应在随机日期上也能复现，那它就不是降准造成的。

方法：抽 200 组随机日期（同样数量、同样年份分布），每组算行业超额，
再和真实降准日的结果比对。真实效应若落在随机分布的中间，就是噪声。
"""
import os, json, struct, mmap, math, random

HERE = os.path.dirname(os.path.abspath(__file__))
from event_study import CUT_RRR, load, cum


def main():
    mm, NI, ND, names, dates = load()
    di = {d: i for i, d in enumerate(dates)}
    # 真实事件对应的索引
    real_j = []
    for d in CUT_RRR:
        j = di.get(d)
        if j is None:
            cand = [k for k, x in enumerate(dates) if x >= d]
            if cand: j = cand[0]
        if j is not None and j + 61 < ND: real_j.append(j)
    print("可用降准事件 %d 个" % len(real_j))

    # 随机日期从同一批年份里抽（保持年份分布一致）
    years = [dates[j][:4] for j in real_j]
    yearset = set(years)
    pool = {}
    for j, dd in enumerate(dates):
        yy = dd[:4]
        if yy in yearset and 5 < j < ND - 61:
            pool.setdefault(yy, []).append(j)
    print("随机池:", {y: len(v) for y, v in pool.items()})

    def excess_at(js, h):
        """返回 行业 x 事件数 的超额矩阵（相对全市场等权）"""
        out = []
        for j in js:
            r = cum(mm, NI, ND, j, h)
            b = sum(r) / NI
            out.append([r[i] - b for i in range(NI)])
        return out

    H = 20
    real = excess_at(real_j, H)
    n = len(real)
    real_mu = [sum(r[i] for r in real) / n for i in range(NI)]

    random.seed(42)
    sims = []
    for b in range(200):
        js = []
        for y in years:
            js.append(random.choice(pool[y]))
        sims.append(excess_at(js, H))

    print("\n%d 日后：真实降准 vs 200 次随机对照（同数量、同年份）" % H)
    print("%-14s %10s %12s %10s" % ("行业", "真实超额", "随机均值", "随机分位"))
    for key in ("股份制银行Ⅲ", "城商行Ⅲ", "农商行Ⅲ", "证券Ⅱ", "半导体设备", "水泥制造", "航运"):
        if key not in names: continue
        i = names.index(key)
        rm = real_mu[i]
        dist = sorted(sum(s[k][i] for k in range(len(s))) / len(s) for s in sims)
        pct = sum(1 for x in dist if x < rm) / len(dist) * 100
        avg = sum(dist) / len(dist)
        print("%-14s %9.2f%% %11.2f%% %9.0f%%" % (key, rm * 100, avg * 100, pct))

    # 全局：真实效应有多少行业落在随机分布 5% 或 95% 之外
    outside = 0
    for i in range(NI):
        dist = sorted(sum(s[k][i] for k in range(len(s))) / len(s) for s in sims)
        rm = real_mu[i]
        if rm < dist[10] or rm > dist[-10]: outside += 1
    print("\n254 个行业中，真实效应落在随机分布两端各 5%% 之外的：%d 个" % outside)
    print("（若无真实效应，期望约 %d 个）" % (NI // 10))


if __name__ == "__main__":
    main()
