"""按 EG 的真实因果链重测：货币扩张 -> 银行板块。

上一版错在哪
-----------
我测的是"降准公告日"，但降准提前一周公布，公告日没有新信息；
而且降准只改总量，不改行业间的相对受益。所以测出来什么都没有，
这不说明 EG 错，说明我测的环节错了。

EG 的句子是"准备金率越低，放水越多，效果越快"。
这句话里可观测的中间变量是 M2/M1 增速，不是公告日。

所以这一版测的是：
  Q1  货币扩张（M2 同比上升）之后，银行板块是否跑赢？
  Q2  这个关系在上市银行数量更多的年代是否更强？
  Q3  银行板块对 M2 的敏感度，是否明显高于其他行业？
"""
import os, json, struct, mmap, math

HERE = os.path.dirname(os.path.abspath(__file__))


def load_ind():
    f = open(os.path.join(HERE, "ind_index.bin"), "rb")
    magic, NI, ND, _ = struct.unpack("<8sIII", f.read(20))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    meta = json.load(open(os.path.join(HERE, "ind_names.json"), encoding="utf-8"))
    return mm, NI, ND, meta["names"], meta["dates"]


def col(mm, NI, ND, j):
    return [struct.unpack_from("<f", mm, 20 + 4 * (i * ND + j))[0] for i in range(NI)]


def main():
    mm, NI, ND, names, dates = load_ind()
    di = {d: i for i, d in enumerate(dates)}

    # 货币数据：月份 -> M2 同比
    raw = json.load(open(os.path.join(HERE, "macro_money.json"), encoding="utf-8"))
    m2s = {}
    for r in raw:
        t = r.get("TIME") or ""
        ym = t[:4] + "-" + t[5:7]
        v = r.get("CURRENCY_SAME")
        if v is not None:
            try: m2s[ym] = float(v)
            except Exception: pass
    print("M2 同比覆盖 %d 个月：%s ~ %s" % (len(m2s), min(m2s), max(m2s)))

    yms = sorted(m2s)
    # 货币扩张事件：M2 同比相较上个月上升 >= 0.3 个百分点
    events = []
    for k in range(1, len(yms)):
        d = m2s[yms[k]] - m2s[yms[k - 1]]
        if d >= 0.3:
            events.append((yms[k], d))
    print("M2 加速月份 %d 个" % len(events))

    # 预计算：全行业累计对数收益的前缀和，一次算完，避免反复遍历
    # pre[j][k] = sum_{t<j} log1p(ret[t][k])
    import array
    pre = array.array("f", [0.0]) * ((ND + 1) * NI)
    for j in range(ND):
        base = j * NI
        nb = (j + 1) * NI
        c = col(mm, NI, ND, j)
        for k in range(NI):
            pre[nb + k] = pre[base + k] + math.log1p(c[k])

    def fwd(i, h):
        a = (i + 1) * NI; b = (min(i + 1 + h, ND)) * NI
        return [pre[b + k] - pre[a + k] for k in range(NI)]

    # 预处理：每个月份 -> 该月之后第一个交易日索引（二分）
    def first_after(ym):
        import bisect
        target = ym + "-32"
        k = bisect.bisect_left(dates, target)
        return k if k < ND else None

    # Q1
    print("\n=== Q1  货币扩张后，各行业相对全市场等权 ===")
    for h in (20, 60, 120):
        rows = []
        for ym, _ in events:
            i = first_after(ym)
            if i is None or i + h + 1 >= ND: continue
            r = fwd(i, h); b = sum(r) / NI
            rows.append([r[k] - b for k in range(NI)])
        n = len(rows)
        if n < 10: continue
        mu = [sum(r[k] for r in rows) / n for k in range(NI)]
        sd = []
        for k in range(NI):
            m = mu[k]
            v = sum((r[k] - m) ** 2 for r in rows) / (n - 1)
            sd.append(math.sqrt(v) if v > 0 else 1e-12)
        t = [mu[k] / (sd[k] / math.sqrt(n)) for k in range(NI)]
        order = sorted(range(NI), key=lambda k: -t[k])
        print("  %3d 日 (n=%d)  最强: %s" % (h, n, " | ".join(
            "%s %+.1f%% t=%.2f" % (names[k], mu[k] * 100, t[k]) for k in order[:4])))
        for key in ("股份制银行Ⅲ", "城商行Ⅲ", "农商行Ⅲ", "证券Ⅱ", "住宅开发", "半导体设备"):
            if key in names:
                k = names.index(key)
                print("      %-12s %+.2f%%  t=%+.2f" % (key, mu[k] * 100, t[k]))

    # Q3 银行对 M2 的敏感度 vs 其他行业（回归）
    print("\n=== Q3  各行业对 M2 同比变化的敏感度（月度收益对 M2 增速变化回归）===")
    res = []
    for key in names:
        k = names.index(key)
        xs, ys = [], []
        for m in range(1, len(yms)):
            ym = yms[m]
            i = first_after(ym)
            if i is None or i + 22 >= ND: continue
            r = fwd(i, 21)
            xs.append(m2s[ym] - m2s[yms[m - 1]])
            ys.append(r[k])
        n = len(xs)
        if n < 40: continue
        mx = sum(xs) / n; my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0: continue
        beta = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / sxx
        resid = [ys[i] - my - beta * (xs[i] - mx) for i in range(n)]
        s2 = sum(e * e for e in resid) / (n - 2)
        se = math.sqrt(s2 / sxx) if sxx > 0 else 1e9
        res.append((beta, beta / se if se > 0 else 0, key, n))
    res.sort(reverse=True)
    print("  M2 敏感度最高 8 个行业:")
    for b, t, nm, n in res[:8]:
        print("      %-16s beta=%+.3f  t=%+.2f  (n=%d)" % (nm, b, t, n))
    print("  M2 敏感度最低 4 个行业:")
    for b, t, nm, n in res[-4:]:
        print("      %-16s beta=%+.3f  t=%+.2f  (n=%d)" % (nm, b, t, n))
    for key in ("股份制银行Ⅲ", "城商行Ⅲ", "农商行Ⅲ"):
        for b, t, nm, n in res:
            if nm == key:
                rank = res.index((b, t, nm, n))
                print("  ** %-14s beta=%+.3f t=%+.2f  排名 %d/%d" % (nm, b, t, rank + 1, len(res)))


if __name__ == "__main__":
    main()
