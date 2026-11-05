"""把 entity-graph 接进 quant：建立「概念 -> 行业 -> 个股」的暴露映射，并逐日存快照。

为什么必须先存快照
------------------
entity-graph 里所有节点的 fetched_at 都是同一天（2026-11-03），也就是说
它记的是"我现在知道什么"，不是"2015 年我知道什么"。
没有历史时点，就无法回测 EG 的判断力 —— 这是硬限制，不能含糊过去。

所以这里做两件事：
  1. 暴露映射：EG 的概念节点 -> 对应的行业板块 -> 成分股。今天是当前口径，
     但因为它依赖的是行业标签而不是未来价格，可以向前回溯（行业归属相对稳定）。
  2. 逐日快照：每次运行都把当天的图状态落盘，未来才有"那天我知道什么"。

快照格式：snapshots/YYYY-MM-DD.json，含节点数、边数、按域统计、因果边清单。
运行方式：python eg_intel.py snapshot
"""
import os, json, hashlib, time, datetime
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EG = os.path.join(HERE, "..", "entity-graph")
SNAP = os.path.join(HERE, "snapshots")

GRAPHS = ["political_economy.graph.json", "thesis_finance.graph.json", "thesis.graph.json"]
CAUSAL = {"causes", "because_of", "therefore", "relies_on", "controls", "exceeds", "becomes"}

# EG 的概念词 -> 行业板块关键词。这张表是"情报 -> 可交易标的"的桥。
# 不是拟合出来的：每一行都是"这个宏观/制度概念如果真的发生，会先动哪个行业"。
CONCEPT_TO_INDUSTRY = {
    "利率": ["银行", "保险", "证券"],
    "货币": ["银行", "证券"],
    "准备金": ["银行"],
    "债": ["证券", "银行"],
    "贷": ["银行"],
    "房": ["住宅开发", "商业地产", "房地产服务"],
    "地产": ["住宅开发", "商业地产"],
    "基建": ["基础建设", "水泥", "钢铁", "建筑装饰"],
    "钢铁": ["钢铁"],
    "煤炭": ["煤炭开采"],
    "能源": ["电力", "石油加工", "燃气"],
    "半导体": ["半导体", "半导体设备", "半导体材料", "模拟芯片设计"],
    "芯片": ["半导体", "集成电路封测"],
    "医药": ["化学制剂", "中药", "医疗器械"],
    "消费": ["食品饮料", "白酒", "家用电器"],
    "出口": ["航运港口", "物流", "纺织制造"],
    "汇率": ["航运港口", "纺织制造", "航空机场"],
    "关税": ["航运港口", "纺织制造", "汽车零部件"],
    "军": ["军工电子", "航空装备", "地面兵装"],
    "税": ["白酒", "食品饮料"],
    "通胀": ["食品饮料", "农产品加工"],
    "就业": ["人力资源服务", "职业教育"],
    "工资": ["人力资源服务"],
    "贸易": ["航运港口", "物流"],
    "制造": ["通用设备", "专用设备"],
    "电力": ["电力"],
    "新能源": ["电池", "光伏设备", "风电设备"],
    "汽车": ["乘用车", "汽车零部件"],
}


def load_graph(path):
    if not os.path.exists(path): return None
    return json.load(open(path, encoding="utf-8"))


def snapshot():
    os.makedirs(SNAP, exist_ok=True)
    today = datetime.date.today().isoformat()
    out = {"date": today, "graphs": {}, "causal": [], "concepts": {}}
    for g in GRAPHS:
        d = load_graph(os.path.join(EG, g))
        if not d: continue
        nodes = d.get("nodes") or []; edges = d.get("edges") or []
        out["graphs"][g] = {
            "nodes": len(nodes), "edges": len(edges),
            "domains": dict(Counter((n.get("attrs") or {}).get("domain") for n in nodes)),
            "rels": dict(Counter(e.get("rel") for e in edges)),
            "digest": hashlib.sha1(json.dumps(
                sorted(n["key"] for n in nodes), sort_keys=True).encode()).hexdigest()[:16],
        }
        nby = {n["key"]: n for n in nodes}
        for e in edges:
            if e.get("rel") in CAUSAL:
                a = nby.get(e["src"], {}).get("label", "")
                b = nby.get(e["dst"], {}).get("label", "")
                if a and b:
                    out["causal"].append({"rel": e["rel"], "from": a[:80], "to": b[:80],
                                          "status": (e.get("attrs") or {}).get("status")})
    # 概念命中：EG 的标签里出现了哪些行业关键词
    hit = defaultdict(list)
    for g, st in out["graphs"].items(): pass
    for g in GRAPHS:
        d = load_graph(os.path.join(EG, g))
        if not d: continue
        for n in (d.get("nodes") or []):
            lab = n.get("label") or ""
            for kw in CONCEPT_TO_INDUSTRY:
                if kw in lab: hit[kw].append(lab[:60])
    out["concepts"] = {k: {"n": len(v), "sample": v[:3]} for k, v in hit.items()}
    p = os.path.join(SNAP, today + ".json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("快照 %s：图 %d 个，因果边 %d 条，命中概念 %d 个" % (
        today, len(out["graphs"]), len(out["causal"]), len(out["concepts"])))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "snapshot":
        snapshot()
    else:
        s = snapshot()
        print(json.dumps(s["graphs"], ensure_ascii=False, indent=1)[:600])
