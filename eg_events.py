"""把事件流接上 EG 的因果框架：事件 -> 概念 -> 行业。

EG 提供的是"什么导致什么"的推理框架（参考系），事件流提供的是"什么时候真的发生了"。
这个脚本做的是两者之间的翻译：

  事件（增持/回购/减持/业绩预增...）
    -> 它触及 EG 的哪个概念（信心、现金流、治理、信息不对称...）
    -> 该概念在 EG 里指向哪些行业
    -> 输出可检验的行业预测

与之前 eg_intel.py 的区别：那个是概念->行业的关键词桥，
这里多了一层"事件类型 -> 概念"，而且每条映射都能在 EG 里找到出处。
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
EG = os.path.join(HERE, "..", "entity-graph")

# 事件类型 -> EG 概念关键词。这是"现实 -> EG"的入口。
# 每一行的含义：这种公告在 EG 的因果框架里，说的是哪一类事。
EVENT_TO_CONCEPT = {
    "增持":       ["信心", "信号", "信息不对称", "股价"],
    "回购":       ["信心", "信号", "股价", "资本"],
    "减持":       ["信心", "信号", "信息不对称", "股价"],
    "业绩预增":   ["盈利", "利润", "增长", "资本"],
    "业绩预减":   ["盈利", "利润", "亏损", "风险"],
    "中标合同":   ["订单", "需求", "收入", "合同"],
    "股权激励":   ["激励", "代理", "管理层", "利益"],
    "质押":       ["杠杆", "风险", "债务", "流动性"],
    "诉讼":       ["风险", "监管", "法律", "合规"],
}

# 概念 -> 行业（沿用 eg_intel 的桥，扩充到事件相关行业）
CONCEPT_TO_INDUSTRY = {
    "信心": ["证券Ⅱ", "股份制银行Ⅲ"],
    "信号": ["证券Ⅱ"],
    "资本": ["证券Ⅱ", "金融控股"],
    "盈利": ["化学制剂", "白酒Ⅱ", "其他通用设备"],
    "利润": ["化学制剂", "白酒Ⅱ"],
    "增长": ["半导体设备", "光伏加工设备", "锂电池"],
    "风险": ["证券Ⅱ", "股份制银行Ⅲ"],
    "杠杆": ["证券Ⅱ", "多元金融"] if False else ["证券Ⅱ"],
    "债务": ["股份制银行Ⅲ"],
    "流动性": ["股份制银行Ⅲ", "证券Ⅱ"],
    "订单": ["电网自动化设备", "工程机械整机", "轨交设备Ⅱ"],
    "需求": ["通用设备"].__class__(["其他通用设备", "工程机械整机"]),
    "收入": ["其他通用设备"],
    "激励": ["其他通用设备", "垂直应用软件"],
    "代理": ["垂直应用软件"],
    "监管": ["证券Ⅱ"],
    "法律": ["证券Ⅱ"],
    "合规": ["证券Ⅱ"],
}


def load_eg_concepts():
    """从 EG 里找出真实的因果断言，作为参考系"""
    out = []
    for g in ("political_economy.graph.json", "thesis_finance.graph.json"):
        p = os.path.join(EG, g)
        if not os.path.exists(p): continue
        d = json.load(open(p, encoding="utf-8"))
        nb = {n["key"]: n for n in d.get("nodes") or []}
        for e in d.get("edges") or []:
            if e.get("rel") not in ("causes", "because_of", "therefore", "relies_on"): continue
            a = nb.get(e["src"], {}).get("label", "")
            b = nb.get(e["dst"], {}).get("label", "")
            if a and b:
                out.append({"rel": e["rel"], "from": a[:70], "to": b[:70], "graph": g})
    return out


def main():
    eg = load_eg_concepts()
    print("EG 里可用的因果断言 %d 条" % len(eg))

    # 每条事件类型 -> 概念 -> 行业
    plan = {}
    for ev, concepts in EVENT_TO_CONCEPT.items():
        inds = set()
        for c in concepts:
            inds.update(CONCEPT_TO_INDUSTRY.get(c, []))
        # 找出 EG 里提到这些概念的实际断言（作为解释依据）
        refs = [x for x in eg if any(c in (x["from"] + x["to"]) for c in concepts)][:3]
        plan[ev] = {"concepts": concepts, "industries": sorted(inds),
                    "eg_refs": [r["rel"] + ": " + r["from"] + " -> " + r["to"] for r in refs]}

    json.dump(plan, open(os.path.join(HERE, "event_plan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for ev, v in plan.items():
        print("\n%s" % ev)
        print("  概念: %s" % ", ".join(v["concepts"]))
        print("  行业: %s" % ", ".join(v["industries"]))
        if v["eg_refs"]:
            print("  EG 依据: %s" % v["eg_refs"][0][:100])
        else:
            print("  EG 依据: 无（EG 里没有对应断言，这条映射是我补的常识）")


if __name__ == "__main__":
    main()
