import json, os, csv

EG = r"D:\Users\china\Desktop\项目_开发\entity-graph"
GRAPH = os.path.join(EG, "political_economy.graph.json")
OUT = os.path.dirname(os.path.abspath(__file__))

CAUSAL = {"causes", "because_of", "therefore", "relies_on", "controls", "exceeds"}
INDUSTRY = ["房", "地产", "银行", "利率", "货币", "出口", "制造", "能源", "煤炭", "钢铁",
            "半导", "医药", "消费", "通胀", "汇率", "军", "税", "债", "贷", "贸易", "基建",
            "准备金", "工资", "就业"]

def evidence(edge, caveat_lines):
    a = edge.get("attrs") or {}
    lines = set(a.get("lines") or [])
    if lines & caveat_lines:
        return 0.5
    if a.get("status") == "asserted":
        return 1.0
    return 0.3

def main():
    d = json.load(open(GRAPH, encoding="utf-8"))
    meta = d.get("meta") or {}
    caveat_lines = set()
    for doc, m in (meta.get("docs") or {}).items():
        for c in m.get("caveats") or []:
            if c.get("line"):
                caveat_lines.add(int(c["line"]))
    label = {n["key"]: n["label"] for n in d["nodes"]}
    domain = {n["key"]: (n.get("attrs") or {}).get("domain", "") for n in d["nodes"]}
    rows = []
    for e in d["edges"]:
        if e["rel"] not in CAUSAL:
            continue
        rows.append({"src": label.get(e["src"], e["src"]), "rel": e["rel"],
                     "dst": label.get(e["dst"], e["dst"]), "evidence": evidence(e, caveat_lines),
                     "domain": domain.get(e["src"], "") or domain.get(e["dst"], ""),
                     "text": ((e.get("attrs") or {}).get("text") or "")[:120]})
    rows.sort(key=lambda r: -r["evidence"])
    with open(os.path.join(OUT, "eg_shocks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src", "rel", "dst", "evidence", "domain", "text"])
        w.writeheader(); w.writerows(rows)
    kw = {}
    for key, lab in label.items():
        hit = [k for k in INDUSTRY if k in lab]
        if hit:
            kw[key] = {"label": lab, "hit": hit, "domain": domain.get(key, "")}
    json.dump(kw, open(os.path.join(OUT, "eg_keywords.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("导出因果边 %d 条（原文断言 %d 条）" % (len(rows), sum(1 for r in rows if r["evidence"] == 1.0)))
    print("行业/宏观关键词节点 %d 个" % len(kw))

if __name__ == "__main__":
    main()
