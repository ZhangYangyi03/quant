"""事件流：从巨潮资讯拉真实公告，抽成「时点 - 主体 - 行业 - 事件类型」。

为什么用公告而不是新闻
----------------------
1. 公告带明确股票代码，能直接落到行业和个股，不用做实体识别猜。
2. 公告是确定性事件（增持、回购、业绩预告、重大合同），不是评论，可检验。
3. 巨潮的 seDate 参数能精确按日期回溯到 2010 年，新闻接口回溯不了。

事件类型只挑有明确方向的几类（这一步不拟合，是事先定好的）：
    增持     大股东/高管增持 -> 看多
    减持     减持           -> 看空
    回购     公司回购       -> 看多
    业绩预增 预增/扭亏      -> 看多
    业绩预减 预减/首亏      -> 看空
    中标合同 重大合同/中标  -> 看多
    股权激励 激励计划       -> 看多（利益绑定）

输出 events.jsonl，每行一条：{date, code, name, kind, dir, title}
"""
import os, json, time, urllib.request, urllib.error, re, sys, datetime, socket, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
}

# 事件类型 -> 方向。事先定，不看结果调。
KINDS = [
    ("增持", "up", ["增持"]),
    ("回购", "up", ["回购"]),
    ("业绩预增", "up", ["预增", "扭亏", "略增"]),
    ("中标合同", "up", ["中标", "重大合同", "签订", "框架协议"]),
    ("股权激励", "up", ["股权激励", "激励计划", "员工持股"]),
    ("减持", "down", ["减持"]),
    ("业绩预减", "down", ["预减", "首亏", "续亏", "略减"]),
    ("质押", "down", ["质押"]),
    ("诉讼", "down", ["诉讼", "仲裁", "立案"]),
]


def classify(title):
    for kind, d, kws in KINDS:
        for kw in kws:
            if kw in title:
                return kind, d
    return None, None


def q_day(day, page=1, size=30, retry=2):
    """单次请求。socket 超时兜底，不用线程（线程版会累积挂死）。"""
    data = ("pageNum=%d&pageSize=%d&column=szse&tabName=fulltext&plate=&stock=&searchkey="
            "&seDate=%s~%s&isHLtitle=true" % (page, size, day, day)).encode()
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(12)
    try:
        for a in range(retry):
            try:
                req = urllib.request.Request(URL, data=data, headers=HDR)
                raw = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", "ignore")
                j = json.loads(raw)
                return j.get("announcements") or [], j.get("totalpages") or 0
            except Exception:
                time.sleep(0.4)
        return [], 0
    finally:
        socket.setdefaulttimeout(old)


def fetch_day(day, out, stats):
    page = 1
    while True:
        anns, tp = q_day(day, page)
        if not anns: break
        for a in anns:
            title = re.sub("<[^>]+>", "", a.get("announcementTitle") or "")
            kind, dr = classify(title)
            if not kind: continue
            stats[kind] = stats.get(kind, 0) + 1
            out.write(json.dumps({
                "date": day, "code": a.get("secCode"), "name": a.get("secName"),
                "kind": kind, "dir": dr, "title": title[:120],
            }, ensure_ascii=False) + "\n")
        if page >= tp: break
        page += 1
        if page > 8: break   # 只扫前 240 条：关键事件集中在这里，后面是例行公告      # 单日上限，防止卡死
    return


def main(start, end):
    d0 = datetime.date.fromisoformat(start)
    d1 = datetime.date.fromisoformat(end)
    days = []
    d = d0
    while d <= d1:
        if d.weekday() < 5: days.append(d.isoformat())
        d += datetime.timedelta(days=1)
    print("待拉 %d 天" % len(days), flush=True)

    out = os.path.join(HERE, "events.jsonl")
    lock = threading.Lock()
    stats = {}
    t0 = time.time()
    done = [0]

    def one(day):
        buf = []
        class W:
            def write(self, x): buf.append(x)
            def flush(self): pass
        fetch_day(day, W(), stats)
        with lock:
            with open(out, "a", encoding="utf-8") as fh:
                fh.writelines(buf)
            done[0] += 1
            if done[0] % 25 == 0:
                print("  已 %d/%d 天  %.0f 条  %.0fs" % (
                    done[0], len(days), sum(stats.values()), time.time() - t0), flush=True)

    with ThreadPoolExecutor(max_workers=1) as ex:
        list(ex.map(one, days))
    print("完成 %d 天，事件 %d 条 %.0fs" % (len(days), sum(stats.values()), time.time() - t0))
    print("分布:", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
