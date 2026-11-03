"""事件流 v2：巨潮公告 -> 结构化事件。单文件，不 import 自己的模块。"""
import os, sys, json, re, time, urllib.request, datetime, socket

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
HDR = {"User-Agent": "Mozilla/5.0",
       "Content-Type": "application/x-www-form-urlencoded",
       "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice"}
KINDS = [("增持","up",["增持"]), ("回购","up",["回购"]),
         ("业绩预增","up",["预增","扭亏","略增"]),
         ("中标合同","up",["中标","重大合同","签订","框架协议"]),
         ("股权激励","up",["股权激励","激励计划","员工持股"]),
         ("减持","down",["减持"]),
         ("业绩预减","down",["预减","首亏","续亏","略减"]),
         ("质押","down",["质押"]),
         ("诉讼","down",["诉讼","仲裁","立案"])]

def cls(t):
    for k,d,ws in KINDS:
        for w in ws:
            if w in t: return k,d
    return None,None

def page(day,pg):
    d=("pageNum=%d&pageSize=30&column=szse&tabName=fulltext&plate=&stock=&searchkey="
       "&seDate=%s~%s&isHLtitle=true"%(pg,day,day)).encode()
    try:
        raw=urllib.request.urlopen(urllib.request.Request(URL,data=d,headers=HDR),timeout=10).read()
        j=json.loads(raw.decode("utf-8","ignore"))
        return j.get("announcements") or [], j.get("totalpages") or 0
    except Exception:
        return [], 0

def day_events(day,maxpage=5):
    out=[]; pg=1
    while pg<=maxpage:
        anns,tp=page(day,pg)
        if not anns: break
        for a in anns:
            t=re.sub("<[^>]+>","",a.get("announcementTitle") or "")
            k,dr=cls(t)
            if k: out.append({"date":day,"code":a.get("secCode"),"name":a.get("secName"),
                              "kind":k,"dir":dr,"title":t[:120]})
        if pg>=tp: break
        pg+=1
    return out

def main(s,e):
    d0=datetime.date.fromisoformat(s); d1=datetime.date.fromisoformat(e)
    out=os.path.join(HERE,"events.jsonl")
    # 断点续跑：已经拉过的日期跳过
    have=set()
    if os.path.exists(out):
        for line in open(out,encoding="utf-8"):
            try: have.add(json.loads(line)["date"])
            except: pass
    days=[]; d=d0
    while d<=d1:
        if d.weekday()<5 and d.isoformat() not in have: days.append(d.isoformat())
        d+=datetime.timedelta(days=1)
    print("待拉 %d 天（已跳过 %d 天）"%(len(days),len(have)),flush=True)
    t0=time.time(); total=0
    for i,day in enumerate(days):
        ev=day_events(day)
        if ev:
            with open(out,"a",encoding="utf-8") as fh:
                for x in ev: fh.write(json.dumps(x,ensure_ascii=False)+"\n")
            total+=len(ev)
        if (i+1)%25==0:
            print("  %d/%d 天 %d 条 %.0fs"%(i+1,len(days),total,time.time()-t0),flush=True)
    print("完成 %d 天 %d 条 %.0fs"%(len(days),total,time.time()-t0))

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2])
