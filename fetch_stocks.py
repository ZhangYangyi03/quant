"""全市场日线抓取：分片、断点续抓、带进度文件。"""
import os, sys, json, time, urllib.request, csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "stockdata")
os.makedirs(DATA, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
PROG = os.path.join(DATA, "_progress.json")

def log(msg):
    with open(os.path.join(DATA, "_fetch.log"), "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")

def get_universe():
    p = os.path.join(DATA, "_universe.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    codes = {}
    for fs, label in (("m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23", "A股"),):
        for pn in range(1, 80):
            u = ("https://push2.eastmoney.com/api/qt/clist/get?pn=%d&pz=100&po=1&np=1&fltt=2&invt=2&stringify=1"
                 "&fid=f3&fs=%s&fields=f12,f13,f14,f26" % (pn, fs))
            try:
                d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=25).read())
            except Exception:
                time.sleep(2); continue
            diff = (d.get("data") or {}).get("diff") or []
            if not diff: break
            for r in diff:
                c = r["f12"]
                if isinstance(c, str) and len(c) == 6 and c.isdigit():
                    # f13 是交易所：0=深 1=沪；f12 可能被解析成数字，补零
                    mkt = str(r.get("f13", 1 if c[0] == "6" else 0))
                    codes[c] = {"code": c, "name": r.get("f14"), "secid": mkt + "." + c, "list": r.get("f26")}
                    assert c == r["f12"], (c, r["f12"])
            time.sleep(0.15)
    json.dump(codes, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return codes

def kline(secid, lmt=6000):
    u = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=%s"
         "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59"
         "&klt=101&fqt=1&lmt=%d&end=20500101" % (secid, lmt))
    for i in range(4):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=25).read())["data"]["klines"]
        except Exception:
            time.sleep(1.5 + i*1.5)
    return None

def main(limit=None):
    uni = get_universe()
    log("universe %d" % len(uni))
    done = set(json.load(open(PROG, encoding="utf-8"))) if os.path.exists(PROG) else set()
    items = [v for k, v in uni.items() if k not in done]
    if limit: items = items[:limit]
    ok = fail = 0
    t0 = time.time()
    for i, it in enumerate(items):
        ks = kline(it["secid"])
        if not ks:
            fail += 1
            continue
        with open(os.path.join(DATA, it["code"] + ".csv"), "w", encoding="utf-8", newline="") as f:
            f.write("date,open,close,high,low,volume,amount,amplitude,pct\n")
            for k in ks:
                f.write(",".join(k.split(",")[:9]) + "\n")
        done.add(it["code"]); ok += 1
        if ok % 100 == 0:
            json.dump(sorted(done), open(PROG, "w", encoding="utf-8"))
            log("done %d ok %d fail %d elapsed %.0fs" % (len(done), ok, fail, time.time()-t0))
        time.sleep(0.12)
    json.dump(sorted(done), open(PROG, "w", encoding="utf-8"))
    log("FINISH ok=%d fail=%d total=%d" % (ok, fail, len(done)))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
