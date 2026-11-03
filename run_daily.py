"""每个交易日收盘后跑一次：刷新 ETF 最新 K 线 -> 专业版信号 -> 纸面账 -> 日志。"""
import os, sys, sqlite3, time, json, urllib.request, datetime, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data")
LOG = os.path.join(HERE, "daily.log")
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
UNIVERSE = ["510300.SH", "510500.SH", "159915.SZ", "512880.SH", "518880.SH", "511010.SH", "510880.SH", "513100.SH"]
DEFENSIVE = "511010.SH"


def refresh():
    ok = 0
    for c in UNIVERSE:
        secid = ("1." if c.endswith(".SH") else "0.") + c[:6]
        u = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=%s"
             "&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59"
             "&klt=101&fqt=1&lmt=6000&end=20500101" % secid)
        for i in range(5):
            try:
                ks = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=25).read())["data"]["klines"]
                rows = [k.split(",")[:9] for k in ks]
                with open(os.path.join(DATA, c + ".csv"), "w", encoding="utf-8", newline="") as f:
                    f.write("date,open,close,high,low,volume,amount,amplitude,pct\n")
                    for r in rows:
                        f.write(",".join(r) + "\n")
                ok += 1
                break
            except Exception:
                time.sleep(2 + i)
        time.sleep(0.6)
    return ok


def build_signal():
    """专业版信号：低波 60 日（回测里唯一通过 Deflated Sharpe 的一族）。"""
    from engine import Bars
    import signals_pro as S
    bars = Bars()
    for c in UNIVERSE:
        p = os.path.join(DATA, c + ".csv")
        if os.path.exists(p):
            bars.load_csv(c, p)
    bars.finalize()
    bars.build_series()
    date = [d for d in bars.dates if all(bars.get(c, d) for c in UNIVERSE)][-1]
    base = S.low_vol(60)
    sc = base({c: bars.series[c][:bars.close_upto(c, date)] for c in UNIVERSE}, UNIVERSE, date)
    ranked = sorted(sc.items(), key=lambda x: -x[1])
    hold = [c for c, _ in ranked[:2]]
    return {"as_of": date, "rank": [(c, round(v, 4)) for c, v in ranked], "hold": hold}


def main():
    t0 = time.time()
    got = refresh()
    sig = build_signal()
    cash, mv = 1.00, 0.0
    rec = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "refreshed": got,
           "as_of": sig["as_of"], "strategy": "lowvol60/k2", "hold": sig["hold"],
           "rank": sig["rank"], "nav": {"cash": cash, "mv": mv, "total": round(cash + mv, 4)},
           "elapsed_s": round(time.time() - t0, 1)}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
