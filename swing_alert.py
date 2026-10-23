"""
Swing alert: D1 + H4 FVG / OB zonalari, narx zonaga kirganda Telegramga xabar.
GitHub Actions uchun: har ishga tushganda bir marta tekshiradi va tugaydi.
"""
import os
import sys
import json
import requests
import pandas as pd
import yfinance as yf

# ===================== SOZLAMALAR =====================
TOKEN = os.getenv("TG_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")

PAIRS = {
    "NAS100": "NQ=F",
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDCAD": "USDCAD=X",
    "USDJPY": "USDJPY=X",
}

TREND_FILTER = True    # True: buy trendda faqat bullish, sell trendda faqat bearish zonalar
FRACTAL_SIDE = 5       # bir tomondan 5 sham (jami 11 sham), faqat D1 da
MIN_FVG_ATR = 0.5      # FVG o'lchami ATR(14) ning kamida shuncha qismi
ATR_LEN = 14
OB_SEARCH = 8          # OB ni impulsdan necha sham orqaga qidirish
H4_OFFSET = "0h"       # H4 sham chegarasi brokeringizga mos kelmasa: "1h", "2h", "3h"
STATE_FILE = "state.json"
MAX_KEYS = 2000
# =======================================================


def send(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": text}, timeout=20)
        if not r.ok:
            print("Telegram xato:", r.status_code, r.text[:200])
        return r.ok
    except Exception as e:
        print("Telegram xato:", e)
        return False


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
            s.setdefault("alerted", [])
            s.setdefault("trend", {})
            return s
        except Exception:
            pass
    return {"alerted": [], "trend": {}}


def save_state(state):
    state["alerted"] = state["alerted"][-MAX_KEYS:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def fmt(sym, x):
    if "JPY" in sym:
        d = 3
    elif sym in ("NAS100", "XAUUSD"):
        d = 2
    else:
        d = 5
    return f"{x:.{d}f}"


def get_data(ticker, tf):
    """(yopilgan shamlar, shakllanayotgan sham yoki None)"""
    now = pd.Timestamp.now(tz="UTC")
    if tf == "D1":
        raw = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=False)
    else:
        raw = yf.download(ticker, period="60d", interval="1h",
                          progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return None, None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["o", "h", "l", "c"]

    if tf == "D1":
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        is_closed = df.index < pd.Timestamp(now.date())
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.resample("4h", offset=H4_OFFSET).agg(
            {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
        is_closed = (df.index + pd.Timedelta(hours=4)) <= now

    closed = df[is_closed]
    forming = df[~is_closed]
    f = forming.iloc[-1] if len(forming) else None
    return closed, f


def find_ob(df, imp, kind):
    """Impuls shamidan oldingi oxirgi qarama-qarshi sham."""
    for j in range(imp - 1, max(imp - 1 - OB_SEARCH, -1), -1):
        r = df.iloc[j]
        if kind == "bull" and r.c < r.o:
            return float(r.l), float(r.h)
        if kind == "bear" and r.c > r.o:
            return float(r.l), float(r.h)
    return None


def find_zones(df):
    """Yopilgan shamlardan FVG va OB; faqat hali tegilmagan zonalar qaytadi."""
    atr = (df.h - df.l).rolling(ATR_LEN).mean()
    n = len(df)
    zones = []
    for i in range(2, n):
        a, c = df.iloc[i - 2], df.iloc[i]
        if c.l > a.h:
            kind, lo, hi = "bull", float(a.h), float(c.l)
        elif c.h < a.l:
            kind, lo, hi = "bear", float(c.h), float(a.l)
        else:
            continue
        if pd.isna(atr.iloc[i]) or (hi - lo) < MIN_FVG_ATR * atr.iloc[i]:
            continue

        later = df.iloc[i + 1:]

        def untouched(zlo, zhi):
            if later.empty:
                return True
            return not bool(((later.l <= zhi) & (later.h >= zlo)).any())

        t = str(df.index[i])
        if untouched(lo, hi):
            zones.append(dict(what="FVG", kind=kind, lo=lo, hi=hi, t=t))
        ob = find_ob(df, i - 1, kind)
        if ob and untouched(*ob):
            zones.append(dict(what="OB", kind=kind, lo=ob[0], hi=ob[1], t=t))
    return zones


def d1_trend(df):
    """Williams fractal (D1). Trend faqat sham yopilishi (tana) fractal ni buzganda o'zgaradi."""
    s = FRACTAL_SIDE
    trend, fh, fl = None, None, None
    for t in range(len(df)):
        c = float(df.c.iloc[t])
        if fh is not None and c > fh:
            trend, fh = "buy", None
        elif fl is not None and c < fl:
            trend, fl = "sell", None
        k = t - s
        if k >= s:
            w = df.iloc[k - s:k + s + 1]
            hk, lk = float(df.h.iloc[k]), float(df.l.iloc[k])
            if hk == float(w.h.max()) and int((w.h == hk).sum()) == 1:
                fh = hk
            if lk == float(w.l.min()) and int((w.l == lk).sum()) == 1:
                fl = lk
    return trend


def overlap(a, b):
    return a["lo"] <= b["hi"] and a["hi"] >= b["lo"]


def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("TG_TOKEN va TG_CHAT_ID o'rnatilmagan (GitHub Secrets)")

    state = load_state()
    if not state["alerted"] and not state["trend"]:
        send("✅ Swing alert bot ishga tushdi")

    for sym, ticker in PAIRS.items():
        try:
            d1c, d1f = get_data(ticker, "D1")
            h4c, h4f = get_data(ticker, "H4")
            if d1c is None or h4c is None or len(d1c) < 40 or len(h4c) < 30:
                print("Ma'lumot yetarli emas:", sym)
                continue

            # ---- D1 trend ----
            trend = d1_trend(d1c)
            prev = state["trend"].get(sym)
            if trend is not None and trend != prev:
                if prev is None:
                    ok = send(f"📊 {sym} D1 trend: {trend.upper()}")
                else:
                    ok = send(f"🔄 {sym} D1 trend o'zgardi: {prev.upper()} → {trend.upper()}")
                if ok:
                    state["trend"][sym] = trend

            # ---- zonalar ----
            zones = {"D1": find_zones(d1c), "H4": find_zones(h4c)}
            forming = {"D1": d1f, "H4": h4f}

            for tf in ("D1", "H4"):
                f = forming[tf]
                if f is None:
                    continue
                price = float(f.c)
                for z in zones[tf]:
                    if not (f.l <= z["hi"] and f.h >= z["lo"]):
                        continue
                    if TREND_FILTER and (trend is None or (trend == "buy") != (z["kind"] == "bull")):
                        continue
                    key = f"{sym}|{tf}|{z['what']}|{z['kind']}|{z['t']}"
                    if key in state["alerted"]:
                        continue

                    # ⭐: H4 zona bir tomondagi D1 zonaga tegib/ichida bo'lsa
                    star = None
                    if tf == "H4":
                        for zd in zones["D1"]:
                            if zd["kind"] == z["kind"] and overlap(z, zd):
                                star = zd
                                break

                    icon = "🟢" if z["kind"] == "bull" else "🔴"
                    side = "Bullish" if z["kind"] == "bull" else "Bearish"
                    mark = "⭐ " if star else ("🔥 " if tf == "D1" else "")
                    msg = (f"{mark}{icon} {sym} {tf} | {side} {z['what']} ga kirdi\n"
                           f"Zona: {fmt(sym, z['lo'])} - {fmt(sym, z['hi'])}\n"
                           f"Narx: {fmt(sym, price)}")
                    if star:
                        msg += (f"\nD1 {star['what']} ichida: "
                                f"{fmt(sym, star['lo'])} - {fmt(sym, star['hi'])}")
                    if TREND_FILTER and trend:
                        msg += f"\nD1 trend: {trend.upper()}"
                    if send(msg):
                        state["alerted"].append(key)
        except Exception as e:
            print(f"Xato {sym}:", repr(e))

    save_state(state)


if __name__ == "__main__":
    main()
