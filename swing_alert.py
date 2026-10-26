"""
Swing alert v2: D1 + H4 FVG / OB zonalari.
- Narx zonaga kirganda grafik rasmi bilan xabar
- M30 / H1 da MSS (sham tanasi bilan struktura buzilishi) bo'lsa "TASDIQ" xabari
- 3 kuchli yangilik (NFP, CPI, FOMC): oldindan eslatma + alertlarda ogohlantirish
- Zona eskirishi: D1 60 kun, H4 30 kun
- Kunlik heartbeat, ma'lumot uzilsa ogohlantirish
GitHub Actions uchun: har ishga tushganda bir marta tekshiradi va tugaydi.
"""
import io
import os
import sys
import json
from zoneinfo import ZoneInfo

import numpy as np
import requests
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# --- zona eskirishi (kun) ---
ZONE_MAX_AGE_DAYS = {"D1": 60, "H4": 30}

# --- MSS tasdig'i ---
CONFIRM_TFS = ("H1",)                       # M30 shovqinli; ikkalasi uchun: ("M30", "H1")
MIN_DEPTH = 0.5                             # zonaning kamida shuncha qismi to'lgach MSS qidiriladi (0.5 = 50%, 0 = istalgan teginish)
STOP_BUFFER_ATR = 0.3                       # stop darajasi uchun ATR buferi
LTF_SWING_SIDE = 2                          # kichik fractal: bir tomondan 2 sham
LTF_LOOKBACK = 40                           # sweep dan oldin necha sham ichida swing qidiriladi
PENDING_DAYS = {"D1": 7, "H4": 3}           # tasdiq necha kun kutiladi

# --- yangilik (faqat USD, High impact) ---
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_CCY = "USD"
NEWS_KEYS = {                               # nom -> sarlavhada qidiriladigan so'zlar
    "NFP": ["non-farm employment change"],
    "CPI": ["cpi"],
    "FOMC": ["fomc statement", "fomc press conference", "federal funds rate"],
}
NEWS_REFETCH_H = 6                          # yangilik jadvalini necha soatda bir yangilash
NEWS_NOTICE_HOURS = (24, 1)                 # yangilikdan necha soat oldin eslatma
NEWS_WARN_BEFORE_H = 2                      # alertga "yangilik" ogohlantirishi: oldin
NEWS_WARN_AFTER_H = 2                       # ... va keyin
NEWS_BLOCK = False                          # True: yangilik oynasida zona alertlari yuborilmaydi

# --- heartbeat / xato ---
HEARTBEAT_HOUR = 9                          # Toshkent vaqti bilan, kuniga bir marta
DATA_FAIL_LIMIT = 6                         # ketma-ket shuncha marta ma'lumot kelmasa ogohlantirish
TZ = ZoneInfo("Asia/Tashkent")
# =======================================================


# ---------------------- Telegram ----------------------
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


def send_photo(png, caption):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption[:1000]},
                          files={"photo": ("chart.png", png, "image/png")}, timeout=40)
        if not r.ok:
            print("Telegram foto xato:", r.status_code, r.text[:200])
        return r.ok
    except Exception as e:
        print("Telegram foto xato:", e)
        return False


def notify(text, png=None):
    """Rasm bilan yuboradi; rasm bo'lmasa yoki yuborilmasa oddiy matn."""
    if png is not None and send_photo(png, text):
        return True
    return send(text)


# ---------------------- State ----------------------
def load_state():
    s = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                s = json.load(f)
        except Exception:
            s = {}
    s.setdefault("alerted", [])
    s.setdefault("trend", {})
    s.setdefault("pending", [])
    s.setdefault("fails", {})
    s.setdefault("news", {})
    s.setdefault("news_sent", [])
    s.setdefault("hb", None)
    return s


def save_state(state):
    state["alerted"] = state["alerted"][-MAX_KEYS:]
    state["pending"] = state["pending"][-100:]
    state["news_sent"] = state["news_sent"][-200:]
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


def local_str(ts, with_date=False):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert(TZ)
    return t.strftime("%d.%m %H:%M" if with_date else "%H:%M")


# ---------------------- Ma'lumot ----------------------
def dl(ticker, interval, period):
    raw = yf.download(ticker, period=period, interval=interval,
                      progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["o", "h", "l", "c"]
    if interval == "1d":
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    return df


def split_d1(df, now):
    is_closed = df.index < pd.Timestamp(now.date())
    closed, forming = df[is_closed], df[~is_closed]
    return closed, (forming.iloc[-1] if len(forming) else None)


def split_intraday(df, delta, now):
    is_closed = (df.index + delta) <= now
    closed, forming = df[is_closed], df[~is_closed]
    return closed, (forming.iloc[-1] if len(forming) else None)


def to_h4(h1):
    return h1.resample("4h", offset=H4_OFFSET).agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def age_days(ts, now):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return (now - t).total_seconds() / 86400


# ---------------------- Zonalar ----------------------
def find_ob(df, imp, kind):
    """Impuls shamidan oldingi oxirgi qarama-qarshi sham."""
    for j in range(imp - 1, max(imp - 1 - OB_SEARCH, -1), -1):
        r = df.iloc[j]
        if kind == "bull" and r.c < r.o:
            return float(r.l), float(r.h)
        if kind == "bear" and r.c > r.o:
            return float(r.l), float(r.h)
    return None


def find_zones(df, max_age, now):
    """Yopilgan shamlardan FVG va OB; faqat tegilmagan va eskirmagan zonalar."""
    atr = (df.h - df.l).rolling(ATR_LEN).mean()
    n = len(df)
    zones = []
    for i in range(2, n):
        if age_days(df.index[i], now) > max_age:
            continue
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


# ---------------------- MSS (tasdiq) ----------------------
def swing_idx(df, side, kind):
    """Kichik fractal pozitsiyalari ('high' yoki 'low')."""
    v = (df.h if kind == "high" else df.l).values
    out = []
    for k in range(side, len(v) - side):
        w = v[k - side:k + side + 1]
        ext = w.max() if kind == "high" else w.min()
        if v[k] == ext and int((w == v[k]).sum()) == 1:
            out.append(k)
    return out


def check_mss(df, p):
    """
    df: pastki TF (M30/H1) yopilgan shamlari.
    Bull zona: zonaga tegib sweep low hosil bo'ladi, undan oldingi oxirgi swing high
    sham TANASI (close) bilan tepaga yopilsa -> MSS.
    Bear zona: aksincha.
    """
    t0 = pd.Timestamp(p["t0"])
    pos = np.where(df.index >= t0)[0]
    if len(pos) == 0:
        return None
    s = int(pos[0])
    bull = p["kind"] == "bull"
    seg = df.iloc[s:]
    width = p["hi"] - p["lo"]
    if bull:   # narx tepadan tushadi: zonaning MIN_DEPTH qismigacha kirishi kerak
        touch = (seg.l <= p["hi"] - MIN_DEPTH * width).values
    else:
        touch = (seg.h >= p["lo"] + MIN_DEPTH * width).values
    if not touch.any():
        return None
    first = s + int(np.argmax(touch))
    seg2 = df.iloc[first:]
    L = first + int(np.argmin(seg2.l.values) if bull else np.argmax(seg2.h.values))

    sw = [k for k in swing_idx(df, LTF_SWING_SIDE, "high" if bull else "low")
          if L - LTF_LOOKBACK <= k < L]
    if not sw:
        return None
    rk = sw[-1]
    ref = float(df.h.iloc[rk] if bull else df.l.iloc[rk])
    after = df.iloc[L + 1:]
    if after.empty:
        return None
    hit = (after.c > ref).values if bull else (after.c < ref).values
    if not hit.any():
        return None
    m = L + 1 + int(np.argmax(hit))
    atr = float((df.h - df.l).rolling(ATR_LEN).mean().iloc[m])
    return dict(ref=ref, ref_k=rk, L=L, m=m, atr=atr,
                sweep=float(df.l.iloc[L] if bull else df.h.iloc[L]),
                close=float(df.c.iloc[m]), ts=df.index[m])


# ---------------------- Grafik ----------------------
def make_chart(df, title, zone=None, hlines=(), mark=None, n=70):
    """df: o,h,l,c. mark: (timestamp, matn). hlines: [(narx, matn, rang)]."""
    try:
        d = df.iloc[-n:]
        if len(d) < 5:
            return None
        bg, fg = "#0f1419", "#c9d1d9"
        up, dn = "#2ebd85", "#f6465d"
        fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        O, H, L, C = (d[k].values.astype(float) for k in ("o", "h", "l", "c"))
        rng = float(H.max() - L.min()) or 1.0
        for i in range(len(d)):
            col = up if C[i] >= O[i] else dn
            ax.vlines(i, L[i], H[i], color=col, lw=1, zorder=3)
            body = max(abs(C[i] - O[i]), rng * 0.003)
            ax.add_patch(Rectangle((i - 0.33, min(O[i], C[i])), 0.66, body,
                                   color=col, zorder=4))
        lo_y, hi_y = float(L.min()), float(H.max())
        if zone:
            zc = up if zone["kind"] == "bull" else dn
            ax.axhspan(zone["lo"], zone["hi"], color=zc, alpha=0.25, zorder=1)
            lo_y, hi_y = min(lo_y, zone["lo"]), max(hi_y, zone["hi"])
        for price, label, color in hlines:
            ax.axhline(price, color=color, ls="--", lw=1, zorder=2)
            ax.text(len(d) + 1.5, price, label, color=color, va="bottom",
                    ha="right", fontsize=8)
            lo_y, hi_y = min(lo_y, price), max(hi_y, price)
        if mark is not None and mark[0] in d.index:
            i = d.index.get_loc(mark[0])
            ax.annotate(mark[1], (i, H[i]), xytext=(i, H[i] + rng * 0.06),
                        color="#f0b90b", fontsize=9, ha="center",
                        arrowprops=dict(arrowstyle="->", color="#f0b90b"))
            hi_y = max(hi_y, H[i] + rng * 0.08)
        pad = (hi_y - lo_y) * 0.06
        ax.set_ylim(lo_y - pad, hi_y + pad)
        ax.set_xlim(-1, len(d) + 2)
        step = max(1, len(d) // 7)
        ticks = list(range(0, len(d), step))
        if d.index.tz is None:
            labels = [d.index[i].strftime("%d.%m") for i in ticks]
        else:
            labels = [d.index[i].tz_convert(TZ).strftime("%d.%m %H:%M") for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, color=fg, fontsize=8)
        ax.tick_params(axis="y", colors=fg, labelsize=8)
        ax.yaxis.tick_right()
        ax.grid(color="#222a33", lw=0.5, zorder=0)
        for sp in ax.spines.values():
            sp.set_color("#222a33")
        ax.set_title(title, color=fg, fontsize=11, loc="left")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=bg)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        print("Grafik xato:", repr(e))
        plt.close("all")
        return None


def with_forming(closed, forming):
    if forming is None:
        return closed
    row = forming.to_frame().T.astype(float)
    return pd.concat([closed, row])


# ---------------------- Yangilik ----------------------
def fetch_news():
    """3 kuchli yangilik (USD, High). Xato bo'lsa None."""
    try:
        r = requests.get(NEWS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print("Yangilik xato:", repr(e))
        return None
    out, seen = [], set()
    for e in data:
        if e.get("country") != NEWS_CCY or e.get("impact") != "High":
            continue
        title = (e.get("title") or "").lower()
        kind = next((k for k, ws in NEWS_KEYS.items() if any(w in title for w in ws)), None)
        if not kind:
            continue
        try:
            t = pd.Timestamp(e["date"]).tz_convert("UTC")
        except Exception:
            continue
        if (kind, t) in seen:
            continue
        seen.add((kind, t))
        out.append(dict(kind=kind, title=e.get("title", kind), t=t.isoformat()))
    return out


def get_news(state, now):
    c = state["news"]
    fetched = pd.Timestamp(c["fetched"]) if c.get("fetched") else None
    if fetched is None or (now - fetched) > pd.Timedelta(hours=NEWS_REFETCH_H):
        ev = fetch_news()
        if ev is None:
            # xato: 1 soatdan keyin qayta urinadi, eski keshdan foydalanadi
            c["fetched"] = (now - pd.Timedelta(hours=NEWS_REFETCH_H - 1)).isoformat()
        else:
            c["fetched"] = now.isoformat()
            c["events"] = ev
    return c.get("events", [])


def news_near(events, now):
    """Hozir yangilik oynasida bo'lsa (oldin/keyin) o'sha yangilik."""
    for e in events:
        t = pd.Timestamp(e["t"])
        if t - pd.Timedelta(hours=NEWS_WARN_BEFORE_H) <= now <= t + pd.Timedelta(hours=NEWS_WARN_AFTER_H):
            return e
    return None


def news_warn_line(e, now):
    if e is None:
        return ""
    t = pd.Timestamp(e["t"])
    mins = int((t - now).total_seconds() / 60)
    when = f"{mins} daqiqadan keyin" if mins > 0 else f"{-mins} daqiqa oldin chiqqan"
    return f"\n⚠️ YANGILIK: {e['kind']} {local_str(t)} (Toshkent) - {when}"


def news_notices(state, events, now):
    for e in events:
        t = pd.Timestamp(e["t"])
        hours = (t - now).total_seconds() / 3600
        if hours <= 0:
            continue
        th = next((x for x in sorted(NEWS_NOTICE_HOURS) if hours <= x), None)
        if th is None:
            continue
        key = f"{e['kind']}|{e['t']}|{th}"
        if key in state["news_sent"]:
            continue
        if th >= 24:
            head = "📅 Ertaga kuchli yangilik"
        else:
            head = f"⏰ {th} soatdan kam qoldi"
        msg = (f"{head}: {e['kind']} ({e['title']})\n"
               f"Vaqt: {local_str(t, True)} (Toshkent)\n"
               f"Yangilik atrofida zona alertlariga ishonmang, narx sakraydi.")
        if send(msg):
            state["news_sent"].append(key)


# ---------------------- Heartbeat / ma'lumot nazorati ----------------------
def heartbeat(state, events, now):
    local = now.tz_convert(TZ)
    day = local.strftime("%Y-%m-%d")
    if local.hour < HEARTBEAT_HOUR or state.get("hb") == day:
        return
    tr = ", ".join(f"{s} {t.upper()}" for s, t in state["trend"].items()) or "-"
    nxt = [e for e in events if pd.Timestamp(e["t"]) > now]
    nxt.sort(key=lambda e: e["t"])
    msg = (f"✅ Bot ishlayapti ({local.strftime('%d.%m')})\n"
           f"D1 trend: {tr}\n"
           f"Tasdiq kutilayotgan zonalar: {len(state['pending'])}")
    if nxt:
        msg += f"\nKeyingi yangilik: {nxt[0]['kind']} {local_str(nxt[0]['t'], True)}"
    if send(msg):
        state["hb"] = day


def mark_data(state, sym, ok):
    f = state["fails"]
    if ok:
        if f.get(sym, 0) >= DATA_FAIL_LIMIT:
            send(f"✅ {sym}: ma'lumot qayta tiklandi")
        f[sym] = 0
    else:
        f[sym] = f.get(sym, 0) + 1
        if f[sym] == DATA_FAIL_LIMIT:
            send(f"⚠️ {sym}: ma'lumot {DATA_FAIL_LIMIT} marta ketma-ket kelmadi "
                 f"(Yahoo muammosi bo'lishi mumkin)")


# ---------------------- Tasdiq jarayoni ----------------------
def since(df, t0):
    if df.index.tz is None:
        t = t0.tz_convert("UTC").tz_localize(None).normalize()
    else:
        t = t0
    return df[df.index >= t]


def process_pending(state, sym, ticker, now, d1c, h4c, h1c, events):
    mine = [p for p in state["pending"] if p["sym"] == sym]
    if not mine:
        return
    ltf = {"H1": h1c}
    if "M30" in CONFIRM_TFS:
        try:
            m30 = dl(ticker, "30m", "30d")
            if m30 is not None:
                ltf["M30"], _ = split_intraday(m30, pd.Timedelta(minutes=30), now)
        except Exception as e:
            print(f"M30 xato {sym}:", repr(e))

    for p in mine:
        t0 = pd.Timestamp(p["t0"])
        if now > pd.Timestamp(p["exp"]):
            state["pending"].remove(p)
            continue
        # zona o'z TF sida to'liq buzildimi (tana bilan narigi chegara)
        own = d1c if p["tf"] == "D1" else h4c
        s = since(own, t0)
        broken = (s.c < p["lo"]).any() if p["kind"] == "bull" else (s.c > p["hi"]).any()
        if broken:
            state["pending"].remove(p)
            send(f"❌ {sym} {p['tf']} {p['what']} zonasi tana bilan buzildi, tasdiq bekor qilindi\n"
                 f"Zona: {fmt(sym, p['lo'])} - {fmt(sym, p['hi'])}")
            continue

        for name in CONFIRM_TFS:
            df = ltf.get(name)
            if df is None or len(df) < 20:
                continue
            res = check_mss(df, p)
            if not res:
                continue
            bull = p["kind"] == "bull"
            icon = "🟢" if bull else "🔴"
            side = "Bullish" if bull else "Bearish"
            lvl = "yuqori" if bull else "pastki"
            sweep_name = "Sweep low" if bull else "Sweep high"
            width = (p["hi"] - p["lo"]) or 1e-9
            atr = res["atr"] if res["atr"] == res["atr"] else 0.0   # nan -> 0
            buf = STOP_BUFFER_ATR * atr
            if bull:
                depth = (p["hi"] - min(res["sweep"], p["hi"])) / width
                far, inside = p["lo"], res["sweep"] > p["lo"]
                st_sweep, st_zone = res["sweep"] - buf, far - buf
            else:
                depth = (min(res["sweep"], p["hi"]) - p["lo"]) / width
                far, inside = p["hi"], res["sweep"] < p["hi"]
                st_sweep, st_zone = res["sweep"] + buf, far + buf
            depth = max(0.0, min(1.0, depth))

            def atr_txt(stop):
                return f" ({abs(res['close'] - stop) / atr:.1f} ATR)" if atr else ""

            msg = (f"✅ TASDIQ | {icon} {sym} {p['tf']} {side} {p['what']}\n"
                   f"{name} da MSS: sham tanasi {fmt(sym, res['ref'])} ({lvl} swing) ni "
                   f"{'tepaga' if bull else 'pastga'} yopdi\n"
                   f"Zona: {fmt(sym, p['lo'])} - {fmt(sym, p['hi'])} (to'lgan: {depth * 100:.0f}%)\n"
                   f"{sweep_name}: {fmt(sym, res['sweep'])}\n"
                   f"Yopilish: {fmt(sym, res['close'])}\n"
                   f"Stop (sweep ortida): {fmt(sym, st_sweep)}{atr_txt(st_sweep)}\n"
                   f"Stop (zona ortida): {fmt(sym, st_zone)}{atr_txt(st_zone)}")
            if inside:
                msg += "\n⚠️ Sweep zona ichida: zona to'liq to'lmagan, sweep stopi tez olinishi mumkin"
            msg += news_warn_line(news_near(events, now), now)
            chart = make_chart(
                df.iloc[:res["m"] + 1 + 3] if res["m"] + 4 <= len(df) else df,
                f"{sym} {name} | MSS tasdiq | {side} {p['what']} ({p['tf']})",
                zone=p,
                hlines=[(res["ref"], "MSS", "#f0b90b")],
                mark=(res["ts"], "MSS"))
            if notify(msg, chart):
                state["pending"].remove(p)
            break


# ---------------------- Asosiy ----------------------
def run(state, now):
    events = get_news(state, now)
    news_notices(state, events, now)
    heartbeat(state, events, now)

    for sym, ticker in PAIRS.items():
        try:
            d1 = dl(ticker, "1d", "1y")
            h1 = dl(ticker, "1h", "60d")
            if d1 is None or h1 is None:
                print("Ma'lumot yo'q:", sym)
                mark_data(state, sym, False)
                continue
            d1c, d1f = split_d1(d1, now)
            h4c, h4f = split_intraday(to_h4(h1), pd.Timedelta(hours=4), now)
            h1c, _ = split_intraday(h1, pd.Timedelta(hours=1), now)
            if len(d1c) < 40 or len(h4c) < 30:
                print("Ma'lumot yetarli emas:", sym)
                mark_data(state, sym, False)
                continue
            mark_data(state, sym, True)

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

            # ---- zonalar (D1 60 kun, H4 30 kun) ----
            zones = {"D1": find_zones(d1c, ZONE_MAX_AGE_DAYS["D1"], now),
                     "H4": find_zones(h4c, ZONE_MAX_AGE_DAYS["H4"], now)}
            forming = {"D1": d1f, "H4": h4f}
            closed = {"D1": d1c, "H4": h4c}
            nw = news_near(events, now)

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
                    if NEWS_BLOCK and nw is not None:
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
                    msg += (f"\n⏳ Tasdiq: zona kamida {MIN_DEPTH * 100:.0f}% to'lgach "
                            f"{'/'.join(CONFIRM_TFS)} da MSS kutilmoqda")
                    msg += news_warn_line(nw, now)

                    chart = make_chart(
                        with_forming(closed[tf], f),
                        f"{sym} {tf} | {side} {z['what']}", zone=z)
                    if notify(msg, chart):
                        state["alerted"].append(key)
                        state["pending"].append(dict(
                            key=key, sym=sym, tf=tf, what=z["what"], kind=z["kind"],
                            lo=z["lo"], hi=z["hi"],
                            t0=(now - pd.Timedelta(hours=2)).isoformat(),
                            exp=(now + pd.Timedelta(days=PENDING_DAYS[tf])).isoformat()))

            # ---- MSS tasdig'i ----
            process_pending(state, sym, ticker, now, d1c, h4c, h1c, events)
        except Exception as e:
            print(f"Xato {sym}:", repr(e))
            mark_data(state, sym, False)


def main():
    if not TOKEN or not CHAT_ID:
        sys.exit("TG_TOKEN va TG_CHAT_ID o'rnatilmagan (GitHub Secrets)")
    state = load_state()
    if not state["alerted"] and not state["trend"]:
        send("✅ Swing alert bot ishga tushdi")
    now = pd.Timestamp.now(tz="UTC")
    try:
        run(state, now)
    finally:
        save_state(state)   # xato bo'lsa ham state yo'qolmasin (takroriy alert bo'lmasin)


if __name__ == "__main__":
    main()
