import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

# ---------------- Page and sidebar ----------------
st.set_page_config(page_title="Trader Copilot", layout="wide")
st.sidebar.title("Trader Copilot")

# Global lookback choices used by all tabs
LOOKBACK_CHOICES = [1, 3, 7, 30, 90]
lookback_days = st.sidebar.selectbox("Lookback (trading days)", LOOKBACK_CHOICES, index=4)

# Universe: top 30 tech tickers (editable)
TOP30_TECH = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","AMD","CRM",
    "NFLX","ADBE","INTC","CSCO","QCOM","TXN","ORCL","NOW","SHOP","UBER",
    "PANW","SNOW","SMCI","PLTR","MU","ASML","TSM","ARM","ABNB","DOCU"
]
tickers_text = st.sidebar.text_area("Tickers (comma separated)", ",".join(TOP30_TECH), height=100, key="tickers_text")
UNIVERSE = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

# Keys (Polygon optional, OpenAI optional)
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# ---------------- Helpers: prices ----------------
@st.cache_data(ttl=600, show_spinner=False)
def yf_ohlc_daily(ticker: str, lookback: int) -> pd.DataFrame:
    cols = ["date","open","high","low","close","volume"]

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        d = df.reset_index()
        if "Date" in d.columns and "date" not in d.columns:
            d = d.rename(columns={"Date":"date"})
        ren = {"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"}
        d = d.rename(columns={k:v for k,v in ren.items() if k in d.columns})
        need = set(cols)
        if not need.issubset(d.columns):
            return pd.DataFrame(columns=cols)
        d = d.dropna(subset=["close"])
        d["date"] = pd.to_datetime(d["date"]).dt.date.astype(str)
        return d[cols].tail(int(lookback)).reset_index(drop=True)

    extra = max(int(lookback) + 20, 40)
    try:
        df1 = yf.download(ticker, period=f"{extra}d", interval="1d", auto_adjust=False, progress=False, threads=False)
        n1 = _norm(df1)
        if not n1.empty:
            return n1
    except Exception:
        pass
    try:
        tk = yf.Ticker(ticker)
        df2 = tk.history(period=f"{extra}d", interval="1d", auto_adjust=False)
        n2 = _norm(df2)
        if not n2.empty:
            return n2
    except Exception:
        pass
    try:
        df3 = yf.download(ticker, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        n3 = _norm(df3)
        if not n3.empty:
            return n3.tail(int(lookback))
    except Exception:
        pass
    return pd.DataFrame(columns=cols)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    p = out["close"].astype(float)

    out["sma20"] = p.rolling(20, min_periods=1).mean()
    out["sma50"] = p.rolling(50, min_periods=1).mean()
    out["sma200"] = p.rolling(200, min_periods=1).mean()

    prev_close = out["close"].shift(1)
    tr = pd.concat([out["high"] - out["low"],
                    (out["high"] - prev_close).abs(),
                    (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14, min_periods=1).mean()
    out["atr_pct"] = (out["atr14"] / out["close"]) * 100.0

    last = out.iloc[-1]
    score = 0.0
    for n, w in [(20, 1.0), (50, 2.0), (200, 2.0)]:
        v = last.get(f"sma{n}")
        if pd.notna(v) and float(last["close"]) > float(v):
            score += w
    out["trend_score"] = float(score)

    if len(out) >= 2:
        ph, pl, pc = out.loc[len(out) - 2, ["high", "low", "close"]]
        pivot = (float(ph) + float(pl) + float(pc)) / 3.0
        r1 = 2 * pivot - float(pl)
        s1 = 2 * pivot - float(ph)
    else:
        pivot = np.nan
        r1 = np.nan
        s1 = np.nan

    out["pivot"] = pivot
    out["r1"] = r1
    out["s1"] = s1
    return out

def scan_universe(tickers, lookback: int) -> pd.DataFrame:
    cols = ["ticker","score","atr_pct","close","pivot","r1","s1","sma20","sma50","sma200","chg_pct"]
    out_rows = []

    def fetch_one(t):
        try:
            base = yf_ohlc_daily(t, lookback)
            if base.empty:
                return None
            ind = add_indicators(base)
            if ind.empty:
                return None
            last = ind.iloc[-1]
            first_close = float(ind["close"].iloc[0]) if len(ind) else np.nan
            chg_pct = (float(last["close"]) / first_close - 1.0) * 100.0 if first_close else np.nan
            row = {
                "ticker": t,
                "score": float(last.get("trend_score", 0) or 0),
                "atr_pct": round(float(last.get("atr_pct", 0) or 0), 2),
                "close": round(float(last.get("close", np.nan)), 2) if pd.notna(last.get("close")) else np.nan,
                "pivot": round(float(last.get("pivot", np.nan)), 2) if pd.notna(last.get("pivot")) else np.nan,
                "r1": round(float(last.get("r1", np.nan)), 2) if pd.notna(last.get("r1")) else np.nan,
                "s1": round(float(last.get("s1", np.nan)), 2) if pd.notna(last.get("s1")) else np.nan,
                "sma20": round(float(last.get("sma20", np.nan)), 2) if pd.notna(last.get("sma20")) else np.nan,
                "sma50": round(float(last.get("sma50", np.nan)), 2) if pd.notna(last.get("sma50")) else np.nan,
                "sma200": round(float(last.get("sma200", np.nan)), 2) if pd.notna(last.get("sma200")) else np.nan,
                "chg_pct": round(float(chg_pct), 2) if pd.notna(chg_pct) else np.nan
            }
            return row
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(10, max(1, len(tickers)))) as ex:
        for r in ex.map(fetch_one, tickers):
            if r:
                out_rows.append(r)

    if not out_rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(out_rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    return df[cols].sort_values(["score", "chg_pct"], ascending=[False, False]).reset_index(drop=True)

# ---------------- Helpers: AI plan ----------------
def rule_based_plan(row: pd.Series) -> str:
    score = float(row.get("score", 0) or 0)
    atrp = float(row.get("atr_pct", 0) or 0)
    if score >= 4:
        bias = "Strong uptrend"
    elif score >= 2:
        bias = "Uptrend"
    elif score > -2:
        bias = "Sideways"
    else:
        bias = "Downtrend"

    if atrp < 2:
        vol, risk = "low volatility", "Standard size. Tight stops beyond S1 or R1"
    elif atrp < 5:
        vol, risk = "medium volatility", "Standard size. Stops beyond S1 or R1"
    else:
        vol, risk = "high volatility", "Half size. Wider stops near S2 or R2"

    txt = (
        f"**{row['ticker']} Game Plan**\n"
        f"- Bias: {bias} (score {score:.1f})\n"
        f"- Volatility: {vol} (ATR% {atrp:.2f})\n"
        f"- Key levels: Pivot {row.get('pivot', np.nan)}, R1 {row.get('r1', np.nan)}, S1 {row.get('s1', np.nan)} | "
        f"SMA20 {row.get('sma20', np.nan)}, SMA50 {row.get('sma50', np.nan)}, SMA200 {row.get('sma200', np.nan)}\n"
        f"- Plays: Mean reversion between S1 and R1; Trade range breaks with volume\n"
        f"- Risk: {risk}"
    )
    return txt

def gpt_plan(tkr: str, row: pd.Series) -> str:
    if not OPENAI_API_KEY:
        return rule_based_plan(row)
    try:
        sys = "You output plain text only. No JSON. No markdown fences."
        user = (
            f"Create a concise intraday options trading plan for {tkr} using these stats:\n"
            f"score={row.get('score')}, atr_pct={row.get('atr_pct')}, close={row.get('close')}, "
            f"pivot={row.get('pivot')}, r1={row.get('r1')}, s1={row.get('s1')}, "
            f"sma20={row.get('sma20')}, sma50={row.get('sma50')}, sma200={row.get('sma200')}.\n"
            f"Include bias, key levels, 2 play ideas and risk. Keep it 6 lines."
        )
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": [{"role": "system", "content": sys}, {"role": "user", "content": user}],
                "max_tokens": 300
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return rule_based_plan(row)

# ---------------- Helpers: Options ----------------
def next_friday(from_dt: datetime) -> datetime:
    # Friday is weekday 4
    days_ahead = (4 - from_dt.weekday()) % 7
    if days_ahead == 0 and from_dt.hour >= 13:
        return from_dt.date()
    return (from_dt + timedelta(days=days_ahead)).date()

def cnd(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def call_delta_bs(spot, strike, t_years, iv, r=0.04):
    try:
        if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
            return np.nan
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
        return cnd(d1)
    except Exception:
        return np.nan

def yf_best_calls_for_friday(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return pd.DataFrame(), None
        today = datetime.now(timezone.utc).date()
        candidate = None
        for e in exps:
            try:
                ed = datetime.strptime(e, "%Y-%m-%d").date()
                if ed >= today:
                    if candidate is None or ed < candidate:
                        candidate = ed
            except Exception:
                continue
        if candidate is None:
            candidate = datetime.strptime(exps[0], "%Y-%m-%d").date()
        opt = tk.option_chain(candidate.strftime("%Y-%m-%d"))
        calls = opt.calls.copy()
        if calls.empty:
            return pd.DataFrame(), candidate

        spot = float(tk.fast_info["last_price"]) if "last_price" in tk.fast_info else float(yf_ohlc_daily(ticker, 1)["close"].iloc[-1])
        days_to_exp = max((candidate - today).days, 0) + 0.5
        t_years = days_to_exp / 365.0

        calls["mid"] = (calls["bid"].fillna(0) + calls["ask"].fillna(0)) / 2.0
        calls["spread"] = (calls["ask"].fillna(0) - calls["bid"].fillna(0))
        calls["spread_pct"] = np.where(calls["mid"] > 0, calls["spread"] / calls["mid"] * 100.0, np.nan)

        iv = calls.get("impliedVolatility", pd.Series([np.nan] * len(calls))).astype(float).fillna(method="ffill").fillna(method="bfill")
        deltas = []
        for k, ivv in zip(calls["strike"], iv):
            deltas.append(call_delta_bs(spot, float(k), t_years, float(ivv)))
        calls["delta"] = deltas

        calls["delta_fit"] = (0.4 - (calls["delta"] - 0.4).abs()).clip(lower=-1e9)
        calls["liq"] = calls["volume"].fillna(0) + 0.5 * calls["openInterest"].fillna(0)
        calls["score"] = (calls["delta_fit"] * 1000) + (calls["liq"]) - (calls["spread_pct"].fillna(50))
        calls = calls.sort_values("score", ascending=False)
        keep_cols = ["contractSymbol","lastTradeDate","strike","bid","ask","mid","spread","spread_pct","volume","openInterest","impliedVolatility","delta","score"]
        return calls[keep_cols].head(12).reset_index(drop=True), candidate
    except Exception:
        return pd.DataFrame(), None

# ---------------- Projected Strike helpers ----------------
def nearest_strike_increment(price: float) -> float:
    if price < 5:
        return 0.5
    if price < 25:
        return 1.0
    if price < 100:
        return 1.0
    if price < 500:
        return 5.0
    return 10.0

def round_to_increment(x: float, inc: float) -> float:
    return round(x / inc) * inc

def days_until_expiry_date(exp_date) -> int:
    today = datetime.now(timezone.utc).date()
    return max((exp_date - today).days, 0)

def avg_3d_pct_change(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 4:
        return 0.0
    p = df["close"].astype(float)
    pct = p.pct_change()
    last3 = pct.tail(3).dropna()
    if last3.empty:
        return 0.0
    return float(last3.mean())

def avg_3d_dollar_change(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 4:
        return 0.0
    p = df["close"].astype(float)
    chg = p.diff()
    last3 = chg.tail(3).dropna()
    if last3.empty:
        return 0.0
    return float(last3.mean())

def projected_strikes_dual(ticker: str) -> dict:
    base = yf_ohlc_daily(ticker, lookback=10)
    if base.empty:
        return {"error": "No price data"}

    last_close = float(base["close"].iloc[-1])

    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return {"error": "No options expiries"}
        today = datetime.now(timezone.utc).date()
        chosen = None
        for e in exps:
            try:
                d = datetime.strptime(e, "%Y-%m-%d").date()
                if d >= today and (chosen is None or d < chosen):
                    chosen = d
            except Exception:
                continue
        if chosen is None:
            chosen = datetime.strptime(exps[0], "%Y-%m-%d").date()
        days_left = days_until_expiry_date(chosen)
        opt = tk.option_chain(chosen.strftime("%Y-%m-%d"))
        calls = opt.calls.copy() if hasattr(opt, "calls") else pd.DataFrame()
    except Exception:
        chosen = None
        days_left = 0
        calls = pd.DataFrame()

    a3_pct = avg_3d_pct_change(base)
    proj_pct = last_close * ((1.0 + a3_pct) ** max(days_left, 0))

    a3_usd = avg_3d_dollar_change(base)
    proj_usd = last_close + a3_usd * max(days_left, 0)

    inc = nearest_strike_increment(last_close)
    guess_pct = round_to_increment(proj_pct, inc)
    guess_usd = round_to_increment(proj_usd, inc)

    rec_pct = guess_pct
    rec_usd = guess_usd
    preview = pd.DataFrame()

    if not calls.empty and "strike" in calls.columns:
        calls = calls.copy()
        calls["dist_pct"] = (calls["strike"] - proj_pct).abs()
        calls["dist_usd"] = (calls["strike"] - proj_usd).abs()
        calls["dist_min"] = calls[["dist_pct", "dist_usd"]].min(axis=1)

        calls_sorted_pct = calls.sort_values("dist_pct")
        calls_sorted_usd = calls.sort_values("dist_usd")
        rec_pct = float(calls_sorted_pct.iloc[0]["strike"])
        rec_usd = float(calls_sorted_usd.iloc[0]["strike"])

        prev = calls.sort_values("dist_min").head(12).copy()
        prev["mid"] = (prev["bid"].fillna(0) + prev["ask"].fillna(0)) / 2.0
        preview = prev[[
            "contractSymbol","strike","bid","ask","mid","volume","openInterest","dist_pct","dist_usd","dist_min"
        ]].reset_index(drop=True)

    return {
        "close": last_close,
        "days_left": days_left,
        "expiry": chosen,
        "avg3_pct": a3_pct * 100.0,
        "avg3_usd": a3_usd,
        "projection_pct": proj_pct,
        "projection_usd": proj_usd,
        "rec_strike_pct": rec_pct,
        "rec_strike_usd": rec_usd,
        "chain_df": preview
    }

# ---------------- UI Tabs ----------------
page = st.sidebar.radio(
    "Go to",
    ["Scanner", "AI Game Plan", "Options (Best Calls)", "Projected Strike", "About"],
    index=0,
    key="nav"
)

# Scanner
if page == "Scanner":
    st.header("Scanner - Top trends")
    if not UNIVERSE:
        st.warning("Enter at least one ticker.")
    else:
        df = scan_universe(UNIVERSE, lookback_days)
        if df.empty:
            st.warning("No data returned.")
        else:
            st.caption("Sorted by trend score, then percent change over the selected lookback")
            st.dataframe(df, use_container_width=True)

# AI Game Plan
elif page == "AI Game Plan":
    st.header("AI Game Plan")
    if not UNIVERSE:
        st.warning("Enter at least one ticker.")
    else:
        base = scan_universe(UNIVERSE, lookback_days)
        if base.empty:
            st.warning("No data.")
        else:
            topn = min(10, len(base))
            st.caption(f"Showing top {topn} by trend score for lookback {lookback_days}d")
            for _, row in base.head(topn).iterrows():
                tkr = row["ticker"]
                txt = gpt_plan(tkr, row) if OPENAI_API_KEY else rule_based_plan(row)
                st.markdown(txt)

# Options
elif page == "Options (Best Calls)":
    st.header("Best Calls expiring Friday")
    tkr = st.text_input("Ticker", UNIVERSE[0] if UNIVERSE else "SPY").strip().upper()
    if not tkr:
        st.warning("Enter a ticker.")
    else:
        df_calls = pd.DataFrame()
        expiry_date = None
        try:
            # Polygon attempt could go here if you want to wire it later
            raise Exception("Use Yahoo fallback")
        except Exception:
            df_calls, expiry_date = yf_best_calls_for_friday(tkr)

        if df_calls is None or df_calls.empty:
            st.warning("No options data found.")
        else:
            if expiry_date:
                st.caption(f"Expiry used: {expiry_date}")
            st.dataframe(df_calls, use_container_width=True)

            picks = df_calls.head(3)
            if not picks.empty:
                st.subheader("Top 3 setups")
                for _, r in picks.iterrows():
                    st.write(
                        f"{r['contractSymbol']} | strike {r['strike']} | "
                        f"mid {round(r['mid'],2)} | delta {round(r['delta'],2) if pd.notna(r['delta']) else 'n/a'} | "
                        f"vol {int(r['volume']) if pd.notna(r['volume']) else 0} | "
                        f"oi {int(r['openInterest']) if pd.notna(r['openInterest']) else 0} | "
                        f"spread% {round(r['spread_pct'],1) if pd.notna(r['spread_pct']) else 'n/a'}"
                    )

# Projected Strike
elif page == "Projected Strike":
    st.header("Projected Strike at expiration - dual model (3-day averages)")
    tkr = st.text_input("Ticker", UNIVERSE[0] if UNIVERSE else "SPY").strip().upper()
    if not tkr:
        st.warning("Enter a ticker.")
    else:
        res = projected_strikes_dual(tkr)
        if "error" in res:
            st.error(res["error"])
        else:
            if res["expiry"]:
                st.caption(f"Expiry used: {res['expiry']}  |  Days to expiry: {res['days_left']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last Close", f"{res['close']:.2f}")
            c2.metric("Avg 3-day daily %", f"{res['avg3_pct']:.2f}%")
            c3.metric("Avg 3-day $/day", f"{res['avg3_usd']:.2f}")
            c4.metric("Days to expiry", f"{res['days_left']}")

            st.subheader("Projections")
            p1, p2 = st.columns(2)
            p1.metric("Percent model projection", f"{res['projection_pct']:.2f}")
            p2.metric("Dollar model projection", f"{res['projection_usd']:.2f}")

            st.subheader("Recommended call strikes")
            s1, s2 = st.columns(2)
            s1.write(f"Nearest to percent model: **{res['rec_strike_pct']:.2f}**")
            s2.write(f"Nearest to dollar model: **{res['rec_strike_usd']:.2f}**")

            dfp = res.get("chain_df")
            if isinstance(dfp, pd.DataFrame) and not dfp.empty:
                st.caption("Nearest available strikes by distance to either projection")
                st.dataframe(dfp, use_container_width=True)
            else:
                st.info("No chain preview available. Strike suggestions above are still valid.")

# About
else:
    st.header("About")
    st.write("Scanner, AI plan, best weekly calls, and projected strike. Data via Yahoo Finance. OpenAI key optional for AI plans. Polygon key can be added later for options.")

