from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import pandas as pd

@dataclass
class TradeConfig:
    pivot: float
    r1: float
    s1: float
    sma20: float
    sma50: float
    sma200: float
    bias_score: float = 5.0
    atr_percent: float = 1.40
    std_risk_per_trade: float = 1.0
    tight_stop_buffer_pct: float = 0.20
    breakout_buffer: float = 0.0
    breakout_vol_mult: float = 1.5
    vol_lookback: int = 20
    entry_zone_pct: float = 0.25
    prefer_longs: bool = True
    min_entry_zone_cents: float = 0.25
    max_entry_zone_pts: float = 3.0

def sma_cluster(sma20: float, sma50: float, sma200: float) -> Tuple[float, float]:
    values = [sma20, sma50, sma200]
    return min(values), sum(values) / 3.0

def build_entry_zone(cfg: TradeConfig, ref_price: float) -> Tuple[float, float]:
    smin, smean = sma_cluster(cfg.sma20, cfg.sma50, cfg.sma200)
    anchor = max(cfg.s1, smean)
    width_pts = max(anchor * (cfg.entry_zone_pct / 100.0), cfg.min_entry_zone_cents / 100.0)
    width_pts = min(width_pts, cfg.max_entry_zone_pts)
    lower = anchor - width_pts
    upper = max(anchor, cfg.sma20, cfg.sma50, cfg.sma200)
    lower = min(lower, cfg.s1 + 0.15 * (upper - cfg.s1))
    if lower > upper:
        lower, upper = upper - 0.05, upper
    return round(lower, 2), round(upper, 2)

def mean_reversion_setup(cfg: TradeConfig) -> Dict[str, Any]:
    _, smean = sma_cluster(cfg.sma20, cfg.sma50, cfg.sma200)
    entry_low, entry_high = build_entry_zone(cfg, smean)
    stop = cfg.s1 - (cfg.tight_stop_buffer_pct / 100.0) * smean
    stop = round(stop, 2)
    t1 = round(cfg.pivot, 2)
    t2 = round(cfg.r1, 2)
    return {
        "name": "Mean Reversion Long",
        "entry_zone": [entry_low, entry_high],
        "stop": stop,
        "targets": [t1, t2],
        "notes": "Scale at Pivot, trail remainder toward R1 if momentum confirms."
    }

def breakout_long_setup(cfg: TradeConfig) -> Dict[str, Any]:
    return {
        "name": "Breakout Long",
        "trigger": f"1H close > {cfg.r1:.2f} with volume >= {cfg.breakout_vol_mult:.1f}x {cfg.vol_lookback}-period avg",
        "stop_rule": "Below breakout candle low or buffer",
        "fixed_stop_buffer": cfg.breakout_buffer,
        "targets": [round(cfg.r1 + 3.6, 2), round(cfg.r1 + 6.6, 2)],
        "notes": "Only take if volume confirms. Avoid chasing extended candles."
    }

def defensive_short_setup(cfg: TradeConfig) -> Dict[str, Any]:
    return {
        "name": "Defensive Countertrend Short",
        "trigger": f"Breakdown < {cfg.s1:.2f} with volume >= {cfg.breakout_vol_mult:.1f}x {cfg.vol_lookback}-period avg",
        "stop_rule": f"Above {max(cfg.sma20, cfg.sma50, cfg.sma200):.2f} or breakdown candle high",
        "targets": [round(cfg.s1 - 2.4, 2), round(cfg.s1 - 5.0, 2)],
        "notes": "Countertrend. Size down. Do not overlap with active longs."
    }

def generate_trade_script(cfg: TradeConfig) -> Dict[str, Any]:
    script = {
        "bias": "Strong uptrend" if cfg.bias_score >= 4.5 else "Neutral to bullish" if cfg.bias_score >= 3.0 else "Mixed",
        "volatility": f"Low (ATR% {cfg.atr_percent:.2f})" if cfg.atr_percent <= 2.0 else f"Medium (ATR% {cfg.atr_percent:.2f})" if cfg.atr_percent <= 3.5 else f"High (ATR% {cfg.atr_percent:.2f})",
        "key_levels": {
            "pivot": round(cfg.pivot,2),
            "r1": round(cfg.r1,2),
            "s1": round(cfg.s1,2),
            "sma20": round(cfg.sma20,2),
            "sma50": round(cfg.sma50,2),
            "sma200": round(cfg.sma200,2)
        },
        "risk": {
            "position_size_R": cfg.std_risk_per_trade,
            "guidance": "Standard size. Tight stops beyond S1 or R1."
        },
        "setups": []
    }
    script["setups"].append(mean_reversion_setup(cfg))
    script["setups"].append(breakout_long_setup(cfg))
    script["setups"].append(defensive_short_setup(cfg))
    return script

def compute_atr_percent(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last_close = c.iloc[-1]
    if pd.isna(atr) or last_close == 0:
        return float("nan")
    return round(100.0 * atr / last_close, 2)

def compute_sma_series(df: pd.DataFrame, periods=(20, 50, 200)) -> Dict[int, float]:
    res = {}
    for p in periods:
        res[p] = float(df["close"].rolling(p).mean().iloc[-1])
    return res

def compute_floor_pivots(prev_high: float, prev_low: float, prev_close: float) -> Tuple[float, float, float]:
    p = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * p - prev_low
    s1 = 2 * p - prev_high
    return round(p, 2), round(r1, 2), round(s1, 2)

def pivots_from_daily(df_daily: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    if len(df_daily) < 2:
        return None
    prev = df_daily.iloc[-2]
    return compute_floor_pivots(prev["high"], prev["low"], prev["close"])

def volume_confirmation(df: pd.DataFrame, lookback: int, mult: float) -> bool:
    if "volume" not in df.columns or len(df) < lookback + 1:
        return False
    v = df["volume"]
    avg = v.rolling(lookback).mean().iloc[-2]
    last_v = v.iloc[-1]
    if pd.isna(avg) or avg == 0:
        return False
    return bool(last_v >= mult * avg)

def detect_breakout_long(df_intraday: pd.DataFrame, r1: float, lookback: int, mult: float) -> Dict[str, Any]:
    if len(df_intraday) < lookback + 1:
        return {"signal": False, "reason": "insufficient data"}
    last_close = float(df_intraday["close"].iloc[-1])
    vol_ok = volume_confirmation(df_intraday, lookback, mult)
    cond = last_close > r1 and vol_ok
    return {
        "signal": bool(cond),
        "close": round(last_close, 2),
        "r1": round(r1, 2),
        "volume_ok": bool(vol_ok),
        "vol_mult": mult,
        "lookback": lookback
    }

def detect_breakdown_short(df_intraday: pd.DataFrame, s1: float, lookback: int, mult: float) -> Dict[str, Any]:
    if len(df_intraday) < lookback + 1:
        return {"signal": False, "reason": "insufficient data"}
    last_close = float(df_intraday["close"].iloc[-1])
    vol_ok = volume_confirmation(df_intraday, lookback, mult)
    cond = last_close < s1 and vol_ok
    return {
        "signal": bool(cond),
        "close": round(last_close, 2),
        "s1": round(s1, 2),
        "volume_ok": bool(vol_ok),
        "vol_mult": mult,
        "lookback": lookback
    }

def generate_trade_script_from_df(
    df_intraday: pd.DataFrame,
    df_daily: Optional[pd.DataFrame] = None,
    bias_score: float = 5.0,
    entry_zone_pct: float = 0.25,
    breakout_vol_mult: float = 1.5,
    vol_lookback: int = 20
) -> Dict[str, Any]:
    smas = compute_sma_series(df_intraday, periods=(20, 50, 200))
    atr_pct = compute_atr_percent(df_intraday, period=14)

    if df_daily is not None and not df_daily.empty:
        piv = pivots_from_daily(df_daily)
    else:
        piv = None

    if piv is None:
        last_close = float(df_intraday["close"].iloc[-1])
        atr_pts = last_close * atr_pct / 100.0 if atr_pct == atr_pct else 3.0
        pivot = last_close
        r1 = pivot + 0.75 * atr_pts
        s1 = pivot - 0.75 * atr_pts
    else:
        pivot, r1, s1 = piv

    cfg = TradeConfig(
        pivot=float(pivot),
        r1=float(r1),
        s1=float(s1),
        sma20=float(round(smas[20], 2)),
        sma50=float(round(smas[50], 2)),
        sma200=float(round(smas[200], 2)),
        bias_score=bias_score,
        atr_percent=float(atr_pct) if atr_pct == atr_pct else 2.0,
        entry_zone_pct=entry_zone_pct,
        breakout_vol_mult=breakout_vol_mult,
        vol_lookback=vol_lookback
    )

    script = generate_trade_script(cfg)
    script["signals"] = {
        "breakout_long": detect_breakout_long(df_intraday, cfg.r1, cfg.vol_lookback, cfg.breakout_vol_mult),
        "breakdown_short": detect_breakdown_short(df_intraday, cfg.s1, cfg.vol_lookback, cfg.breakout_vol_mult)
    }
    return script

