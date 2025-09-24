
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

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

def sma_cluster(sma20: float, sma50: float, sma200: float):
    values = [sma20, sma50, sma200]
    return min(values), sum(values) / 3.0

def build_entry_zone(cfg: TradeConfig, ref_price: float):
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

def mean_reversion_setup(cfg: TradeConfig):
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

def breakout_long_setup(cfg: TradeConfig):
    return {
        "name": "Breakout Long",
        "trigger": f"1H close > {cfg.r1:.2f} with volume >= {cfg.breakout_vol_mult:.1f}x {cfg.vol_lookback}-period avg",
        "stop_rule": "Below breakout candle low or buffer",
        "fixed_stop_buffer": cfg.breakout_buffer,
        "targets": [round(cfg.r1 + 3.6, 2), round(cfg.r1 + 6.6, 2)],
        "notes": "Only take if volume confirms. Avoid chasing extended candles."
    }

def defensive_short_setup(cfg: TradeConfig):
    return {
        "name": "Defensive Countertrend Short",
        "trigger": f"Breakdown < {cfg.s1:.2f} with volume >= {cfg.breakout_vol_mult:.1f}x {cfg.vol_lookback}-period avg",
        "stop_rule": f"Above {max(cfg.sma20, cfg.sma50, cfg.sma200):.2f} or breakdown candle high",
        "targets": [round(cfg.s1 - 2.4, 2), round(cfg.s1 - 5.0, 2)],
        "notes": "Countertrend. Size down. Do not overlap with active longs."
    }

def generate_trade_script(cfg: TradeConfig):
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
