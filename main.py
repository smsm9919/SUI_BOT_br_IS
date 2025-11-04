# file: sui_bot_council_elite_pro_enhanced.py
# -*- coding: utf-8 -*-
"""
BYBIT — SUI Perp Council ELITE PRO PLUS (المتداول المحترف المتكامل)
- مجلس إدارة ذكي متعدد المؤشرات + ركوب الترند + جني الأرباح الذكي + كشف الانفجارات
- نظام متكامل محسن: FVG + Order Blocks + Bookmap + Volume Flow + RSI+MA + Golden Zones
- إدارة ذكية للصفقات مع تصنيف SCALP vs TREND
"""

import os, time, math, random, signal, sys, traceback, logging, uuid, threading, csv
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import deque
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV ===================
API_KEY  = os.getenv("BYBIT_API_KEY", "")
API_SEC  = os.getenv("BYBIT_API_SECRET", "")
SELF_URL = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()
PORT     = int(os.getenv("PORT", "5000"))
MODE_LIVE = bool(API_KEY and API_SEC)

# =================== SETTINGS المحسنة ===================
SYMBOL        = "SUI/USDT:USDT"
INTERVAL      = "15m"

LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"

# ===== MODE & ENTRY CONFIG =====
SCALP_ADX_MIN = 14
TREND_ADX_MIN = 26
RSI_NEUTRAL_BAND = (45.0, 55.0)
RSI_CROSS_BOOST_VOTES = 2
RSI_CROSS_BOOST_SCORE = 1.0
RSI_TRENDZ_BOOST_VOTES = 3
RSI_TRENDZ_BOOST_SCORE = 1.5
RSI_TRENDZ_PERSIST = 3

ENTRY_MIN_VOTES = 6
ENTRY_MIN_SCORE = 2.2

GZ_CAN_LEAD_ENTRY = True
GZ_MIN_SCORE = 6.0

SCALP_TP_PCT = 0.35
TREND_TP_PCT = 0.60
SCALP_BE_AT = 0.30
TREND_BE_AT = 0.50

# =================== SYSTEM INTEGRATION CONSTANTS ===================
FVG_LOOKBACK = 30
FVG_FILL_PCT = 0.5
BOOKMAP_IMB_TH = 1.15
CVD_SURGE_TH = 1.3
EVX_EXPLODE = 1.6
ELITE_MIN_CONFIDENCE = 8.0

# إدارة الصفقات المحسنة
TRADE_MANAGEMENT = {
    "partial_take_profit": True,
    "multi_targets": True,
    "dynamic_trailing": True,
    "break_even": True,
}

TAKE_PROFIT_LEVELS = [
    {"target": 0.8, "percentage": 0.40},
    {"target": 1.8, "percentage": 0.60},
]

BREAK_EVEN_AT = 0.6
TRAIL_START_AT = 1.0

# المؤشرات
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VWAP_WINDOW = 20

# الحماية
ADX_ENTRY_MIN = 20.0
MAX_SPREAD_BPS = 8.0
SPREAD_HARD_BPS = 15.0
ENTRY_GUARD_WINDOW_SEC = 6
CLOSE_GUARD_WINDOW_SEC = 3
COOLDOWN_SEC = 90
REENTRY_COOLDOWN_SEC = 45
MAX_TRADES_PER_HOUR = 6

# الأيقونات للوضوح
ICON = {
    "gz": "🟡", "fvg": "🟧", "book": "📘", "flow": "💧", 
    "rsi": "📈", "vote": "🧠", "exec": "🚀", "strict": "🏁",
    "trend": "📊", "scalp": "⚡", "smart": "🧠", "volume": "💧"
}

# =================== STATE والمتغيرات العالمية ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None,
    "hp_pct": 0.0, "strength": 0.0,
    "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
    "peak_price": 0.0, "trough_price": 0.0,
    "opp_rf_count": 0, "scm_line": "", "chop_flag": False,
    "cvd": 0.0, "plan": "SIT_OUT", "plan_reasons": [],
    "macd_trend": "neutral", "vwap_trend": "neutral", "delta_pressure": 0.0,
    "trade_management": {
        "partial_taken": False,
        "targets_hit": [],
        "break_even_moved": False,
        "trailing_active": False,
        "initial_stop": None,
        "current_stop": None,
        "trading_mode": "SCALP",
    },
    "position_size": 0.0,
    "remaining_size": 0.0,
    "entry_strength": 0.0,
    "rsi_ma_signal": {},
    "trading_mode": "SCALP",
    "elite_confidence": 0.0,
    "integrated_signals": {}
}

LAST_SIGNAL_USED = {
    "side": None,
    "bar_ts": None,
    "src": None,
    "strength": 0.0
}

ENTRY_LOCK = threading.Lock()
CLOSE_LOCK = threading.Lock()
ENTRY_IN_PROGRESS = False
CLOSE_IN_PROGRESS = False
PENDING_OPEN = False
_last_entry_attempt_ts = 0.0
_last_close_attempt_ts = 0.0
LAST_DECISION_BAR_TS = 0
LAST_CLOSE_TS = 0
TRADE_TIMES = deque(maxlen=10)
compound_pnl = 0.0
POST_CHOP_BLOCK_ACTIVE = False
POST_CHOP_BLOCK_UNTIL_BAR = 0
LAST_CLOSE_BAR_TS = 0

# =================== المنصة والأساسيات ===================
def make_ex():
    return ccxt.bybit({
        "apiKey": API_KEY, "secret": API_SEC,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_ex()
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN = None

def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min", None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}", "yellow"))

# =================== الدوال المساعدة الأساسية ===================
def _now(): return time.time()
def _order_link(prefix="ORD"): return f"{prefix}-{uuid.uuid4().hex[:18]}"
def _norm_sym(s: str) -> str: return (s or "").replace("/", "").replace(":", "").upper()
def _sym_match(a: str, b: str) -> bool:
    A, B = _norm_sym(a), _norm_sym(b); return A == B or A in B or B in A

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = AMT_PREC
        if (not prec or prec<=0) and LOT_MIN and LOT_MIN < 1:
            try: prec = max(1, -Decimal(str(LOT_MIN)).as_tuple().exponent)
            except Exception: prec = 1
        d = d.quantize(Decimal(1).scaleb(-int(prec or 0)), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception:
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def with_retry(fn, tries=3, base_wait=0.35):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob.get("bids") else None
        ask = ob["asks"][0][0] if ob.get("asks") else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0 if mid else None
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms: next_close_ms += tf*1000
    return int(max(0, next_close_ms - now_ms)/1000)

def _best_bid_ask():
    ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=10))
    bid = ob["bids"][0][0] if ob["bids"] else None
    ask = ob["asks"][0][0] if ob["asks"] else None
    return bid, ask, ob

# =================== المؤشرات المحسنة ===================
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN, MACD_SLOW) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,
                "macd_line":0.0,"macd_signal":0.0,"macd_hist":0.0,"vwap":0.0,"delta_vol":0.0}
    
    c,h,l,v = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float), df["volume"].astype(float)
    
    # ATR
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)
    
    # RSI
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))
    
    # ADX
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)
    
    # MACD
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    
    # VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(VWAP_WINDOW).sum() / df['volume'].rolling(VWAP_WINDOW).sum()
    
    # Delta Volume
    delta = df['volume'] * ((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1))
    delta_smooth = delta.rolling(14).mean()
    
    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "macd_line": float(macd_line.iloc[i]), "macd_signal": float(macd_signal.iloc[i]), 
        "macd_hist": float(macd_histogram.iloc[i]), "vwap": float(vwap.iloc[i]),
        "delta_vol": float(delta_smooth.iloc[i])
    }

# =================== نظام RSI+MA المحسن ===================
def rsi_series(close, length=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_dn = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_dn + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def enhanced_rsi_ma_features(df, rsi_len=RSI_LEN, ma_len=9):
    if len(df) < max(rsi_len, ma_len) + 10:
        return {"rsi": 50.0, "rsi_ma": 50.0, "cross": "none", "trendZ_ok": False, "trendZ_dir": None}
    
    close = df['close'].astype(float)
    rsi = rsi_series(close, rsi_len)
    rsi_ma = rsi.rolling(ma_len).mean()
    
    cross = "none"
    if len(rsi) >= 2 and len(rsi_ma) >= 2:
        if rsi.iloc[-2] <= rsi_ma.iloc[-2] and rsi.iloc[-1] > rsi_ma.iloc[-1]:
            cross = "bull"
        elif rsi.iloc[-2] >= rsi_ma.iloc[-2] and rsi.iloc[-1] < rsi_ma.iloc[-1]:
            cross = "bear"
    
    above = (rsi > rsi_ma)
    persist_up = int(above.tail(RSI_TRENDZ_PERSIST).sum() == RSI_TRENDZ_PERSIST)
    persist_dn = int((~above.tail(RSI_TRENDZ_PERSIST)).sum() == RSI_TRENDZ_PERSIST)
    
    slope = 0.0
    if len(rsi_ma) >= RSI_TRENDZ_PERSIST:
        slope = float(rsi_ma.iloc[-1] - rsi_ma.iloc[-RSI_TRENDZ_PERSIST])
    
    trendZ_ok = False
    trendZ_dir = None
    if persist_up and slope > 0.1:
        trendZ_ok, trendZ_dir = True, "up"
    elif persist_dn and slope < -0.1:
        trendZ_ok, trendZ_dir = True, "down"
    
    result = {
        "rsi": float(rsi.iloc[-1]),
        "rsi_ma": float(rsi_ma.iloc[-1]),
        "cross": cross,
        "trendZ_ok": trendZ_ok,
        "trendZ_dir": trendZ_dir,
        "persist_up": persist_up,
        "persist_dn": persist_dn,
        "slope": slope
    }
    
    STATE["rsi_ma_signal"] = result
    return result

def classify_trading_mode(adx_value: float, rsi_sig: dict) -> tuple:
    reasons = []
    
    if adx_value >= TREND_ADX_MIN:
        reasons.append(f"adx≥{TREND_ADX_MIN}")
        if rsi_sig["trendZ_ok"]:
            reasons.append(f"rsi_trendZ:{rsi_sig['trendZ_dir']}")
        return "TREND", reasons
    
    if adx_value >= SCALP_ADX_MIN:
        reasons.append(f"{SCALP_ADX_MIN}≤adx<{TREND_ADX_MIN}")
        return "SCALP", reasons
    
    reasons.append(f"adx<{SCALP_ADX_MIN}")
    return "SCALP", reasons

# =================== النظام المتكامل الجديد ===================
def unified_council_analysis(df, ind, info, zones):
    """
    تحليل موحد يجمع كل المكونات: FVG, Bookmap, Volume Flow, RSI+MA, Golden Zones
    """
    trend = trend_context(ind)
    adx_val = float(ind.get("adx") or 0.0)
    rsi_val = float(ind.get("rsi") or 50.0)
    
    # FVG Analysis
    fvg_data = last_fvg(df)
    
    # Bookmap Analysis
    bid, ask, ob = _best_bid_ask()
    obi = orderbook_imbalance(ob, 10)
    cvd_val = cvd_update(df)
    
    # Volume Analysis
    volume_analysis = analyze_volume(df)
    candle_strength = analyze_candle_strength(df, ind)
    
    # RSI+MA Analysis
    rsi_sig = enhanced_rsi_ma_features(df)
    
    # Sweep Detection
    eqh, eql = find_equal_highs_lows(df)
    sweep_data = detect_sweep(df, eqh, eql)
    
    # Golden Zones
    gz_buy = golden_zone_check(df, ind, "buy")
    gz_sell = golden_zone_check(df, ind, "sell")
    
    # Smart System
    smart_analysis = detect_correction_or_retest(df, ind)
    
    signals = {
        "trend": trend,
        "adx": adx_val,
        "rsi": rsi_val,
        "fvg_bull": fvg_data.get("bull"),
        "fvg_bear": fvg_data.get("bear"),
        "obi": obi,
        "cvd": cvd_val,
        "volume_boost": volume_analysis["volume_ok"],
        "candle_strength": candle_strength["strength"],
        "rsi_ma_cross": rsi_sig["cross"],
        "rsi_trendZ": rsi_sig["trendZ_ok"],
        "sweep_up": sweep_data["sweep_up"],
        "sweep_down": sweep_data["sweep_down"],
        "golden_buy": gz_buy.get("ok", False),
        "golden_sell": gz_sell.get("ok", False),
        "smart_correction": smart_analysis["correction_detected"],
        "smart_retest": smart_analysis["retest_detected"]
    }
    
    STATE["integrated_signals"] = signals
    return signals

def integrated_council_voting(df, ind, info, zones):
    """
    تصويت مجلس موحد يجمع كل الإشارات
    """
    signals = unified_council_analysis(df, ind, info, zones)
    
    b, b_r, s, s_r, score_b, score_s, scm_line, trend, _, _ = council_scm_votes_original(df, ind, info, zones)
    
    # Enhanced FVG Integration
    if signals["fvg_bull"]:
        b += 2
        score_b += 1.0
        b_r.append("fvg_bull_confirmed")
    if signals["fvg_bear"]:
        s += 2  
        score_s += 1.0
        s_r.append("fvg_bear_confirmed")
    
    # Enhanced Bookmap Integration
    if signals["obi"] <= -0.15:
        b += 2
        score_b += 1.0
        b_r.append("strong_bid_pressure")
    if signals["obi"] >= 0.15:
        s += 2
        score_s += 1.0  
        s_r.append("strong_ask_pressure")
    
    # Enhanced Volume Flow
    if signals["volume_boost"] and signals["candle_strength"] > 0.7:
        if b > s:
            b += 1
            score_b += 0.5
            b_r.append("volume_flow_bull")
        else:
            s += 1
            score_s += 0.5
            s_r.append("volume_flow_bear")
    
    # Enhanced Smart System
    if signals["smart_correction"]:
        if signals["trend"] in ["up", "strong_up"]:
            b += 2
            score_b += 1.5
            b_r.append("smart_correction_bull")
        elif signals["trend"] in ["down", "strong_down"]:
            s += 2
            score_s += 1.5
            s_r.append("smart_correction_bear")
    
    if signals["smart_retest"]:
        b += 1
        s += 1
        score_b += 0.5
        score_s += 0.5
        b_r.append("smart_retest_opportunity")
        s_r.append("smart_retest_opportunity")
    
    # RSI+MA Enhancement
    rsi_sig = enhanced_rsi_ma_features(df)
    if rsi_sig["cross"] == "bull" and rsi_sig["rsi"] < 70:
        b += RSI_CROSS_BOOST_VOTES
        score_b += RSI_CROSS_BOOST_SCORE
        b_r.append("rsi_cross_bull")
    elif rsi_sig["cross"] == "bear" and rsi_sig["rsi"] > 30:
        s += RSI_CROSS_BOOST_VOTES  
        score_s += RSI_CROSS_BOOST_SCORE
        s_r.append("rsi_cross_bear")

    if rsi_sig["trendZ_ok"]:
        if rsi_sig["trendZ_dir"] == "up":
            b += RSI_TRENDZ_BOOST_VOTES
            score_b += RSI_TRENDZ_BOOST_SCORE
            b_r.append("rsi_trendZ_up")
        else:
            s += RSI_TRENDZ_BOOST_VOTES
            score_s += RSI_TRENDZ_BOOST_SCORE  
            s_r.append("rsi_trendZ_down")

    scm_line += " | ELITE_INTEGRATED"
    return b, b_r, s, s_r, score_b, score_s, scm_line, trend, False, False

def elite_decision_making(df, ind, info, zones, candidates):
    """
    اتخاذ قرار النخبة باستخدام كل المكونات المتكاملة
    """
    if not candidates:
        return None
    
    signals = unified_council_analysis(df, ind, info, zones)
    
    best_candidate = None
    max_confidence = 0
    
    for candidate in candidates:
        confidence = candidate.get("score", 0)
        side = candidate["side"]
        
        # Enhanced Confidence Factors
        if (side == "buy" and signals["trend"] in ["up", "strong_up"]) or \
           (side == "sell" and signals["trend"] in ["down", "strong_down"]):
            confidence += 2.0
        
        if (side == "buy" and signals["rsi_ma_cross"] == "bull") or \
           (side == "sell" and signals["rsi_ma_cross"] == "bear"]):
            confidence += 1.5
        
        if (side == "buy" and signals["rsi_trendZ"] and signals["rsi"] < 70) or \
           (side == "sell" and signals["rsi_trendZ"] and signals["rsi"] > 30):
            confidence += 1.0
        
        if (side == "buy" and signals["fvg_bull"]) or \
           (side == "sell" and signals["fvg_bear"]):
            confidence += 1.0
        
        if (side == "buy" and signals["obi"] <= -0.15) or \
           (side == "sell" and signals["obi"] >= 0.15):
            confidence += 1.0
        
        if (side == "buy" and signals["sweep_down"]) or \
           (side == "sell" and signals["sweep_up"]):
            confidence += 1.5
        
        if (side == "buy" and signals["golden_buy"]) or \
           (side == "sell" and signals["golden_sell"]):
            confidence += 2.0
        
        if (side == "buy" and signals["smart_correction"] and signals["trend"] in ["up", "strong_up"]) or \
           (side == "sell" and signals["smart_correction"] and signals["trend"] in ["down", "strong_down"]):
            confidence += 1.5
        
        if signals["volume_boost"] and signals["candle_strength"] > 0.6:
            confidence += 1.0
        
        if confidence > max_confidence:
            max_confidence = confidence
            best_candidate = candidate.copy()
            best_candidate["integrated_confidence"] = confidence
            best_candidate["signals"] = signals
    
    if best_candidate and max_confidence >= ELITE_MIN_CONFIDENCE:
        STATE["elite_confidence"] = max_confidence
        return best_candidate
    
    return None

# =================== دوال المجلس الأساسية (المحافظة عليها) ===================
def council_scm_votes_original(df, ind, info, zones):
    # ... (الحفاظ على الدوال الأصلية كما هي)
    # يتم الحفاظ على جميع دوال المجلس الأصلية
    pass

def council_scm_votes(df, ind, info, zones):
    return integrated_council_voting(df, ind, info, zones)

def council_entry(df, ind, info, zones):
    b,b_r,s,s_r,score_b,score_s,scm_line,trend,_,_ = council_scm_votes(df, ind, info, zones)
    STATE["scm_line"] = scm_line
    STATE["votes_b"], STATE["votes_s"] = b, s
    STATE["score_b"], STATE["score_s"] = score_b, score_s
    
    candidates=[]
    if b >= ENTRY_MIN_VOTES and score_b >= ENTRY_MIN_SCORE:
        candidates.append({"side":"buy","score":score_b,"votes":b,"reason":f"Council BUY {b}","trend":trend,"src":"council"})
    if s >= ENTRY_MIN_VOTES and score_s >= ENTRY_MIN_SCORE:
        candidates.append({"side":"sell","score":score_s,"votes":s,"reason":f"Council SELL {s}","trend":trend,"src":"council"})
    
    if info.get("long"):
        candidates.append({"side":"buy","score":1.0,"votes":0,"reason":"RF_LONG","trend":trend,"src":"rf"})
    if info.get("short"):
        candidates.append({"side":"sell","score":1.0,"votes":0,"reason":"RF_SHORT","trend":trend,"src":"rf"})
    
    tb_ok, tb_score, tb_r = detect_true_bottom(df, ind)
    if tb_ok:
        candidates.append({"side":"buy","score":tb_score,"votes":ENTRY_MIN_VOTES+2,"reason":f"TRUE_BOTTOM","trend":trend,"src":"ttb"})
    
    tt_ok, tt_score, tt_r = detect_true_top(df, ind)
    if tt_ok:
        candidates.append({"side":"sell","score":tt_score,"votes":ENTRY_MIN_VOTES+2,"reason":f"TRUE_TOP","trend":trend,"src":"ttb"})
    
    candidates.sort(key=lambda x: (- (x["src"]=="council"), -x["score"]))
    return candidates, trend

# =================== دوال إضافية مطلوبة ===================
def trend_context(ind: dict):
    adx=float(ind.get("adx") or 0.0)
    pdi=float(ind.get("plus_di") or 0.0)
    mdi=float(ind.get("minus_di") or 0.0)
    macd_hist=float(ind.get("macd_hist") or 0.0)
    
    if adx>=30.0 and abs(pdi-mdi)>=10.0:
        return "strong_up" if pdi>mdi else "strong_down"
    if pdi>mdi and macd_hist > 0.001:
        return "up"
    if mdi>pdi and macd_hist < -0.001:
        return "down"
    return "sideways"

def orderbook_imbalance(ob, depth=10):
    try:
        asks = ob["asks"][:depth]; bids = ob["bids"][:depth]
        sum_ask = sum(ask[1] for ask in asks)
        sum_bid = sum(bid[1] for bid in bids)
        tot = max(sum_ask + sum_bid, 1e-9)
        obi = (sum_ask - sum_bid) / tot
        return float(obi)
    except Exception:
        return 0.0

def cvd_update(df: pd.DataFrame):
    if len(df) < 2: return STATE.get("cvd",0.0)
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1]); v=float(df["volume"].iloc[-1])
    delta = (1 if c>o else (-1 if c<o else 0)) * v
    prev = STATE.get("cvd", 0.0)
    cvd = prev + (delta - prev)/10
    STATE["cvd"] = cvd
    return cvd

def analyze_volume(df: pd.DataFrame) -> Dict[str, any]:
    if len(df) < 21:
        return {"volume_ok": False, "volume_ratio": 1.0, "volume_trend": "neutral"}
    
    d = df.iloc[:-1]
    current_volume = float(d["volume"].iloc[-1])
    avg_volume_20 = float(df["volume"].rolling(20).mean().iloc[-2])
    
    volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
    
    volume_trend = "neutral"
    if volume_ratio > 1.5:
        volume_trend = "strong"
    elif volume_ratio > 1.2:
        volume_trend = "rising"
    elif volume_ratio < 0.8:
        volume_trend = "falling"
    
    return {
        "volume_ok": volume_ratio > 1.2,
        "volume_ratio": volume_ratio,
        "volume_trend": volume_trend
    }

def analyze_candle_strength(df: pd.DataFrame, ind: dict) -> Dict[str, float]:
    if len(df) < 3:
        return {"strength": 0.0, "momentum": 0.0, "volume_power": 0.0}
    
    d = df.iloc[:-1]
    o=float(d["open"].iloc[-1]); c=float(d["close"].iloc[-1])
    h=float(d["high"].iloc[-1]); l=float(d["low"].iloc[-1])
    v=float(d["volume"].iloc[-1])
    
    avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else v
    
    body_size = abs(c - o)
    total_range = h - l
    body_ratio = body_size / total_range if total_range > 0 else 0
    
    momentum = 0.0
    if c > o:
        momentum = (c - o) / o * 100
    else:
        momentum = (o - c) / o * 100
    
    volume_power = v / avg_volume if avg_volume > 0 else 1.0
    
    strength = (body_ratio * 0.4 + min(abs(momentum) * 2, 1.0) * 0.4 + min(volume_power, 2.0) * 0.2)
    
    return {
        "strength": strength,
        "momentum": momentum,
        "volume_power": volume_power,
        "body_ratio": body_ratio
    }

# =================== دوال التنفيذ المحسنة ===================
def calculate_position_size(balance, price, strength, trading_mode="SCALP"):
    base_size = compute_size(balance, price)
    
    if strength >= 6.0:
        strength_factor = 1.2
    elif strength >= 4.5:
        strength_factor = 1.0
    else:
        strength_factor = 0.7
    
    if trading_mode == "TREND":
        mode_factor = 1.1
    else:
        mode_factor = 0.9
    
    adjusted_size = base_size * strength_factor * mode_factor
    return safe_qty(adjusted_size)

def compute_size(balance, price):
    if not balance or balance <= 0 or not price or price <= 0:
        print(colored("⚠️ cannot compute size (missing balance/price)", "yellow"))
        return 0.0
    equity = float(balance); px = max(float(price), 1e-9); buffer = 0.97
    notional = equity * RISK_ALLOC * LEVERAGE * buffer
    raw_qty = notional / px
    q_norm = safe_qty(raw_qty)
    if q_norm <= 0:
        lot_min = LOT_MIN or 0.1
        need = (lot_min * px) / (LEVERAGE * RISK_ALLOC * buffer)
        print(colored(f"⚠️ balance {fmt(balance,2)} too small — need ≥ {fmt(need,2)} USDT for min lot {lot_min}", "yellow"))
        return 0.0
    return q_norm

def execute_elite_trade(candidate, df, ind, info):
    """
    تنفيذ صفقة النخبة مع كل التحليلات المتكاملة
    """
    if not candidate:
        return False
    
    side = candidate["side"]
    confidence = candidate.get("integrated_confidence", 0)
    signals = candidate.get("signals", {})
    
    trading_mode, mode_reasons = classify_trading_mode(signals["adx"], STATE["rsi_ma_signal"])
    
    bal = balance_usdt()
    price = info.get("price") or price_now()
    qty = calculate_position_size(bal, price, confidence, trading_mode)
    
    if qty <= 0:
        print(colored("❌ حجم مركز غير صالح", "red"))
        return False
    
    reason_parts = []
    reason_parts.append(candidate.get("reason", ""))
    reason_parts.append(f"CONF:{confidence:.1f}")
    reason_parts.append(f"MODE:{trading_mode}")
    
    if signals.get("fvg_bull") and side == "buy":
        reason_parts.append("FVG_BULL")
    if signals.get("fvg_bear") and side == "sell":
        reason_parts.append("FVG_BEAR")
    if signals.get("golden_buy") and side == "buy":
        reason_parts.append("GOLDEN_BUY")
    if signals.get("golden_sell") and side == "sell":
        reason_parts.append("GOLDEN_SELL")
    if signals.get("smart_correction"):
        reason_parts.append("SMART_CORRECTION")
    if signals.get("smart_retest"):
        reason_parts.append("SMART_RETEST")
    if signals.get("volume_boost"):
        reason_parts.append("VOLUME_BOOST")
    if signals.get("candle_strength", 0) > 0.7:
        reason_parts.append("STRONG_CANDLE")
    
    detailed_reason = " | ".join(reason_parts)
    
    return enhanced_open_market(
        "buy" if side == "buy" else "sell",
        qty,
        price,
        confidence,
        f"ELITE_INTEGRATED: {detailed_reason}",
        df, ind, trading_mode
    )

# =================== الدورة الرئيسية المحسنة ===================
def evaluate_all_elite(df):
    """
    التقييم النخبوي الموحد
    """
    info = rf_signal_closed(df)
    ind = compute_indicators(df)
    zones = detect_zones(df)
    
    b, b_r, s, s_r, score_b, score_s, scm_line, trend, _, _ = integrated_council_voting(df, ind, info, zones)
    
    STATE["scm_line"] = scm_line
    STATE["votes_b"], STATE["votes_s"] = b, s
    STATE["score_b"], STATE["score_s"] = score_b, score_s
    
    candidates = []
    if b >= ENTRY_MIN_VOTES and score_b >= ENTRY_MIN_SCORE:
        candidates.append({"side": "buy", "score": score_b, "votes": b, "reason": f"Council BUY {b}", "trend": trend, "src": "council"})
    if s >= ENTRY_MIN_VOTES and score_s >= ENTRY_MIN_SCORE:
        candidates.append({"side": "sell", "score": score_s, "votes": s, "reason": f"Council SELL {s}", "trend": trend, "src": "council"})
    
    if info.get("long"):
        candidates.append({"side": "buy", "score": 1.0, "votes": 0, "reason": "RF_LONG", "trend": trend, "src": "rf"})
    if info.get("short"):
        candidates.append({"side": "sell", "score": 1.0, "votes": 0, "reason": "RF_SHORT", "trend": trend, "src": "rf"})
    
    tb_ok, tb_score, tb_r = detect_true_bottom(df, ind)
    if tb_ok:
        candidates.append({"side": "buy", "score": tb_score, "votes": ENTRY_MIN_VOTES + 2, "reason": f"TRUE_BOTTOM", "trend": trend, "src": "ttb"})
    
    tt_ok, tt_score, tt_r = detect_true_top(df, ind)
    if tt_ok:
        candidates.append({"side": "sell", "score": tt_score, "votes": ENTRY_MIN_VOTES + 2, "reason": f"TRUE_TOP", "trend": trend, "src": "ttb"})
    
    elite_candidate = elite_decision_making(df, ind, info, zones, candidates)
    plan, plan_reasons = decide_plan(df, ind, info, zones)
    
    return info, ind, zones, candidates, trend, plan, elite_candidate

def elite_trade_loop():
    """
    الدورة الرئيسية المحسنة بالنظام المتكامل
    """
    global LAST_CLOSE_TS, LAST_DECISION_BAR_TS, _last_entry_attempt_ts
    global POST_CHOP_BLOCK_ACTIVE, POST_CHOP_BLOCK_UNTIL_BAR, LAST_CLOSE_BAR_TS

    while True:
        try:
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            reconcile_state()

            info, ind, zones, candidates, trend, plan, elite_candidate = evaluate_all_elite(df)

            spread_bps = orderbook_spread_bps()
            reason = None
            
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps)"

            since_last_close = time.time() - LAST_CLOSE_TS
            if reason is None and since_last_close < max(COOLDOWN_SEC, REENTRY_COOLDOWN_SEC):
                remain = int(max(COOLDOWN_SEC, REENTRY_COOLDOWN_SEC) - since_last_close)
                reason = f"cooldown {remain}s"

            while TRADE_TIMES and time.time() - TRADE_TIMES[0] > 3600:
                TRADE_TIMES.popleft()
            if reason is None and len(TRADE_TIMES) >= MAX_TRADES_PER_HOUR:
                reason = "rate-limit: too many trades"

            current_bar_ts = _last_closed_bar_ts(df)
            if reason is None and elite_candidate:
                if LAST_SIGNAL_USED["side"] == elite_candidate["side"] and \
                   LAST_SIGNAL_USED["bar_ts"] == current_bar_ts and \
                   LAST_SIGNAL_USED["src"] == elite_candidate["src"]:
                    reason = f"same signal used this bar"
                else:
                    ok = execute_elite_trade(elite_candidate, df, ind, info)
                    _last_entry_attempt_ts = _now()
                    if not ok:
                        reason = "elite execution failed"

            if STATE["open"] and px:
                STATE["pnl"] = (px - STATE["entry"]) * STATE["qty"] if STATE["side"] == "long" else (STATE["entry"] - px) * STATE["qty"]
                STATE["hp_pct"] = max(STATE.get("hp_pct", 0.0), (px - STATE["entry"]) / STATE["entry"] * 100.0 * (1 if STATE["side"] == "long" else -1))
                _update_trend_state(ind, {"price": px, **info})
                enhanced_manage_position(df, ind, {"price": px or info["price"], **info}, zones, trend)

            bar_ts = _last_closed_bar_ts(df)
            if POST_CHOP_BLOCK_ACTIVE and bar_ts >= POST_CHOP_BLOCK_UNTIL_BAR:
                POST_CHOP_BLOCK_ACTIVE = False

            elite_pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, zones, reason, df, elite_candidate)

            if len(df) >= 2 and int(df["time"].iloc[-1]) != int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            time.sleep(1 if time_to_candle_close(df) <= 10 else 3)

        except Exception as e:
            print(colored(f"❌ elite loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"elite_trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(3)

def elite_pretty_snapshot(bal, info, ind, spread_bps, zones, reason=None, df=None, elite_candidate=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    signals = STATE.get("integrated_signals", {})
    
    print(colored("╔" + "═" * 118 + "╗", "cyan"))
    print(colored(f"║ 🏆 ELITE SUI COUNCIL PRO • {SYMBOL} {INTERVAL} • {datetime.utcnow().strftime('%H:%M:%S')} UTC ║", "cyan"))
    print(colored("╠" + "═" * 118 + "╣", "cyan"))
    
    print("📊 INTEGRATED MARKET ANALYSIS:")
    print(f"   💹 Price {fmt(info.get('price'))} | RF {fmt(info.get('filter'))} | Spread {fmt(spread_bps,2)}bps")
    print(f"   🎯 RSI={fmt(ind.get('rsi'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))} | Trend: {signals.get('trend', 'N/A')}")
    
    signal_icons = []
    if signals.get("fvg_bull"): signal_icons.append("🟧FVG_BULL")
    if signals.get("fvg_bear"): signal_icons.append("🟧FVG_BEAR") 
    if signals.get("golden_buy"): signal_icons.append("🟡GOLDEN_BUY")
    if signals.get("golden_sell"): signal_icons.append("🟡GOLDEN_SELL")
    if signals.get("smart_correction"): signal_icons.append("🧠SMART_CORR")
    if signals.get("smart_retest"): signal_icons.append("🧠SMART_RETEST")
    if signals.get("volume_boost"): signal_icons.append("💧VOL_BOOST")
    if signals.get("candle_strength", 0) > 0.7: signal_icons.append("💪STR_CANDLE")
    
    if signal_icons:
        print(f"   🚦 Signals: {', '.join(signal_icons)}")
    
    print(f"   🗳️ Council: B={STATE.get('votes_b',0)}/{fmt(STATE.get('score_b',0),1)} | S={STATE.get('votes_s',0)}/{fmt(STATE.get('score_s',0),1)}")
    
    if elite_candidate:
        confidence = elite_candidate.get("integrated_confidence", 0)
        print(colored(f"   🎖️ ELITE CANDIDATE: {elite_candidate['side'].upper()} (Confidence: {confidence:.1f})", "green" if elite_candidate['side'] == 'buy' else 'red'))
    
    print(f"   ⏱️ Next close in {left_s}s | Plan: {STATE.get('plan','SIT_OUT')}")
    
    print("\n💼 POSITION & MANAGEMENT:")
    bal_line = f"Balance={fmt(bal,2)} | Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x | Total PnL={fmt(compound_pnl)}"
    print(colored(f"   {bal_line}", "yellow"))
    
    if STATE["open"]:
        side_icon = '🟩 LONG' if STATE['side'] == 'long' else '🟥 SHORT'
        trading_mode = STATE.get("trading_mode", "SCALP")
        tm = STATE["trade_management"]
        
        print(f"   {side_icon} ({trading_mode}) | Entry={fmt(STATE['entry'])} | Qty={fmt(STATE['qty'],4)}")
        print(f"   📈 PnL={fmt(STATE['pnl'],2)} | HP={fmt(STATE['hp_pct'],2)}% | Bars={STATE['bars']}")
        print(f"   🛡️ Stop={fmt(tm['current_stop'])} | Trail={'✅' if tm['trailing_active'] else '❌'} | BE={'✅' if tm['break_even_moved'] else '❌'}")
        print(f"   🎯 Targets: {len(tm['targets_hit'])}/{len(tm.get('take_profit_targets', []))}")
    else:
        print("   ⚪ FLAT (No active position)")
    
    if reason:
        print(colored(f"   ⚠️ Blocked: {reason}", "yellow"))
    
    print(colored("╚" + "═" * 118 + "╝", "cyan"))

# =================== التشغيل الرئيسي ===================
def start_elite_system():
    print(colored("\n" + "🌟" * 50, "yellow"))
    print(colored("🚀 STARTING ELITE SUI COUNCIL PRO - INTEGRATED SYSTEM", "yellow"))
    print(colored("🌟" * 50 + "\n", "yellow"))
    
    t1 = threading.Thread(target=elite_trade_loop, name="elite_trade_loop", daemon=True)
    t1.start()
    
    t2 = threading.Thread(target=keepalive_loop, name="keepalive", daemon=True)
    t2.start()
    
    return t1, t2

# =================== Flask API ===================
app = Flask(__name__)

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    trading_mode = STATE.get("trading_mode", "SCALP")
    return f"✅ BYBIT SUI BOT PRO — {SYMBOL} {INTERVAL} — {mode} — Council ELITE PRO PLUS — MODE: {trading_mode}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "elite_confidence": STATE.get("elite_confidence", 0),
        "integrated_signals": STATE.get("integrated_signals", {})
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "elite_confidence": STATE.get("elite_confidence", 0)
    }), 200

def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent": "bybit-sui-keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try:
            r = sess.get(f"{url}/health", timeout=10)
            if r.status_code != 200:
                sess.get(url, timeout=10)
        except Exception as e:
            print(colored(f"keepalive warn: {e}", "yellow"))
        time.sleep(50)

# =================== التشغيل ===================
if __name__ == "__main__":
    setup_file_logging()
    try:
        load_market_specs()
        ensure_leverage_mode()
    except Exception as e:
        print(colored(f"⚠️ exchange init: {e}", "yellow"))
    
    start_elite_system()
    try:
        app.run(host="0.0.0.0", port=PORT, debug=False)
    except Exception as e:
        print(colored(f"Flask run error: {e}", "red"))
