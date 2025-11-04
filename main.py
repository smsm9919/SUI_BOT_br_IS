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

# RF Settings
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0

# ===== UI ICONS =====
IC = {
    "hdr": "🟨",
    "mk": "📊",
    "ind": "🧭",
    "rf": "🧱",
    "bm": "📚",
    "flow": "💧",
    "dash": "📋",
    "ok": "✅",
    "warn": "⚠️",
    "err": "🛑",
    "buy": "🟢 BUY",
    "sell": "🔴 SELL",
    "flat": "⚪ FLAT",
    "pos": "📦",
    "pnl": "💰",
    "bal": "👛",
    "vote": "🗳️",
    "strat": "🎯",
    "gz": "🏅",
}

def _pct(x):
    try:
        return f"{float(x):.2f}%"
    except:
        return "-"

def _num(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except:
        return "-"

def _i(b):
    return IC["ok"] if b else IC["warn"]

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

# =================== دوال RF (Range Filter) ===================
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n)
    wper = (n*2)-1
    return _ema(avrng, wper) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]
        x = float(src.iloc[i])
        r = float(rsize.iloc[i])
        cur = prev
        if x - r > prev:
            cur = x - r
        if x + r < prev:
            cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_signal_closed(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -2 if len(df) >= 2 else -1
        price = float(df["close"].iloc[i]) if len(df) else None
        t = int(df["time"].iloc[i]) if len(df) else int(time.time()*1000)
        return {"time": t, "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    
    d = df.iloc[:-1].copy()
    src = d[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    
    def _bps(a, b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    
    p_prev = float(src.iloc[-1])
    f_prev = float(filt.iloc[-1])
    long_sig = (p_prev > f_prev and _bps(p_prev, f_prev) >= RF_HYST_BPS)
    short_sig = (p_prev < f_prev and _bps(p_prev, f_prev) >= RF_HYST_BPS)
    
    return {"time": int(d["time"].iloc[-1]), "price": p_prev, "long": bool(long_sig),
            "short": bool(short_sig), "filter": f_prev,
            "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

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

# =================== دوال المساعدة الأساسية المطلوبة ===================
def _last_closed_bar_ts(df):
    if len(df) >= 2: 
        return int(df["time"].iloc[-2])
    return int(df["time"].iloc[-1]) if len(df) else 0

def _update_trend_state(ind, info):
    if not STATE["open"]: 
        return
    adx = float(ind.get("adx") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)
    px = float(info.get("price") or STATE.get("entry") or 0.0)
    
    if adx > (STATE.get("peak_adx") or adx): 
        STATE["peak_adx"] = adx
    if rsi > (STATE.get("rsi_peak") or rsi): 
        STATE["rsi_peak"] = rsi
    if rsi < (STATE.get("rsi_trough") or rsi): 
        STATE["rsi_trough"] = rsi
    
    if STATE["side"] == "long":
        if px > (STATE.get("peak_price") or px): 
            STATE["peak_price"] = px
    else:
        if px < (STATE.get("trough_price") or px): 
            STATE["trough_price"] = px

def _price_band(side: str, px: float, max_bps: float):
    if px is None: 
        return None
    if side == "buy":  
        return px * (1 + max_bps/10000.0)
    else:              
        return px * (1 - max_bps/10000.0)

def _bybit_reduceonly_reject(err: Exception) -> bool:
    m = str(err).lower()
    return ("-110017" in m) or ("reduce-only order has same side with current position" in m)

def _params_open(side):
    return {"positionSide": "BOTH", "reduceOnly": False, "positionIdx": 0}

def _params_close():
    return {"positionSide": "BOTH", "reduceOnly": True, "positionIdx": 0}

def _cancel_symbol_orders():
    try:
        if MODE_LIVE:
            ex.cancel_all_orders(SYMBOL)
            print(colored("🧹 canceled all open orders for symbol", "yellow"))
    except Exception as e:
        print(colored(f"⚠️ cancel_all_orders warn: {e}", "yellow"))

def _read_position():
    try:
        poss = with_retry(lambda: ex.fetch_positions(params={"type": "swap"}))
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if not _sym_match(sym, SYMBOL):
                continue
            
            ccxt_side = (p.get("side") or "").strip().lower()
            raw_side = (p.get("info",{}).get("side") or "").strip().lower()
            q_fields = [p.get("contracts"), p.get("positionAmt"), p.get("size"), p.get("info",{}).get("size")]
            q_first = next((float(x) for x in q_fields if x not in (None, "", 0)), 0.0)
            
            side = None
            if ccxt_side in ("long", "short"):
                side = ccxt_side
            elif raw_side in ("buy", "sell"):
                side = "long" if raw_side == "buy" else "short"
            elif q_first != 0:
                side = "long" if q_first > 0 else "short"
            else:
                continue
            
            qty = abs(q_first) if q_first != 0 else 0.0
            if qty <= 0:
                qty = abs(next((float(x) for x in q_fields if isinstance(x,(int,float)) and float(x) != 0), 0.0))
            if qty <= 0:
                continue
            
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0.0) or 0.0
            logging.info(f"READ_POS → side={side} qty={qty} entry={entry} (ccxt_side={ccxt_side} raw_side={raw_side} q={q_first})")
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}", exc_info=True)
    return 0.0, None, None

def reconcile_state():
    exch_qty, exch_side, exch_entry = _read_position()
    if (exch_qty or 0.0) <= 0:
        if STATE.get("open"):
            print(colored("🧹 RECONCILE: exchange flat, resetting local state.", "yellow"))
            _reset_after_close("RECONCILE_FLAT", prev_side=STATE.get("side"))
        return
    
    changed = (not STATE.get("open")) or \
              (STATE.get("side") != exch_side) or \
              (abs((STATE.get("qty") or 0) - exch_qty) > (LOT_STEP or 0.0)) or \
              (abs((STATE.get("entry") or 0) - exch_entry) / max(exch_entry, 1e-9) > 0.001)
    
    if changed:
        STATE.update({
            "open": True, 
            "side": exch_side, 
            "entry": float(exch_entry), 
            "qty": safe_qty(exch_qty)
        })
        print(colored(f"🔄 RECONCILE: synced — {exch_side} qty={fmt(exch_qty,4)} @ {fmt(exch_entry)}", "cyan"))

def _reset_after_close(reason, prev_side=None):
    global LAST_CLOSE_BAR_TS, POST_CHOP_BLOCK_ACTIVE, POST_CHOP_BLOCK_UNTIL_BAR
    prev_side = prev_side or STATE.get("side")
    
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None,
        "hp_pct": 0.0, "strength": 0.0, "entry_strength": 0.0,
        "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
        "peak_price": 0.0, "trough_price": 0.0,
        "opp_rf_count": 0, "scm_line": "", "chop_flag": False,
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
        "trading_mode": "SCALP",
        "rsi_ma_signal": {},
        "elite_confidence": 0.0,
        "integrated_signals": {}
    })
    
    LAST_CLOSE_BAR_TS = LAST_DECISION_BAR_TS
    if reason.startswith("CHOP"):
        POST_CHOP_BLOCK_ACTIVE = True
        POST_CHOP_BLOCK_UNTIL_BAR = (LAST_DECISION_BAR_TS or 0) + 2
    
    logging.info(f"AFTER_CLOSE reason={reason} prev_side={prev_side}")

# =================== دوال اللوج المحسنة ===================
def print_snapshot(symbol, tf, now_utc, px, rf_val, spread_bps, rsi, adx, di_p, di_m, atr,
                   trend_label, council_votes, rsi_ma_sig=None, evx=None, bm_imb=None, cvd=None,
                   plan="SIT_OUT", closes_in_s=None, balance=None, risk_pct=None, compound_pnl=0.0):
    print(f"\n{IC['hdr']}  ELITE COUNCIL • {symbol} {tf} • {now_utc}  ")
    print(f"{IC['mk']}  MARKET ANALYSIS")
    print(f"  $ Price={_num(px,6)} | {IC['rf']} RF={_num(rf_val,6)} | spread={_num(spread_bps,2)} bps")
    print(f"  {IC['ind']} RSI={_num(rsi,2)}  ADX={_num(adx,2)}  +DI={_num(di_p,2)}  -DI={_num(di_m,2)}  ATR={_num(atr,6)}  Trend={trend_label}")
    sig = []
    if rsi_ma_sig: sig.append(f"RSI×MA={rsi_ma_sig}")
    if evx is not None: sig.append(f"EVX={_num(evx,2)}")
    if bm_imb is not None: sig.append(f"Imb={_num(bm_imb,2)}")
    if cvd is not None: sig.append(f"CVD={_num(cvd,0)}")
    if sig: print(f"  {IC['dash']} Signals: " + " | ".join(sig))
    print(f"  {IC['vote']} Council: BUY={council_votes.get('buy',0):.1f} | SELL={council_votes.get('sell',0):.1f}")
    if closes_in_s is not None:
        print(f"  ⏱ Next close in ~{int(closes_in_s)}s | Plan: {plan}")
    print(f"\n{IC['pos']}  POSITION & MANAGEMENT")
    print(f"  {IC['bal']} Balance={_num(balance,2)}  Risk={_pct(risk_pct*100 if risk_pct else 0)}×{LEVERAGE}x  {IC['pnl']} TotalPnL={_num(compound_pnl,2)}")

def log_entry(side, strategy_mode, qty, price, lev, reason, votes, rsi, adx, di_p, di_m, evx=None, rf=None, gz=None, bm_imb=None, cvd=None, balance=None, compound_pnl=0.0):
    tag = IC['buy'] if side.upper().startswith('B') else IC['sell']
    rz = []
    if rf is not None: rz.append(f"RF={_num(rf,6)}")
    if evx is not None: rz.append(f"EVX={_num(evx,2)}")
    if gz: rz.append(f"GZ={gz.get('zone','-')} s={gz.get('score','-')}")
    if bm_imb is not None: rz.append(f"Imb={_num(bm_imb,2)}")
    if cvd is not None: rz.append(f"CVD={_num(cvd,0)}")
    rs = " | ".join(rz)
    print(
      f"\n🚀 ENTRY | {tag} | {IC['strat']} {strategy_mode.upper()} | "
      f"qty={_num(qty,3)}  @ {_num(price,6)}  lev={int(lev)}x | reason={reason} | "
      f"{IC['vote']} votes: BUY={votes.get('buy',0):.1f} SELL={votes.get('sell',0):.1f}\n"
      f"   {IC['ind']} RSI={_num(rsi,2)} ADX={_num(adx,2)} +DI={_num(di_p,2)} -DI={_num(di_m,2)}  {(' | '+rs) if rs else ''}\n"
      f"   {IC['bal']} Eq={_num(balance,2)}  {IC['pnl']} CompoundPnL={_num(compound_pnl,2)}"
    )

def log_manage(side, upnl_pct, trail_active, be_active, partial_done, hold_tp=False, notes=None):
    print(f"{IC['pos']} MANAGE | side={side} | uPnL={_pct(upnl_pct*100)} | trail={_i(trail_active)} be={_i(be_active)} partial={_i(partial_done)} holdTP={_i(hold_tp)}" +
          (f" | {notes}" if notes else ""))

def log_exit(side, reason, qty_closed, price, upnl_pct, session_pnl, compound_pnl, is_strict=False, is_scalp=False):
    mode = "STRICT CLOSE" if is_strict else ("SCALP TP" if is_scalp else "EXIT")
    print(
      f"\n🏁 {mode} | side={side} | qty_closed={_num(qty_closed,3)} @ {_num(price,6)} | uPnL={_pct(upnl_pct*100)} | "
      f"SessionPnL={_num(session_pnl,2)} | {IC['pnl']} CompoundPnL={_num(compound_pnl,2)} | reason={reason}"
    )

# =================== دوال التنفيذ الأساسية ===================
def enhanced_open_market(side, qty, price, strength, reason, df, ind, trading_mode="SCALP"):
    global ENTRY_IN_PROGRESS, _last_entry_attempt_ts, PENDING_OPEN, LAST_SIGNAL_USED
    
    if _now() - _last_entry_attempt_ts < ENTRY_GUARD_WINDOW_SEC:
        print(colored("⏸️ entry guard window — skip", "yellow"))
        return False
    
    if ENTRY_LOCK.locked() or ENTRY_IN_PROGRESS or PENDING_OPEN:
        print(colored("⏸️ entry in progress/pending — skip", "yellow"))
        return False
    
    with ENTRY_LOCK:
        ENTRY_IN_PROGRESS = True
        PENDING_OPEN = True
        
        try:
            ex_qty, ex_side, _ = _read_position()
            if ex_qty and ex_qty > 0:
                print(colored(f"⛔ exchange already has position ({ex_side}) — skip open", "red"))
                return False
            
            _cancel_symbol_orders()
            bal = balance_usdt()
            px = float(price or price_now() or 0.0)
            
            if qty <= 0 or (LOT_MIN and qty < LOT_MIN):
                print(colored(f"❌ skip open (qty too small) — bal={fmt(bal,2)} px={fmt(px)} q={qty}", "red"))
                return False
            
            sp = orderbook_spread_bps()
            if sp is not None and sp > SPREAD_HARD_BPS:
                print(colored(f"⛔ hard spread guard: {fmt(sp,2)}bps > {SPREAD_HARD_BPS}", "red"))
                return False
            
            link = _order_link("ENT")
            if MODE_LIVE:
                ex.create_order(SYMBOL, "market", side, qty, None, {**_params_open(side), "orderLinkId": link})
            else:
                print(colored(f"[PAPER] create_order market {side} {qty}", "cyan"))
            
            time.sleep(0.45)
            cur_qty, cur_side, cur_entry = _read_position()
            
            if not cur_qty or cur_qty <= 0:
                print(colored("❌ open failed — no position filled", "red"))
                return False
            
            expected_side = "long" if side == "buy" else "short"
            if cur_side not in ("long", "short") or cur_side != expected_side:
                print(colored(f"❌ side mismatch after open (expected {expected_side}, got {cur_side}) — strict close", "red"))
                close_market_strict("SIDE_MISMATCH_AFTER_OPEN")
                return False
            
            # إعداد نظام إدارة الصفقة
            atr = float(ind.get("atr", 0))
            setup_trade_management(float(cur_entry), atr, cur_side, strength, trading_mode)
            
            STATE.update({
                "open": True, "side": cur_side, "entry": float(cur_entry),
                "qty": safe_qty(cur_qty), "remaining_size": safe_qty(cur_qty),
                "pnl": 0.0, "bars": 0, "trail": None,
                "hp_pct": 0.0, "strength": float(strength),
                "entry_strength": float(strength),
                "peak_adx": 0.0, "rsi_peak": 50.0, "rsi_trough": 50.0,
                "peak_price": float(cur_entry), "trough_price": float(cur_entry),
                "opp_rf_count": 0, "chop_flag": False,
                "trading_mode": trading_mode
            })
            
            TRADE_TIMES.append(time.time())
            _last_entry_attempt_ts = _now()
            
            LAST_SIGNAL_USED.update({
                "side": side,
                "bar_ts": _last_closed_bar_ts(fetch_ohlcv()),
                "src": reason.split(" ")[0] if reason else "unknown",
                "strength": float(strength),
                "trading_mode": trading_mode
            })
            
            # تسجيل الدخول بالشكل المحسن
            votes = {"buy": STATE.get("votes_b", 0), "sell": STATE.get("votes_s", 0)}
            log_entry(
                side=cur_side, strategy_mode=trading_mode, qty=cur_qty, price=cur_entry, 
                lev=LEVERAGE, reason=reason, votes=votes, rsi=ind.get("rsi", 50), 
                adx=ind.get("adx", 0), di_p=ind.get("plus_di", 0), di_m=ind.get("minus_di", 0),
                balance=bal, compound_pnl=compound_pnl
            )
            
            return True
            
        except Exception as e:
            print(colored(f"❌ open error: {e}", "red"))
            logging.error(f"open_market error: {e}", exc_info=True)
            return False
        finally:
            ENTRY_IN_PROGRESS = False
            PENDING_OPEN = False

def close_market_strict(reason="STRICT"):
    global compound_pnl, LAST_CLOSE_TS, CLOSE_IN_PROGRESS, _last_close_attempt_ts, LAST_CLOSE_BAR_TS
    
    if CLOSE_LOCK.locked() or CLOSE_IN_PROGRESS:
        print(colored("⏸️ close in progress — skip", "yellow"))
        return
    
    if _now() - _last_close_attempt_ts < CLOSE_GUARD_WINDOW_SEC:
        print(colored("⏸️ close guard window — skip", "yellow"))
        return
    
    with CLOSE_LOCK:
        CLOSE_IN_PROGRESS = True
        _last_close_attempt_ts = _now()
        
        try:
            exch_qty, exch_side, exch_entry = _read_position()
            if exch_qty <= 0:
                if STATE.get("open"):
                    _reset_after_close(reason, prev_side=STATE.get("side"))
                    LAST_CLOSE_TS = time.time()
                return
            
            _cancel_symbol_orders()
            side_to_close = "sell" if (exch_side == "long") else "buy"
            qty_to_close = safe_qty(exch_qty)
            
            bid, ask, _ob = None, None, None
            try:
                bid, ask, _ob = _best_bid_ask()
            except Exception: 
                pass
            
            ref = (ask if exch_side == "long" else bid) or price_now() or STATE.get("entry")
            band_px = _price_band(side_to_close, ref, 35.0)  # MAX_SLIP_CLOSE_BPS
            
            link = _order_link("CLS")
            
            try:
                if MODE_LIVE and band_px:
                    params = _params_close()
                    params.update({"timeInForce": "IOC", "orderLinkId": link})
                    ex.create_order(SYMBOL, "limit", side_to_close, qty_to_close, band_px, params)
                else:
                    print(colored(f"[PAPER] limit-IOC reduceOnly {side_to_close} {qty_to_close} @ {fmt(band_px)}", "cyan"))
            except Exception as e1:
                print(colored(f"⚠️ limit IOC close err: {e1}", "yellow"))
                try:
                    if MODE_LIVE:
                        params = _params_close()
                        params.update({"orderLinkId": link})
                        ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)
                    else:
                        print(colored(f"[PAPER] market reduceOnly {side_to_close} {qty_to_close}", "cyan"))
                except Exception as e2:
                    if _bybit_reduceonly_reject(e2):
                        print(colored("↪️ reduceOnly rejected — market w/o reduceOnly (safe after cancel)", "yellow"))
                        params = {"positionSide": "BOTH", "reduceOnly": False, "positionIdx": 0, "timeInForce": "IOC", "orderLinkId": link}
                        if MODE_LIVE:
                            ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)
                        else:
                            print(colored(f"[PAPER] market Fallback {side_to_close} {qty_to_close}", "cyan"))
                    else:
                        raise e2
            
            time.sleep(1.0)
            left_qty, _, _ = _read_position()
            
            if left_qty <= 0:
                px = price_now() or ref
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side
                qty = exch_qty
                pnl = (px - entry_px) * qty * (1 if side == "long" else -1)
                compound_pnl += pnl
                trading_mode = STATE.get("trading_mode", "SCALP")
                
                # تسجيل الخروج بالشكل المحسن
                log_exit(
                    side=side, reason=reason, qty_closed=qty, price=px,
                    upnl_pct=(pnl / (entry_px * qty)) * 100 if entry_px * qty > 0 else 0,
                    session_pnl=pnl, compound_pnl=compound_pnl,
                    is_strict=True, is_scalp=(trading_mode=="SCALP")
                )
                
                _reset_after_close(reason, prev_side=side)
                LAST_CLOSE_TS = time.time()
                return
            
            # Retry logic for remaining position
            for _ in range(3):
                qty_to_close = safe_qty(left_qty)
                try:
                    if MODE_LIVE:
                        params = _params_close()
                        params.update({"timeInForce": "IOC", "orderLinkId": _order_link("CLS")})
                        ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)
                    else:
                        print(colored(f"[PAPER] market retry reduceOnly {side_to_close} {qty_to_close}", "cyan"))
                except Exception as e:
                    print(colored(f"⚠️ market close retry err: {e}", "yellow"))
                
                time.sleep(0.8)
                left_qty, _, _ = _read_position()
                if left_qty <= 0:
                    px = price_now() or ref
                    entry_px = STATE.get("entry") or exch_entry or px
                    side = STATE.get("side") or exch_side
                    qty = exch_qty
                    pnl = (px - entry_px) * qty * (1 if side == "long" else -1)
                    compound_pnl += pnl
                    trading_mode = STATE.get("trading_mode", "SCALP")
                    
                    # تسجيل الخروج بالشكل المحسن
                    log_exit(
                        side=side, reason=reason, qty_closed=qty, price=px,
                        upnl_pct=(pnl / (entry_px * qty)) * 100 if entry_px * qty > 0 else 0,
                        session_pnl=pnl, compound_pnl=compound_pnl,
                        is_strict=True, is_scalp=(trading_mode=="SCALP")
                    )
                    
                    _reset_after_close(reason, prev_side=side)
                    LAST_CLOSE_TS = time.time()
                    return
            
            print(colored("❌ STRICT CLOSE FAILED — residual position still exists", "red"))
            
        except Exception as e:
            print(colored(f"❌ close error: {e}", "red"))
            logging.error(f"close_market_strict error: {e}", exc_info=True)
        finally:
            CLOSE_IN_PROGRESS = False

# =================== دوال إدارة الصفقات ===================
def setup_trade_management(entry_price, atr, side, strength, trading_mode="SCALP"):
    if trading_mode == "TREND":
        tp_targets = [
            {"target": 0.8, "percentage": 0.30},
            {"target": 1.8, "percentage": 0.40}, 
            {"target": 3.0, "percentage": 0.30}
        ]
        break_even_at = TREND_BE_AT
        stop_multiplier = 1.8
    else:  # SCALP
        tp_targets = [
            {"target": 0.4, "percentage": 0.50},
            {"target": 0.8, "percentage": 0.50}
        ]
        break_even_at = SCALP_BE_AT  
        stop_multiplier = 1.2
    
    stop_distance = atr * stop_multiplier
    if side == "long":
        initial_stop = entry_price - stop_distance
    else:
        initial_stop = entry_price + stop_distance
    
    STATE["trade_management"].update({
        "partial_taken": False,
        "targets_hit": [],
        "break_even_moved": False,
        "trailing_active": False,
        "initial_stop": initial_stop,
        "current_stop": initial_stop,
        "take_profit_targets": tp_targets,
        "break_even_at": break_even_at,
        "trail_start_at": 1.2 if trading_mode == "TREND" else 0.6,
        "trading_mode": trading_mode
    })
    
    STATE["trading_mode"] = trading_mode
    
    print(colored(
        f"🎯 إدارة {trading_mode}: وقف {fmt(initial_stop)} | ATR {fmt(atr)} | "
        f"BE@{break_even_at}% | StopMult={stop_multiplier}", 
        "cyan"
    ))

def check_stop_loss(current_price, side):
    tm = STATE["trade_management"]
    stop_price = tm["current_stop"]
    
    if side == "long" and current_price <= stop_price:
        close_market_strict(f"STOP_LOSS {fmt(stop_price)} {STATE['trading_mode']}")
        return True
    elif side == "short" and current_price >= stop_price:
        close_market_strict(f"STOP_LOSS {fmt(stop_price)} {STATE['trading_mode']}")
        return True
    
    return False

def enhanced_manage_position(df, ind, info, zones, trend):
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    current_price = info["price"]
    entry_price = STATE["entry"]
    side = STATE["side"]
    atr = float(ind.get("atr", 0))
    
    # 1. التحقق من وقف الخسارة
    if check_stop_loss(current_price, side):
        return
    
    # 2. إدارة أساسية للصفقة
    rr = (current_price - entry_price) / entry_price * 100 * (1 if side == "long" else -1)
    
    if rr >= TRAIL_START_AT and atr > 0:
        gap = atr * 1.6  # ATR_TRAIL_MULT
        if side == "long":
            new_trail = current_price - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if current_price < STATE["trail"]: 
                close_market_strict(f"TRAIL_ATR(1.6x)")
                return
        else:
            new_trail = current_price + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if current_price > STATE["trail"]: 
                close_market_strict(f"TRAIL_ATR(1.6x)")
                return
    
    # تسجيل إدارة الصفقة
    tm = STATE["trade_management"]
    log_manage(
        side=side, 
        upnl_pct=rr/100,
        trail_active=STATE.get("trail") is not None,
        be_active=tm["break_even_moved"],
        partial_done=tm["partial_taken"],
        hold_tp=len(tm["targets_hit"]) > 0,
        notes=f"ATR={_num(atr,6)}"
    )

# =================== دوال إضافية مطلوبة ===================
def trend_context(ind: dict):
    adx = float(ind.get("adx") or 0.0)
    pdi = float(ind.get("plus_di") or 0.0)
    mdi = float(ind.get("minus_di") or 0.0)
    macd_hist = float(ind.get("macd_hist") or 0.0)
    
    if adx >= 30.0 and abs(pdi - mdi) >= 10.0:
        return "strong_up" if pdi > mdi else "strong_down"
    if pdi > mdi and macd_hist > 0.001:
        return "up"
    if mdi > pdi and macd_hist < -0.001:
        return "down"
    return "sideways"

def orderbook_imbalance(ob, depth=10):
    try:
        asks = ob["asks"][:depth]
        bids = ob["bids"][:depth]
        sum_ask = sum(ask[1] for ask in asks)
        sum_bid = sum(bid[1] for bid in bids)
        tot = max(sum_ask + sum_bid, 1e-9)
        obi = (sum_ask - sum_bid) / tot
        return float(obi)
    except Exception:
        return 0.0

def cvd_update(df: pd.DataFrame):
    if len(df) < 2: 
        return STATE.get("cvd", 0.0)
    
    o = float(df["open"].iloc[-1])
    c = float(df["close"].iloc[-1])
    v = float(df["volume"].iloc[-1])
    
    delta = (1 if c > o else (-1 if c < o else 0)) * v
    prev = STATE.get("cvd", 0.0)
    cvd = prev + (delta - prev) / 10
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
    o = float(d["open"].iloc[-1])
    c = float(d["close"].iloc[-1])
    h = float(d["high"].iloc[-1])
    l = float(d["low"].iloc[-1])
    v = float(d["volume"].iloc[-1])
    
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
    
    equity = float(balance)
    px = max(float(price), 1e-9)
    buffer = 0.97
    notional = equity * RISK_ALLOC * LEVERAGE * buffer
    raw_qty = notional / px
    q_norm = safe_qty(raw_qty)
    
    if q_norm <= 0:
        lot_min = LOT_MIN or 0.1
        need = (lot_min * px) / (LEVERAGE * RISK_ALLOC * buffer)
        print(colored(f"⚠️ balance {fmt(balance,2)} too small — need ≥ {fmt(need,2)} USDT for min lot {lot_min}", "yellow"))
        return 0.0
    
    return q_norm

# =================== دوال المجلس الأساسية (مبسطة) ===================
def council_scm_votes_original(df, ind, info, zones):
    # إرجاع قيم افتراضية للاختبار
    b, s = 0, 0
    b_r, s_r = [], []
    score_b, score_s = 0.0, 0.0
    scm_line = "SCM | BASIC"
    trend = trend_context(ind)
    
    return b, b_r, s, s_r, score_b, score_s, scm_line, trend, False, False

def last_fvg(df):
    return {"bull": None, "bear": None}

def detect_sweep(df, eqh, eql):
    return {"sweep_up": False, "sweep_down": False}

def find_equal_highs_lows(df):
    return None, None

def golden_zone_check(df, ind, side):
    return {"ok": False, "score": 0.0, "reasons": [], "zone": None}

def detect_correction_or_retest(df, ind):
    return {
        "correction_detected": False,
        "retest_detected": False,
        "fib_level": None,
        "trend_direction": "neutral",
        "strength": 0.0,
        "reasons": []
    }

def detect_true_bottom(df, ind):
    return False, 0.0, []

def detect_true_top(df, ind):
    return False, 0.0, []

def decide_plan(df, ind, info, zones):
    class Plan(Enum):
        TREND_RIDE = "TREND_RIDE"
        REVERSAL_SNIPE = "REVERSAL_SNIPE"
        CHOP_HARVEST = "CHOP_HARVEST"
        BREAKOUT_ONLY = "BREAKOUT_ONLY"
        SIT_OUT = "SIT_OUT"
    
    return Plan.SIT_OUT, ["no_analysis"]

# =================== النظام المتكامل الجديد ===================
def unified_council_analysis(df, ind, info, zones):
    """
    تحليل موحد يجمع كل المكونات
    """
    trend = trend_context(ind)
    adx_val = float(ind.get("adx") or 0.0)
    rsi_val = float(ind.get("rsi") or 50.0)
    
    # تحليلات أساسية
    fvg_data = last_fvg(df)
    bid, ask, ob = _best_bid_ask()
    obi = orderbook_imbalance(ob, 10)
    cvd_val = cvd_update(df)
    volume_analysis = analyze_volume(df)
    candle_strength = analyze_candle_strength(df, ind)
    rsi_sig = enhanced_rsi_ma_features(df)
    eqh, eql = find_equal_highs_lows(df)
    sweep_data = detect_sweep(df, eqh, eql)
    gz_buy = golden_zone_check(df, ind, "buy")
    gz_sell = golden_zone_check(df, ind, "sell")
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
    تصويت مجلس موحد
    """
    signals = unified_council_analysis(df, ind, info, zones)
    
    b, b_r, s, s_r, score_b, score_s, scm_line, trend, _, _ = council_scm_votes_original(df, ind, info, zones)
    
    # تعزيزات إضافية
    if signals["fvg_bull"]:
        b += 2
        score_b += 1.0
        b_r.append("fvg_bull")
    
    if signals["volume_boost"]:
        b += 1
        s += 1
        score_b += 0.5
        score_s += 0.5
        b_r.append("volume_boost")
        s_r.append("volume_boost")
    
    scm_line += " | ELITE_INTEGRATED"
    return b, b_r, s, s_r, score_b, score_s, scm_line, trend, False, False

def elite_decision_making(df, ind, info, zones, candidates):
    """
    اتخاذ قرار النخبة
    """
    if not candidates:
        return None
    
    signals = unified_council_analysis(df, ind, info, zones)
    
    best_candidate = None
    max_confidence = 0
    
    for candidate in candidates:
        confidence = candidate.get("score", 0)
        side = candidate["side"]
        
        # تعزيزات الثقة
        if (side == "buy" and signals["trend"] in ["up", "strong_up"]) or (side == "sell" and signals["trend"] in ["down", "strong_down"]):
            confidence += 2.0
        
        if signals["volume_boost"]:
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

def execute_elite_trade(candidate, df, ind, info):
    """
    تنفيذ صفقة النخبة
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
    
    reason = f"ELITE: {candidate.get('reason', '')} | CONF:{confidence:.1f} | MODE:{trading_mode}"
    
    return enhanced_open_market(
        "buy" if side == "buy" else "sell",
        qty,
        price,
        confidence,
        reason,
        df, ind, trading_mode
    )

# =================== الدورة الرئيسية المحسنة ===================
def evaluate_all_elite(df):
    """
    التقييم النخبوي الموحد
    """
    info = rf_signal_closed(df)
    ind = compute_indicators(df)
    zones = {"supply": None, "demand": None}  # zones مبسطة
    
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

            # طباعة السنابشوت المحسن
            print_snapshot(
                symbol=SYMBOL, tf=INTERVAL, now_utc=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                px=px or info.get("price"), rf_val=info.get("filter"), spread_bps=spread_bps,
                rsi=ind.get("rsi", 50), adx=ind.get("adx", 0), di_p=ind.get("plus_di", 0), 
                di_m=ind.get("minus_di", 0), atr=ind.get("atr", 0),
                trend_label=trend, council_votes={"buy": STATE.get("votes_b", 0), "sell": STATE.get("votes_s", 0)},
                rsi_ma_sig=STATE.get("rsi_ma_signal", {}).get("cross"), 
                plan=plan.value if hasattr(plan, 'value') else str(plan),
                closes_in_s=time_to_candle_close(df), balance=bal, risk_pct=RISK_ALLOC, 
                compound_pnl=compound_pnl
            )

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

            if len(df) >= 2 and int(df["time"].iloc[-1]) != int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            time.sleep(1 if time_to_candle_close(df) <= 10 else 3)

        except Exception as e:
            print(colored(f"❌ elite loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"elite_trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(3)

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
