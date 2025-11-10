# -*- coding: utf-8 -*-
"""
SUI Futures Bot — RF-LIVE ONLY (Multi-Exchange: BingX & Bybit)
• Council PRO Ultra Intelligent Decision System 
• Golden Entry + Golden Reversal + Wick Exhaustion
• Smart Scalp & Trend Management with Advanced Profit Taking
• Dynamic TP ladder + Breakeven + ATR-trailing
• Professional Logging & Dashboard
• Multi-Exchange Support: BingX & Bybit
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
# Exchange Selection
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()

# API Keys - Multi-Exchange Support
if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:  # Default to BingX
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")

MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# ==== Run mode / Logging toggles ====
LOG_LEGACY = False
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = f"SUI Council PRO ULTRA v5.0 — {EXCHANGE_NAME.upper()} Multi-Exchange"
print("🔁 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# === Addons config ===
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.30

FLOW_WINDOW = 20
FLOW_SPIKE_Z = 1.60
CVD_SMOOTH = 8

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# RF Settings - Optimized for SUI
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))
RF_MULT   = float(os.getenv("RF_MULT", 3.0))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# ==== SMART PROFIT MANAGEMENT ====
# Scalp Mode - Take profit in one shot
SCALP_TP_TARGETS = [0.45]  # Take profit once at 0.45%
SCALP_TP_FRACTIONS = [1.0]  # Close 100% at once

# Trend Mode - Ride the trend with 2-3 takes based on strength
TREND_TP_TARGETS = [0.50, 1.20, 2.00]  # 3 TP levels for strong trends
TREND_TP_FRACTIONS = [0.40, 0.40, 0.20]  # Scale out gradually

# Dynamic TP / trail
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Smart Exit Tuning ====
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 6.0
GOLDEN_ENTRY_ADX   = 20.0
GOLDEN_REVERSAL_SCORE = 6.5

# ==== Golden Zone Constants ====
FIB_LOW, FIB_HIGH = 0.618, 0.786
MIN_WICK_PCT = 0.35
VOL_MA_LEN = 20
RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
MIN_DISP = 0.8

# ==== Execution & Strategy Thresholds ====
ADX_TREND_MIN = 20
DI_SPREAD_TREND = 6
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (45, 55)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 6.0
GZ_REQ_ADX = 20
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 6
COOLDOWN_SECS_AFTER_CLOSE = 60
ADX_GATE = 17

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)

def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log_i(f"state saved → {STATE_PATH}")
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state() -> dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

# =================== EXCHANGE FACTORY ===================
def make_ex():
    """Factory function for multi-exchange support"""
    exchange_config = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    }
    
    if EXCHANGE_NAME == "bybit":
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bybit(exchange_config)
    else:  # BingX (default)
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bingx(exchange_config)

ex = make_ex()

# =================== EXCHANGE-SPECIFIC ADAPTERS ===================
def exchange_specific_params(side, is_close=False):
    """Handle exchange-specific parameters"""
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:  # BingX
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    """Exchange-specific leverage setting"""
    try:
        if EXCHANGE_NAME == "bybit":
            exchange.set_leverage(leverage, symbol)
        else:  # BingX
            exchange.set_leverage(leverage, symbol, params={"side": "BOTH"})
        log_g(f"✅ {EXCHANGE_NAME.upper()} leverage set: {leverage}x")
    except Exception as e:
        log_w(f"⚠️ set_leverage warning: {e}")

# =================== MARKET SPECS ===================
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        log_i(f"🎯 {SYMBOL} specs → precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        exchange_set_leverage(ex, LEVERAGE, SYMBOL)
        log_i(f"📊 {EXCHANGE_NAME.upper()} position mode: {POSITION_MODE}")
    except Exception as e:
        log_w(f"ensure_leverage_mode: {e}")

# Initialize exchange
try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== CANDLES MODULE ===================
def _body(o,c): return abs(c-o)
def _rng(h,l):  return max(h-l, 1e-12)
def _upper_wick(h,o,c): return h - max(o,c)
def _lower_wick(l,o,c): return min(o,c) - l

def _is_doji(o,c,h,l,th=0.1):
    return _body(o,c) <= th * _rng(h,l)

def _engulfing(po,pc,o,c, min_ratio=1.05):
    bull = (c>o) and (pc<po) and _body(po,pc)>0 and _body(o,c)>=min_ratio*_body(po,pc) and (o<=pc and c>=po)
    bear = (c<o) and (pc>po) and _body(po,pc)>0 and _body(o,c)>=min_ratio*_body(po,pc) and (o>=pc and c<=po)
    return bull, bear

def _hammer_like(o,c,h,l, body_max=0.35, wick_ratio=2.0):
    rng, body = _rng(h,l), _body(o,c)
    lower, upper = _lower_wick(l,o,c), _upper_wick(h,o,c)
    hammer  = (body/rng<=body_max) and (lower>=wick_ratio*body) and (upper<=0.4*body)
    inv_ham = (body/rng<=body_max) and (upper>=wick_ratio*body) and (lower<=0.4*body)
    return hammer, inv_ham

def _shooting_star(o,c,h,l, body_max=0.35, wick_ratio=2.0):
    rng, body = _rng(h,l), _body(o,c)
    return (body/rng<=body_max) and (_upper_wick(h,o,c)>=wick_ratio*body) and (_lower_wick(l,o,c)<=0.4*body)

def _marubozu(o,c,h,l, min_body=0.9): return _body(o,c)/_rng(h,l) >= min_body
def _piercing(po,pc,o,c, min_pen=0.5): return (pc<po) and (c>o) and (c>(po - min_pen*(po-pc))) and (o<pc)
def _dark_cloud(po,pc,o,c, min_pen=0.5): return (pc>po) and (c<o) and (c<(po + min_pen*(pc-po))) and (o>pc)

def _tweezer(ph,pl,h,l, tol=0.15):
    top = abs(h-ph) <= tol*max(h,ph)
    bot = abs(l-pl) <= tol*max(l,pl)
    return top, bot

def compute_candles(df):
    """
    يرجّع: buy/sell + score لكل اتجاه + فتائل كبيرة (exhaustion) + tags
    """
    if len(df) < 5:
        return {"buy":False,"sell":False,"score_buy":0.0,"score_sell":0.0,
                "wick_up_big":False,"wick_dn_big":False,"doji":False,"pattern":None}

    o1,h1,l1,c1 = float(df["open"].iloc[-2]), float(df["high"].iloc[-2]), float(df["low"].iloc[-2]), float(df["close"].iloc[-2])
    o0,h0,l0,c0 = float(df["open"].iloc[-3]), float(df["high"].iloc[-3]), float(df["low"].iloc[-3]), float(df["close"].iloc[-3])

    strength_b = strength_s = 0.0
    tags = []

    bull_eng, bear_eng = _engulfing(o0,c0,o1,c1)
    if bull_eng: strength_b += 2.0; tags.append("bull_engulf")
    if bear_eng: strength_s += 2.0; tags.append("bear_engulf")

    ham, inv = _hammer_like(o1,c1,h1,l1)
    if ham: strength_b += 1.5; tags.append("hammer")
    if inv: strength_s += 1.5; tags.append("inverted_hammer")

    if _shooting_star(o1,c1,h1,l1): strength_s += 1.5; tags.append("shooting_star")
    if _piercing(o0,c0,o1,c1):      strength_b += 1.2; tags.append("piercing")
    if _dark_cloud(o0,c0,o1,c1):    strength_s += 1.2; tags.append("dark_cloud")

    is_doji = _is_doji(o1,c1,h1,l1)
    if is_doji: tags.append("doji")

    tw_top, tw_bot = _tweezer(h0,l0,h1,l1)
    if tw_bot: strength_b += 1.0; tags.append("tweezer_bottom")
    if tw_top: strength_s += 1.0; tags.append("tweezer_top")

    if _marubozu(o1,c1,h1,l1):
        if c1>o1: strength_b += 1.0; tags.append("marubozu_bull")
        else:     strength_s += 1.0; tags.append("marubozu_bear")

    # فتائل كبيرة = إرهاق
    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:  # تخفيف ثقة
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
    }

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    """التحقق من بيئة التنفيذ عند الإقلاع"""
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 SMART PROFIT: Scalp(1-shot) | Trend(3-TP)", flush=True)
    print(f"📈 CANDLES: Full patterns + Wick exhaustion + Golden reversal", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

# =================== ENHANCED INDICATORS ===================
def sma(series, n: int):
    return series.rolling(n, min_periods=1).mean()

def compute_rsi(close, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def rsi_ma_context(df):
    if len(df) < max(RSI_MA_LEN, 14):
        return {"rsi": 50, "rsi_ma": 50, "cross": "none", "trendZ": "none", "in_chop": True}
    
    rsi = compute_rsi(df['close'].astype(float), 14)
    rsi_ma = sma(rsi, RSI_MA_LEN)
    
    cross = "none"
    if len(rsi) >= 2:
        if (rsi.iloc[-2] <= rsi_ma.iloc[-2]) and (rsi.iloc[-1] > rsi_ma.iloc[-1]):
            cross = "bull"
        elif (rsi.iloc[-2] >= rsi_ma.iloc[-2]) and (rsi.iloc[-1] < rsi_ma.iloc[-1]):
            cross = "bear"
    
    above = (rsi > rsi_ma)
    below = (rsi < rsi_ma)
    persist_bull = above.tail(RSI_TREND_PERSIST).all() if len(above) >= RSI_TREND_PERSIST else False
    persist_bear = below.tail(RSI_TREND_PERSIST).all() if len(below) >= RSI_TREND_PERSIST else False
    
    current_rsi = float(rsi.iloc[-1])
    in_chop = RSI_NEUTRAL_BAND[0] <= current_rsi <= RSI_NEUTRAL_BAND[1]
    
    return {
        "rsi": current_rsi,
        "rsi_ma": float(rsi_ma.iloc[-1]),
        "cross": cross,
        "trendZ": "bull" if persist_bull else ("bear" if persist_bear else "none"),
        "in_chop": in_chop
    }

# =================== SMART GOLDEN ZONE DETECTION ===================
def _ema_gz(series, n):
    return series.ewm(span=n, adjust=False).mean()

def _rsi_fallback_gz(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def _body_wicks_gz(h, l, o, c):
    rng = max(1e-9, h - l)
    body = abs(c - o) / rng
    up_wick = (h - max(c, o)) / rng
    low_wick = (min(c, o) - l) / rng
    return body, up_wick, low_wick

def _displacement_gz(closes):
    if len(closes) < 22:
        return 0.0
    recent_std = closes.tail(20).std()
    return abs(closes.iloc[-1] - closes.iloc[-2]) / max(recent_std, 1e-9)

def _last_impulse_gz(df):
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    
    lookback = min(120, len(df))
    recent_highs = h.tail(lookback)
    recent_lows = l.tail(lookback)
    
    hh_idx = recent_highs.idxmax()
    ll_idx = recent_lows.idxmin()
    
    hh = recent_highs.max()
    ll = recent_lows.min()
    
    if hh_idx < ll_idx:
        return ("down", hh_idx, ll_idx, hh, ll)
    else:
        return ("up", ll_idx, hh_idx, ll, hh)

def golden_zone_check(df, ind=None, side_hint=None):
    if len(df) < 60:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        v = df['volume'].astype(float)
        
        impulse_data = _last_impulse_gz(df)
        if not impulse_data:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["no_clear_impulse"]}
            
        side, idx1, idx2, p1, p2 = impulse_data
        
        if side == "down":
            swing_hi, swing_lo = p1, p2
            f618 = swing_lo + FIB_LOW * (swing_hi - swing_lo)
            f786 = swing_lo + FIB_HIGH * (swing_hi - swing_lo)
            zone_type = "golden_bottom"
        else:
            swing_lo, swing_hi = p1, p2
            f618 = swing_hi - FIB_HIGH * (swing_hi - swing_lo)
            f786 = swing_hi - FIB_LOW * (swing_hi - swing_lo)
            zone_type = "golden_top"
        
        last_close = float(c.iloc[-1])
        in_zone = (f618 <= last_close <= f786) if side == "down" else (f786 <= last_close <= f618)
        
        if not in_zone:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"price_not_in_zone {last_close:.6f} vs [{f618:.6f},{f786:.6f}]"]}
        
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        current_open = float(o.iloc[-1])
        
        body, up_wick, low_wick = _body_wicks_gz(current_high, current_low, current_open, last_close)
        
        vol_ma = v.rolling(VOL_MA_LEN).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma * 0.9
        
        rsi_series = _rsi_fallback_gz(c, RSI_LEN_GZ)
        rsi_ma_series = _ema_gz(rsi_series, RSI_MA_LEN_GZ)
        rsi_last = float(rsi_series.iloc[-1])
        rsi_ma_last = float(rsi_ma_series.iloc[-1])
        
        adx = ind.get('adx', 0) if ind else 0
        disp = _displacement_gz(c)
        
        if side == "down":
            wick_ok = low_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last > rsi_ma_last and rsi_last < 70
            candle_bullish = last_close > current_open
        else:
            wick_ok = up_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last < rsi_ma_last and rsi_last > 30
            candle_bullish = last_close < current_open
        
        score = 0.0
        reasons = []
        
        if adx >= GZ_REQ_ADX:
            score += 2.0
            reasons.append(f"ADX_{adx:.1f}")
        
        if disp >= MIN_DISP:
            score += 1.5
            reasons.append(f"DISP_{disp:.2f}")
        
        if wick_ok:
            score += 1.5
            reasons.append("wick_ok")
        
        if vol_ok:
            score += 1.0
            reasons.append("vol_ok")
        
        if rsi_ok:
            score += 1.5
            reasons.append("rsi_ok")
        
        if candle_bullish:
            score += 0.5
            reasons.append("candle_confirm")
        
        score += 2.0
        reasons.append("in_zone")
        
        ok = (score >= GZ_MIN_SCORE and in_zone and adx >= GZ_REQ_ADX)
        
        if LOG_ADDONS and in_zone:
            print(f"[GZ DEBUG] type={zone_type} zone={f618:.6f}-{f786:.6f} price={last_close:.6f} score={score:.1f} adx={adx:.1f} disp={disp:.2f} wick_ok={wick_ok} vol_ok={vol_ok} rsi_ok={rsi_ok}")
        
        return {
            "ok": ok,
            "score": round(score, 2),
            "zone": {
                "type": zone_type,
                "f618": f618,
                "f786": f786,
                "swing_high": swing_hi if side == "down" else swing_lo,
                "swing_low": swing_lo if side == "down" else swing_hi
            } if ok else None,
            "reasons": reasons
        }
        
    except Exception as e:
        log_w(f"golden_zone_check error: {e}")
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {str(e)}"]}

def decide_strategy_mode(df, adx=None, di_plus=None, di_minus=None, rsi_ctx=None):
    if adx is None or di_plus is None or di_minus is None:
        ind = compute_indicators(df)
        adx = ind.get('adx', 0)
        di_plus = ind.get('plus_di', 0)
        di_minus = ind.get('minus_di', 0)
    
    if rsi_ctx is None:
        rsi_ctx = rsi_ma_context(df)
    
    di_spread = abs(di_plus - di_minus)
    
    strong_trend = (
        (adx >= ADX_TREND_MIN and di_spread >= DI_SPREAD_TREND) or
        (rsi_ctx["trendZ"] in ("bull", "bear") and not rsi_ctx["in_chop"])
    )
    
    mode = "trend" if strong_trend else "scalp"
    why = "adx/di_trend" if adx >= ADX_TREND_MIN else ("rsi_trendZ" if rsi_ctx["trendZ"] != "none" else "scalp_default")
    
    return {"mode": mode, "why": why}

# =================== ADVANCED SMART INDICATORS ===================
def super_trend(df, period=10, multiplier=3):
    """مؤشر Super Trend المتقدم"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period).mean()
        
        hl2 = (h + l) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        st = [0.0] * len(df)
        trend = [0] * len(df)
        
        st[0] = upper_band.iloc[0]
        trend[0] = 1
        
        for i in range(1, len(df)):
            if c.iloc[i] > st[i-1]:
                trend[i] = 1
                st[i] = max(lower_band.iloc[i], st[i-1])
            else:
                trend[i] = -1
                st[i] = min(upper_band.iloc[i], st[i-1])
        
        current_trend = trend[-1]
        trend_strength = abs(c.iloc[-1] - st[-1]) / atr.iloc[-1]
        
        return {
            "trend": current_trend,
            "strength": trend_strength,
            "value": st[-1],
            "signal": "buy" if current_trend == 1 else "sell"
        }
    except Exception as e:
        return {"trend": 0, "strength": 0, "value": 0, "signal": "neutral"}

def volume_profile_analysis(df, window=20):
    """تحليل توزيع الأحجام للسعر"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        v = df['volume'].astype(float)
        c = df['close'].astype(float)
        
        price_levels = np.linspace(l.tail(window).min(), h.tail(window).max(), 50)
        
        volume_at_price = []
        for i in range(len(price_levels)-1):
            mask = (c >= price_levels[i]) & (c < price_levels[i+1])
            volume_sum = v[mask].sum()
            volume_at_price.append(volume_sum)
        
        max_volume_idx = np.argmax(volume_at_price)
        poc = price_levels[max_volume_idx]
        
        vwap = (c * v).cumsum() / v.cumsum()
        current_vwap = vwap.iloc[-1]
        
        volume_strength = volume_at_price[max_volume_idx] / np.mean(volume_at_price)
        
        return {
            "poc": poc,
            "vwap": current_vwap,
            "volume_strength": volume_strength,
            "above_vwap": c.iloc[-1] > current_vwap,
            "signal": "bullish" if c.iloc[-1] > poc else "bearish"
        }
    except Exception as e:
        return {"poc": 0, "vwap": 0, "volume_strength": 0, "above_vwap": False, "signal": "neutral"}

def market_structure_break(df, lookback=50):
    """كسر هيكل السوق - اكتشاف القمم والقيعان الجديدة"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        
        recent_high = h.tail(lookback).max()
        recent_low = l.tail(lookback).min()
        
        current_price = c.iloc[-1]
        prev_high = h.tail(lookback+10).head(lookback).max()
        prev_low = l.tail(lookback+10).head(lookback).min()
        
        break_high = current_price > prev_high
        break_low = current_price < prev_low
        
        if break_high:
            strength = (current_price - prev_high) / prev_high * 100
            signal = "strong_buy"
        elif break_low:
            strength = (prev_low - current_price) / prev_low * 100
            signal = "strong_sell"
        else:
            strength = 0
            signal = "neutral"
        
        return {
            "break_high": break_high,
            "break_low": break_low,
            "strength": strength,
            "signal": signal,
            "resistance": prev_high,
            "support": prev_low
        }
    except Exception as e:
        return {"break_high": False, "break_low": False, "strength": 0, "signal": "neutral", "resistance": 0, "support": 0}

def momentum_oscillator(df, fast=12, slow=26, signal=9):
    """مؤشر الزخم المتقدم مع تأكيدات متعددة"""
    try:
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        ema_fast = c.ewm(span=fast).mean()
        ema_slow = c.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        volume_ema = v.ewm(span=20).mean()
        volume_momentum = (v / volume_ema).iloc[-1]
        
        macd_trend = "bullish" if macd.iloc[-1] > macd_signal.iloc[-1] else "bearish"
        histogram_trend = "rising" if macd_histogram.iloc[-1] > macd_histogram.iloc[-2] else "falling"
        
        momentum_strength = abs(macd_histogram.iloc[-1]) / (c.rolling(50).std().iloc[-1] + 1e-12)
        
        return {
            "macd": macd.iloc[-1],
            "macd_signal": macd_signal.iloc[-1],
            "macd_histogram": macd_histogram.iloc[-1],
            "trend": macd_trend,
            "histogram_trend": histogram_trend,
            "volume_momentum": volume_momentum,
            "strength": momentum_strength,
            "signal": "buy" if (macd_trend == "bullish" and histogram_trend == "rising" and volume_momentum > 1) else "sell"
        }
    except Exception as e:
        return {"macd": 0, "macd_signal": 0, "macd_histogram": 0, "trend": "neutral", "histogram_trend": "neutral", "volume_momentum": 1, "strength": 0, "signal": "neutral"}

def advanced_market_analysis(df):
    """تحليل السوق المتقدم يجمع كل المؤشرات"""
    super_trend_data = super_trend(df)
    volume_data = volume_profile_analysis(df)
    structure_data = market_structure_break(df)
    momentum_data = momentum_oscillator(df)
    
    trend_score = 0
    trend_signals = []
    
    if super_trend_data["trend"] == 1:
        trend_score += 2
        trend_signals.append("SuperTrend صاعد")
    if volume_data["above_vwap"]:
        trend_score += 1
        trend_signals.append("فوق VWAP")
    if structure_data["break_high"]:
        trend_score += 3
        trend_signals.append("كسر مقاومة")
    if momentum_data["trend"] == "bullish":
        trend_score += 1
        trend_signals.append("MACD إيجابي")
    
    bearish_score = 0
    bearish_signals = []
    
    if super_trend_data["trend"] == -1:
        bearish_score += 2
        bearish_signals.append("SuperTrend هابط")
    if not volume_data["above_vwap"]:
        bearish_score += 1
        bearish_signals.append("تحت VWAP")
    if structure_data["break_low"]:
        bearish_score += 3
        bearish_signals.append("كسر دعم")
    if momentum_data["trend"] == "bearish":
        bearish_score += 1
        bearish_signals.append("MACD سلبي")
    
    return {
        "trend_score": trend_score,
        "bearish_score": bearish_score,
        "trend_signals": trend_signals,
        "bearish_signals": bearish_signals,
        "super_trend": super_trend_data,
        "volume_profile": volume_data,
        "market_structure": structure_data,
        "momentum": momentum_data,
        "overall_signal": "buy" if trend_score > bearish_score + 1 else "sell" if bearish_score > trend_score + 1 else "neutral"
    }

# =================== ULTRA INTELLIGENT COUNCIL VOTING ===================
def council_votes_pro_enhanced(df):
    """مجلس تصويت ذكي خارق مع مؤشرات متقدمة"""
    try:
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)
        cd = compute_candles(df)
        
        advanced_analysis = advanced_market_analysis(df)
        
        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []

        adx = ind.get('adx', 0)
        plus_di = ind.get('plus_di', 0)
        minus_di = ind.get('minus_di', 0)
        di_spread = abs(plus_di - minus_di)

        # === المؤشرات المتقدمة ===
        super_trend_signal = advanced_analysis["super_trend"]["signal"]
        volume_signal = advanced_analysis["volume_profile"]["signal"]
        structure_signal = advanced_analysis["market_structure"]["signal"]
        momentum_signal = advanced_analysis["momentum"]["signal"]
        
        # Super Trend
        if super_trend_signal == "buy":
            votes_b += 3; score_b += 2.0; logs.append("🚀 Super Trend صاعد قوي")
        elif super_trend_signal == "sell":
            votes_s += 3; score_s += 2.0; logs.append("💥 Super Trend هابط قوي")
        
        # Volume Profile
        if volume_signal == "bullish":
            votes_b += 2; score_b += 1.5; logs.append("📊 حجم الشراء مهيمن")
        elif volume_signal == "bearish":
            votes_s += 2; score_s += 1.5; logs.append("📊 حجم البيع مهيمن")
        
        # Market Structure Break
        if structure_signal == "strong_buy":
            votes_b += 4; score_b += 2.5; logs.append("🔄 كسر مقاومة - اتجاه جديد صاعد")
        elif structure_signal == "strong_sell":
            votes_s += 4; score_s += 2.5; logs.append("🔄 كسر دعم - اتجاه جديد هابط")
        
        # Momentum
        if momentum_signal == "buy":
            votes_b += 2; score_b += 1.0; logs.append("📈 زخم إيجابي قوي")
        elif momentum_signal == "sell":
            votes_s += 2; score_s += 1.0; logs.append("📉 زخم سلبي قوي")

        # === المؤشرات الأساسية المحسنة ===
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                votes_b += 2; score_b += 1.5; logs.append("📈 ترند صاعد قوي ADX+DI")
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                votes_s += 2; score_s += 1.5; logs.append("📉 ترند هابط قوي ADX+DI")

        # RSI-MA cross / Trend-Z
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 70:
            votes_b += 2; score_b += 1.0; logs.append("🟢 RSI-MA إيجابي")
        elif rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 30:
            votes_s += 2; score_s += 1.0; logs.append("🔴 RSI-MA سلبي")

        if rsi_ctx["trendZ"] == "bull":
            votes_b += 3; score_b += 1.5; logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_ctx["trendZ"] == "bear":
            votes_s += 3; score_s += 1.5; logs.append("💥 RSI ترند هابط مستمر")

        # Golden Zones
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom':
                votes_b += 3; score_b += 1.5; logs.append(f"🏆 قاع ذهبي (قوة: {gz['score']:.1f})")
            elif gz['zone']['type'] == 'golden_top':
                votes_s += 3; score_s += 1.5; logs.append(f"🏆 قمة ذهبية (قوة: {gz['score']:.1f})")

        # الشموع اليابانية
        if cd["score_buy"]>0:
            score_b += min(2.5, cd["score_buy"]); logs.append(f"🕯️ شموع BUY ({cd['pattern']}) +{cd['score_buy']:.1f}")
        if cd["score_sell"]>0:
            score_s += min(2.5, cd["score_sell"]); logs.append(f"🕯️ شموع SELL ({cd['pattern']}) +{cd['score_sell']:.1f}")

        # === التأكيدات الذكية ===
        advanced_bull_signals = sum([
            super_trend_signal == "buy",
            volume_signal == "bullish",
            structure_signal == "strong_buy",
            momentum_signal == "buy"
        ])
        
        advanced_bear_signals = sum([
            super_trend_signal == "sell",
            volume_signal == "bearish",
            structure_signal == "strong_sell",
            momentum_signal == "sell"
        ])
        
        if advanced_bull_signals >= 3:
            score_b *= 1.3; logs.append("🎯 تأكيد قوي متعدد للإشارة الصاعدة")
        if advanced_bear_signals >= 3:
            score_s *= 1.3; logs.append("🎯 تأكيد قوي متعدد للإشارة الهابطة")

        # === مرشحات الذكاء الاصطناعي ===
        if rsi_ctx["in_chop"]:
            score_b *= 0.7; score_s *= 0.7; logs.append("⚖️ RSI محايد — تخفيض ثقة")

        if adx < ADX_GATE:
            score_b *= 0.8; score_s *= 0.8; logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        if adx > 30:
            if plus_di > minus_di:
                score_b *= 1.2; logs.append("🔥 تعزيز صاعد في ترند قوي")
            else:
                score_s *= 1.2; logs.append("🔥 تعزيز هابط في ترند قوي")

        ind.update({
            "rsi_ma": rsi_ctx["rsi_ma"],
            "rsi_trendz": rsi_ctx["trendZ"],
            "di_spread": di_spread,
            "gz": gz,
            "candle_buy_score": cd["score_buy"],
            "candle_sell_score": cd["score_sell"],
            "wick_up_big": cd["wick_up_big"],
            "wick_dn_big": cd["wick_dn_big"],
            "candle_tags": cd["pattern"],
            "super_trend": advanced_analysis["super_trend"],
            "volume_profile": advanced_analysis["volume_profile"],
            "market_structure": advanced_analysis["market_structure"],
            "momentum": advanced_analysis["momentum"],
            "advanced_signal": advanced_analysis["overall_signal"]
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": round(score_b, 2), "score_s": round(score_s, 2),
            "logs": logs, "ind": ind, "gz": gz, "candles": cd,
            "advanced": advanced_analysis
        }
    except Exception as e:
        log_w(f"council_votes_pro_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"gz":None,"candles":{},"advanced":{}}

# =================== SMART ENTRY SYSTEM ===================
def smart_entry_system(df, council_data):
    """نظام دخول ذكي يحدد أفضل لحظة للدخول"""
    advanced = council_data.get("advanced", {})
    ind = council_data.get("ind", {})
    
    signals_buy = []
    signals_sell = []
    
    # Super Trend
    if advanced.get("super_trend", {}).get("signal") == "buy":
        signals_buy.append("SuperTrend")
    elif advanced.get("super_trend", {}).get("signal") == "sell":
        signals_sell.append("SuperTrend")
    
    # Volume Profile
    if advanced.get("volume_profile", {}).get("signal") == "bullish":
        signals_buy.append("Volume")
    elif advanced.get("volume_profile", {}).get("signal") == "bearish":
        signals_sell.append("Volume")
    
    # Market Structure
    if advanced.get("market_structure", {}).get("signal") == "strong_buy":
        signals_buy.append("Structure")
    elif advanced.get("market_structure", {}).get("signal") == "strong_sell":
        signals_sell.append("Structure")
    
    # Momentum
    if advanced.get("momentum", {}).get("signal") == "buy":
        signals_buy.append("Momentum")
    elif advanced.get("momentum", {}).get("signal") == "sell":
        signals_sell.append("Momentum")
    
    # Golden Zone
    gz = council_data.get("gz", {})
    if gz and gz.get("ok"):
        if gz['zone']['type'] == 'golden_bottom':
            signals_buy.append("GoldenZone")
        elif gz['zone']['type'] == 'golden_top':
            signals_sell.append("GoldenZone")
    
    # Candles
    candles = council_data.get("candles", {})
    if candles.get("buy"):
        signals_buy.append("Candles")
    if candles.get("sell"):
        signals_sell.append("Candles")
    
    confidence_buy = len(signals_buy)
    confidence_sell = len(signals_sell)
    
    min_signals = 3
    
    entry_signal = None
    if confidence_buy >= min_signals and council_data["score_b"] > 8:
        entry_signal = "buy"
        log_i(f"🎯 ENTRY SIGNAL: BUY | إشارات: {', '.join(signals_buy)} | ثقة: {confidence_buy}/7")
    
    elif confidence_sell >= min_signals and council_data["score_s"] > 8:
        entry_signal = "sell" 
        log_i(f"🎯 ENTRY SIGNAL: SELL | إشارات: {', '.join(signals_sell)} | ثقة: {confidence_sell}/7")
    
    return {
        "entry_signal": entry_signal,
        "confidence_buy": confidence_buy,
        "confidence_sell": confidence_sell,
        "signals_buy": signals_buy,
        "signals_sell": signals_sell,
        "min_signals": min_signals
    }

# =================== SMART PROFIT MANAGEMENT ===================
def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط"""
    if mode == "scalp":
        return {
            "tp_targets": SCALP_TP_TARGETS,
            "tp_fractions": SCALP_TP_FRACTIONS,
            "be_activate_pct": SCALP_BE_AFTER / 100.0,
            "trail_activate_pct": 0.8 / 100.0,
            "atr_trail_mult": SCALP_ATR_MULT,
            "close_aggression": "high",
            "max_tp_levels": len(SCALP_TP_TARGETS)
        }
    else:  # trend
        return {
            "tp_targets": TREND_TP_TARGETS,
            "tp_fractions": TREND_TP_FRACTIONS,
            "be_activate_pct": TREND_BE_AFTER / 100.0,
            "trail_activate_pct": 1.2 / 100.0,
            "atr_trail_mult": TREND_ATR_MULT,
            "close_aggression": "medium",
            "max_tp_levels": len(TREND_TP_TARGETS)
        }

def manage_take_profits(state, current_price, pnl_pct, management_config, mode):
    """إدارة جني الأرباح الذكية حسب نمط التداول"""
    if state["qty"] <= 0:
        return
    
    tp_targets = management_config["tp_targets"]
    tp_fractions = management_config["tp_fractions"]
    
    # Scalp Mode - Take profit in one shot
    if mode == "scalp":
        if not state.get("tp1_done") and pnl_pct >= tp_targets[0]/100.0:
            close_fraction = tp_fractions[0]
            close_qty = safe_qty(state["qty"] * close_fraction)
            if close_qty > 0:
                close_side = "sell" if state["side"] == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"✅ SCALP TP HIT: closed {close_fraction*100}% at {tp_targets[0]}%")
                    except Exception as e:
                        log_e(f"❌ Scalp TP close failed: {e}")
                state["qty"] = safe_qty(state["qty"] - close_qty)
                state["tp1_done"] = True
                state["profit_targets_achieved"] += 1
                log_i(f"💰 SCALP SUCCESS: Taken {tp_targets[0]}% profit in one shot!")
    
    # Trend Mode - Multiple take profits (2-3 levels)
    else:
        tp_levels_hit = state.get("tp_levels_hit", [False] * len(tp_targets))
        
        for i, (tp_pct, frac) in enumerate(zip(tp_targets, tp_fractions)):
            if not tp_levels_hit[i] and pnl_pct >= tp_pct/100.0:
                close_qty = safe_qty(state["qty"] * frac)
                if close_qty > 0:
                    close_side = "sell" if state["side"] == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            log_g(f"✅ TREND TP{i+1} HIT: closed {frac*100}% at {tp_pct}%")
                        except Exception as e:
                            log_e(f"❌ Trend TP{i+1} close failed: {e}")
                    state["qty"] = safe_qty(state["qty"] - close_qty)
                    tp_levels_hit[i] = True
                    state["profit_targets_achieved"] += 1
                    log_i(f"📈 TREND RIDING: Taken {tp_pct}% profit ({frac*100}% of position)")
        
        state["tp_levels_hit"] = tp_levels_hit

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع جني أرباح ذكي حسب النمط"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    management = STATE.get("management", {})
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    # إدارة جني الأرباح الذكية
    manage_take_profits(STATE, px, pnl_pct, management, mode)

    snap = emit_snapshots(ex, SYMBOL, df)
    gz = snap.get("gz", {})  # إصلاح الخطأ - استخدام get بدلاً من []
    
    exit_signal = smart_exit_guard(STATE, df, ind, snap.get("flow", {}), snap.get("bm", {}), 
                                 px, pnl_pct/100, mode, side, entry, gz)
    
    if exit_signal["log"]:
        print(f"🔔 {exit_signal['log']}", flush=True)

    if exit_signal["action"] == "partial" and not STATE.get("partial_taken"):
        partial_qty = safe_qty(qty * exit_signal.get("qty_pct", 0.3))
        if partial_qty > 0:
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, partial_qty, None, params)
                    log_g(f"✅ PARTIAL CLOSE: {partial_qty:.4f} | {exit_signal['why']}")
                    STATE["partial_taken"] = True
                    STATE["qty"] = safe_qty(qty - partial_qty)
                except Exception as e:
                    log_e(f"❌ Partial close failed: {e}")
            else:
                log_i(f"DRY_RUN: Partial close {partial_qty:.4f}")
    
    elif exit_signal["action"] == "tighten" and not STATE.get("trail_tightened"):
        STATE["trail_tightened"] = True
        STATE["trail"] = None
        log_i(f"🔄 TRAIL TIGHTENED: {exit_signal['why']}")
    
    elif exit_signal["action"] == "close":
        log_w(f"🚨 SMART EXIT: {exit_signal['why']}")
        close_market_strict(f"smart_exit_{exit_signal['why']}")
        return

    current_atr = ind.get("atr", 0.0)
    be_activate_pct = management.get("be_activate_pct", BREAKEVEN_AFTER/100.0)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT/100.0)
    atr_trail_mult = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    if not STATE.get("breakeven_armed") and pnl_pct/100 >= be_activate_pct:
        STATE["breakeven_armed"] = True
        STATE["breakeven"] = entry
        log_i("BREAKEVEN ARMED")

    if not STATE.get("trail_active") and pnl_pct/100 >= trail_activate_pct:
        STATE["trail_active"] = True
        log_i("TRAIL ACTIVATED")

    if STATE.get("trail_active"):
        trail_mult = TRAIL_TIGHT_MULT if STATE.get("trail_tightened") else atr_trail_mult
        if side == "long":
            new_trail = px - (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:
            new_trail = px + (current_atr * trail_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail

    if STATE.get("trail"):
        if (side == "long" and px <= STATE["trail"]) or (side == "short" and px >= STATE["trail"]):
            log_w(f"TRAIL STOP: {px} vs trail {STATE['trail']}")
            close_market_strict("trail_stop")

    if STATE.get("breakeven"):
        if (side == "long" and px <= STATE["breakeven"]) or (side == "short" and px >= STATE["breakeven"]):
            log_w(f"BREAKEVEN STOP: {px} vs breakeven {STATE['breakeven']}")
            close_market_strict("breakeven_stop")

    if STATE["qty"] <= FINAL_CHUNK_QTY:
        log_w(f"DUST GUARD: qty {STATE['qty']} <= {FINAL_CHUNK_QTY}, closing...")
        close_market_strict("dust_guard")

# =================== POSITION RECOVERY ===================
def _normalize_side(pos):
    side = pos.get("side") or pos.get("positionSide") or ""
    if side: return side.upper()
    qty = float(pos.get("contracts") or pos.get("positionAmt") or pos.get("size") or 0)
    return "LONG" if qty > 0 else ("SHORT" if qty < 0 else "")

def fetch_live_position(exchange, symbol: str):
    try:
        if hasattr(exchange, "fetch_positions"):
            arr = exchange.fetch_positions([symbol])
            for p in arr or []:
                sym = p.get("symbol") or p.get("info", {}).get("symbol")
                if sym and symbol.replace(":","") in sym.replace(":",""):
                    side = _normalize_side(p)
                    qty = abs(float(p.get("contracts") or p.get("positionAmt") or p.get("info",{}).get("size",0) or 0))
                    if qty > 0:
                        entry = float(p.get("entryPrice") or p.get("info",{}).get("entryPrice") or 0.0)
                        lev = float(p.get("leverage") or p.get("info",{}).get("leverage") or 0.0)
                        unr = float(p.get("unrealizedPnl") or 0.0)
                        return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
        if hasattr(exchange, "fetch_position"):
            p = exchange.fetch_position(symbol)
            side = _normalize_side(p); qty = abs(float(p.get("size") or 0))
            if qty > 0:
                entry = float(p.get("entryPrice") or 0.0)
                lev   = float(p.get("leverage") or 0.0)
                unr   = float(p.get("unrealizedPnl") or 0.0)
                return {"ok": True, "side": side, "qty": qty, "entry": entry, "unrealized": unr, "leverage": lev, "raw": p}
    except Exception as e:
        log_w(f"fetch_live_position error: {e}")
    return {"ok": False, "why": "no_open_position"}

def resume_open_position(exchange, symbol: str, state: dict) -> dict:
    if not RESUME_ON_RESTART:
        log_i("resume disabled"); return state

    live = fetch_live_position(exchange, symbol)
    if not live.get("ok"):
        log_i("no live position to resume"); return state

    ts = int(time.time())
    prev = load_state()
    if prev.get("ts") and (ts - int(prev["ts"])) > RESUME_LOOKBACK_SECS:
        log_w("found old local state — will override with exchange live snapshot")

    state.update({
        "in_position": True,
        "side": live["side"],
        "entry_price": live["entry"],
        "position_qty": live["qty"],
        "leverage": live.get("leverage") or state.get("leverage") or 10,
        "partial_taken": prev.get("partial_taken", False),
        "breakeven_armed": prev.get("breakeven_armed", False),
        "trail_active": prev.get("trail_active", False),
        "trail_tightened": prev.get("trail_tightened", False),
        "mode": prev.get("mode", "trend"),
        "gz_snapshot": prev.get("gz_snapshot", {}),
        "cv_snapshot": prev.get("cv_snapshot", {}),
        "opened_at": prev.get("opened_at", ts),
    })
    save_state(state)
    log_g(f"RESUME: {state['side']} qty={state['position_qty']} @ {state['entry_price']:.6f} lev={state['leverage']}x")
    return state

# =================== LOGGING SETUP ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    log_i("log rotation ready")

setup_file_logging()

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q = _round_amt(q)
    if q<=0: log_w(f"qty invalid after normalize → {q}")
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

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
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
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
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# ========= Professional logging helpers =========
def fmt_walls(walls):
    return ", ".join([f"{p:.6f}@{q:.0f}" for p, q in walls]) if walls else "-"

# ========= Bookmap snapshot =========
def bookmap_snapshot(exchange, symbol, depth=BOOKMAP_DEPTH):
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]; asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return {"ok": False, "why": "empty"}
        b_sizes = np.array([b[1] for b in bids]); b_prices = np.array([b[0] for b in bids])
        a_sizes = np.array([a[1] for a in asks]); a_prices = np.array([a[0] for a in asks])
        b_idx = b_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        a_idx = a_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        buy_walls = [(float(b_prices[i]), float(b_sizes[i])) for i in b_idx]
        sell_walls = [(float(a_prices[i]), float(a_sizes[i])) for i in a_idx]
        imb = b_sizes.sum() / max(a_sizes.sum(), 1e-12)
        return {"ok": True, "buy_walls": buy_walls, "sell_walls": sell_walls, "imbalance": float(imb)}
    except Exception as e:
        return {"ok": False, "why": f"{e}"}

# ========= Volume flow / Delta & CVD =========
def compute_flow_metrics(df):
    try:
        if len(df) < max(30, FLOW_WINDOW+2):
            return {"ok": False, "why": "short_df"}
        close = df["close"].astype(float).copy()
        vol = df["volume"].astype(float).copy()
        up_mask = close.diff().fillna(0) > 0
        up_vol = (vol * up_mask).astype(float)
        dn_vol = (vol * (~up_mask)).astype(float)
        delta = up_vol - dn_vol
        cvd = delta.cumsum()
        cvd_ma = cvd.rolling(CVD_SMOOTH).mean()
        wnd = delta.tail(FLOW_WINDOW)
        mu = float(wnd.mean()); sd = float(wnd.std() or 1e-12)
        z = float((wnd.iloc[-1] - mu) / sd)
        trend = "up" if (cvd_ma.iloc[-1] - cvd_ma.iloc[-min(CVD_SMOOTH, len(cvd_ma))]) >= 0 else "down"
        return {"ok": True, "delta_last": float(delta.iloc[-1]), "delta_mean": mu, "delta_z": z,
                "cvd_last": float(cvd.iloc[-1]), "cvd_trend": trend, "spike": abs(z) >= FLOW_SPIKE_Z}
    except Exception as e:
        return {"ok": False, "why": str(e)}

# ========= Unified snapshot emitter =========
def emit_snapshots(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro_enhanced(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if cv["b"]>=cv["s"] else "sell")
        smart_entry = smart_entry_system(df, cv)

        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        if bm.get("ok"):
            imb_tag = "🟢" if bm["imbalance"]>=IMBALANCE_ALERT else ("🔴" if bm["imbalance"]<=1/IMBALANCE_ALERT else "⚖️")
            bm_note = f"Bookmap: {imb_tag} Imb={bm['imbalance']:.2f} | Buy[{fmt_walls(bm['buy_walls'])}] | Sell[{fmt_walls(bm['sell_walls'])}]"
        else:
            bm_note = f"Bookmap: N/A ({bm.get('why')})"

        if flow.get("ok"):
            dtag = "🟢Buy" if flow["delta_last"]>0 else ("🔴Sell" if flow["delta_last"]<0 else "⚖️Flat")
            spk = " ⚡Spike" if flow["spike"] else ""
            fl_note = f"Flow: {dtag} Δ={flow['delta_last']:.0f} z={flow['delta_z']:.2f}{spk} | CVD {'↗️' if flow['cvd_trend']=='up' else '↘️'} {flow['cvd_last']:.0f}"
        else:
            fl_note = f"Flow: N/A ({flow.get('why')})"

        side_hint = "BUY" if cv["b"]>=cv["s"] else "SELL"
        dash = (f"DASH → hint-{side_hint} | Council BUY({cv['b']},{cv['score_b']:.1f}) "
                f"SELL({cv['s']},{cv['score_s']:.1f}) | "
                f"RSI={cv['ind'].get('rsi',0):.1f} ADX={cv['ind'].get('adx',0):.1f} "
                f"DI={cv['ind'].get('di_spread',0):.1f}")

        strat_icon = "⚡" if mode["mode"]=="scalp" else "📈" if mode["mode"]=="trend" else "ℹ️"
        strat = f"Strategy: {strat_icon} {mode['mode'].upper()}"

        bal_note = f"Balance={bal:.2f}" if bal is not None else ""
        pnl_note = f"CompoundPnL={cpnl:.6f}" if cpnl is not None else ""
        wallet = (" | ".join(x for x in [bal_note, pnl_note] if x)) or ""

        gz_note = ""
        if gz and gz.get("ok"):
            gz_note = f" | 🟡 {gz['zone']['type']} s={gz['score']:.1f}"

        if LOG_ADDONS:
            print(f"🧱 {bm_note}", flush=True)
            print(f"📦 {fl_note}", flush=True)
            print(f"📊 {dash}{gz_note}", flush=True)
            print(f"{strat}{(' | ' + wallet) if wallet else ''}", flush=True)
            
            # عرض المؤشرات المتقدمة
            advanced = cv.get("advanced", {})
            super_trend = advanced.get("super_trend", {})
            volume_profile = advanced.get("volume_profile", {})
            market_structure = advanced.get("market_structure", {})
            
            print(f"🧠 ADVANCED INDICATORS:", flush=True)
            print(f"   SuperTrend: {super_trend.get('signal', 'neutral')} | Strength: {super_trend.get('strength', 0):.2f}", flush=True)
            print(f"   Volume POC: {volume_profile.get('poc', 0):.6f} | Strength: {volume_profile.get('volume_strength', 0):.2f}", flush=True)
            print(f"   Structure: {market_structure.get('signal', 'neutral')} | Break Strength: {market_structure.get('strength', 0):.2f}%", flush=True)
            print(f"   Smart Entry: {smart_entry['entry_signal'] or 'waiting'} | Confidence: {max(smart_entry['confidence_buy'], smart_entry['confidence_sell'])}/{smart_entry['min_signals']}", flush=True)

            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}", 
                  flush=True)
            
            print("✅ ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet, "smart_entry": smart_entry}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": "", "smart_entry": {}}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data):
    """تنفيذ قرار التداول مع التسجيل الواضح"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    gz_note = ""
    if gz_data and gz_data.get("ok"):
        gz_note = f" | 🟡 {gz_data['zone']['type']} s={gz_data['score']:.1f}"
    
    votes = council_data
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{gz_note}", flush=True)

    try:
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
    gz = snap.get("gz", {})  # إصلاح الخطأ
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
    if success:
        # تهيئة حالة جني الأرباح حسب النمط
        tp_levels_hit = [False] * len(management_config["tp_targets"])
        
        STATE.update({
            "open": True, 
            "side": "long" if side=="buy" else "short", 
            "entry": price,
            "qty": qty, 
            "pnl": 0.0, 
            "bars": 0, 
            "trail": None, 
            "breakeven": None,
            "tp1_done": False, 
            "tp_levels_hit": tp_levels_hit,
            "highest_profit_pct": 0.0, 
            "profit_targets_achieved": 0,
            "mode": mode,
            "management": management_config
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
            "tp_levels_hit": tp_levels_hit
        })
        
        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode}")
        return True
    
    return False

open_market = open_market_enhanced

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): 
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# =================== RANGE FILTER ===================
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty

def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rf_signal_live(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "tp_levels_hit": [],
    "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== WAIT FOR NEXT SIGNAL ===================
def _arm_wait_after_close(prev_side):
    global wait_for_next_signal_side
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")

def wait_gate_allow(df, info):
    if wait_for_next_signal_side is None: 
        return True, ""
    
    bar_ts = int(info.get("time") or 0)
    need = (wait_for_next_signal_side=="buy" and info.get("long")) or (wait_for_next_signal_side=="sell" and info.get("short"))
    
    if need:
        return True, ""
    return False, f"wait-for-next-RF({wait_for_next_signal_side})"

# =================== ORDERS ===================
def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                params = exchange_specific_params(side_to_close, is_close=True)
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                log_i(f"STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}")
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            log_w(f"strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}")
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    log_e(f"STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}")
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "tp_levels_hit": [],
        "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    save_state({"in_position": False, "position_qty": 0})
    
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== SMART EXIT GUARD ===================
def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    atr = ind.get('atr', 0.0)
    adx = ind.get('adx', 0.0)
    rsi = ind.get('rsi', 50.0)
    rsi_ma = ind.get('rsi_ma', 50.0)
    
    if len(df) >= 3:
        adx_slope = adx - ind.get('adx_prev', adx)
    else:
        adx_slope = 0.0

    wick_signal = False
    if len(df) > 0:
        c = df.iloc[-1]
        wick_up = float(c['high']) - max(float(c['close']), float(c['open']))
        wick_down = min(float(c['close']), float(c['open'])) - float(c['low'])
        wick_signal = (wick_up >= WICK_ATR_MULT * atr) if side == "long" else (wick_down >= WICK_ATR_MULT * atr)

    rsi_cross_down = (rsi < rsi_ma) if side == "long" else (rsi > rsi_ma)
    adx_falling = (adx_slope < 0)
    cvd_down = (flow and flow.get('ok') and flow.get('cvd_trend') == 'down')
    evx_spike = False
    
    bm_wall_close = False
    if bm and bm.get('ok'):
        if side == "long":
            sell_walls = bm.get('sell_walls', [])
            if sell_walls:
                best_ask = min([p for p, _ in sell_walls])
                bps = abs((best_ask - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)
        else:
            buy_walls = bm.get('buy_walls', [])
            if buy_walls:
                best_bid = max([p for p, _ in buy_walls])
                bps = abs((best_bid - now_price) / now_price) * 10000.0
                bm_wall_close = (bps <= BM_WALL_PROX_BPS)

    # إصلاح الخطأ - التحقق من وجود gz أولاً
    if gz and state.get('tp1_done') and gz.get("ok"):
        opp = (gz['zone']['type']=='golden_top' and side=='long') or (gz['zone']['type']=='golden_bottom' and side=='short')
        if opp and gz.get('score',0) >= GOLDEN_REVERSAL_SCORE:
            return {
                "action": "close", 
                "why": "golden_reversal",
                "log": f"🔴 CLOSE STRONG | golden reversal after TP1 | score={gz['score']:.1f}"
            }

    if pnl_pct > 0:
        if wick_signal or evx_spike or bm_wall_close or cvd_down:
            return {
                "action": "tighten", 
                "why": "exhaustion/flow/wall",
                "trail_mult": TRAIL_TIGHT_MULT,
                "log": f"🛡️ Tighten | wick={int(bool(wick_signal))} evx={int(bool(evx_spike))} wall={bm_wall_close} cvd_down={cvd_down}"
            }

    bearish_signals = [rsi_cross_down, adx_falling, cvd_down, evx_spike, bm_wall_close]
    bearish_count = sum(bearish_signals)
    
    if pnl_pct >= HARD_CLOSE_PNL_PCT and bearish_count >= 2:
        reasons = []
        if rsi_cross_down: reasons.append("rsi↓")
        if adx_falling: reasons.append("adx↓")
        if cvd_down: reasons.append("cvd↓")
        if evx_spike: reasons.append("evx")
        if bm_wall_close: reasons.append("wall")
        
        return {
            "action": "close", 
            "why": "hard_close_signal",
            "log": f"🔴 CLOSE STRONG | pnl={pnl_pct*100:.2f}% | {', '.join(reasons)}"
        }

    return {
        "action": "hold", 
        "why": "keep_riding", 
        "log": None
    }

# =================== ULTRA INTELLIGENT TRADE LOOP ===================
def trade_loop_enhanced():
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # تعريف gz بشكل آمن من البداية
            gz = {}
            
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث gz من البيانات الجديدة
            gz = snap.get("gz", {})
            
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            if STATE["open"]:
                manage_after_entry_enhanced(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap.get("bm", {}),
                    "flow": snap.get("flow", {}),
                    **info
                })
            
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            smart_entry = smart_entry_system(df, council_data)
            
            sig = smart_entry["entry_signal"]

            # إصلاح الخطأ - التحقق من وجود gz أولاً
            if gz and gz.get("ok") and gz.get("score", 0) >= GOLDEN_ENTRY_SCORE:
                if gz["zone"]["type"]=="golden_bottom" and smart_entry["confidence_buy"] >= 2:
                    sig = "buy"
                    log_i(f"🎯 GOLDEN ENTRY OVERRIDE: BUY | score={gz['score']:.1f}")
                elif gz["zone"]["type"]=="golden_top" and smart_entry["confidence_sell"] >= 2:
                    sig = "sell" 
                    log_i(f"🎯 GOLDEN ENTRY OVERRIDE: SELL | score={gz['score']:.1f}")
            
            if not STATE["open"] and sig and reason is None:
                allow_wait, wait_reason = wait_gate_allow(df, info)
                if not allow_wait:
                    reason = wait_reason
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        ok = open_market(sig, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                            log_i(f"🎯 SMART COUNCIL DECISION: {sig.upper()} | "
                                  f"الثقة: {smart_entry['confidence_buy'] if sig=='buy' else smart_entry['confidence_sell']}/{smart_entry['min_signals']} | "
                                  f"الإشارات: {', '.join(smart_entry['signals_buy'] if sig=='buy' else smart_entry['signals_sell'])}")
                            for log_msg in council_data.get("logs", []):
                                log_i(f"   - {log_msg}")
                    else:
                        reason = "qty<=0"
            
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

trade_loop = trade_loop_enhanced

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    if LOG_LEGACY:
        left_s = time_to_candle_close(df) if df is not None else 0
        print(colored("─"*100,"cyan"))
        print(colored(f"📊 {SYMBOL} {INTERVAL} • {EXCHANGE_NAME.upper()} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
        print(colored("─"*100,"cyan"))
        print("📈 INDICATORS & RF")
        print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}")
        print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
        print(f"   🎯 ENTRY: COUNCIL PRO ULTRA + SMART PROFIT MANAGEMENT  |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            mode_info = f" | MODE: {'⚡SCALP' if STATE.get('mode')=='scalp' else '📈TREND'}"
            print(f"   {lamp}{mode_info}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
        else:
            print("   ⚪ FLAT")
            if wait_for_next_signal_side:
                print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI Council PRO ULTRA Bot — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Smart Profit Management"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "COUNCIL_PRO_ULTRA", "wait_for_next_signal": wait_for_next_signal_side,
        "profit_management": {
            "scalp_tp": SCALP_TP_TARGETS,
            "trend_tp": TREND_TP_TARGETS,
            "scalp_fractions": SCALP_TP_FRACTIONS,
            "trend_fractions": TREND_TP_FRACTIONS
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "COUNCIL_PRO_ULTRA", "wait_for_next_signal": wait_for_next_signal_side,
        "trading_mode": STATE.get("mode", "unknown")
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-live-bot/keepalive"})
    log_i(f"KEEPALIVE every 50s → {url}")
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("SUI COUNCIL PRO ULTRA BOT - SMART PROFIT MANAGEMENT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"🎯 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • COUNCIL_PRO_ULTRA=ENABLED", "yellow"))
    print(colored(f"💰 SMART PROFIT: Scalp(1-shot at {SCALP_TP_TARGETS[0]}%) | Trend(3-TP: {TREND_TP_TARGETS})", "yellow"))
    print(colored(f"🏆 GOLDEN ENTRY: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"🧠 ADVANCED INDICATORS: SuperTrend + Volume Profile + Market Structure", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
