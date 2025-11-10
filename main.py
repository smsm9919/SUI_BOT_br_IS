# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (Multi-Exchange: BingX & Bybit)
• Council ULTIMATE with Footprint Analysis & Advanced Indicators
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Exit
• Dynamic TP ladder + ATR-trailing + Volume Momentum + Liquidity Analysis
• Professional Logging & Dashboard + Multi-Exchange Support
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
BOT_VERSION = f"SUI Council ULTIMATE v6.0 — {EXCHANGE_NAME.upper()} Multi-Exchange"
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

# =================== FOOTPRINT ANALYSIS SETTINGS ===================
FOOTPRINT_PERIOD = 20
FOOTPRINT_VOLUME_THRESHOLD = 2.0
DELTA_THRESHOLD = 1.5
ABSORPTION_RATIO = 0.65
EFFICIENCY_THRESHOLD = 0.85

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

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
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

SCALP_TP1 = 0.40
SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_TP1 = 1.20
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 6
COOLDOWN_SECS_AFTER_CLOSE = 60
ADX_GATE = 17

# ==== ULTIMATE COUNCIL SETTINGS ====
ULTIMATE_MIN_CONFIDENCE = 8.0
VOLUME_MOMENTUM_PERIOD = 20
STOCH_RSI_PERIOD = 14
DYNAMIC_PIVOT_PERIOD = 20
TREND_FAST_PERIOD = 10
TREND_SLOW_PERIOD = 20
TREND_SIGNAL_PERIOD = 9

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
    print(f"🎯 ULTIMATE COUNCIL: min_confidence={ULTIMATE_MIN_CONFIDENCE}", flush=True)
    print(f"📈 ADVANCED INDICATORS: Volume Momentum + Stochastic RSI + Dynamic Pivots", flush=True)
    print(f"👣 FOOTPRINT ANALYSIS: Volume Analysis + Absorption + Real Momentum", flush=True)
    print(f"⚡ RF SETTINGS: period={RF_PERIOD} | mult={RF_MULT} (SUI Optimized)", flush=True)
    
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

# =================== ADVANCED INDICATORS - ULTIMATE COUNCIL ===================
def enhanced_volume_momentum(df, period=20):
    """الزخم الحجمي المتقدم"""
    if len(df) < period + 5:
        return {"trend": "neutral", "strength": 0, "signal": 0}
    
    volume = df['volume'].astype(float)
    close = df['close'].astype(float)
    
    # متوسط حجم متحرك
    volume_ma = volume.rolling(period).mean()
    volume_ratio = volume / volume_ma.replace(0, 1)
    
    # زخم السعر مع الحجم
    price_change = close.pct_change(period)
    volume_weighted_momentum = price_change * volume_ratio
    
    current_momentum = volume_weighted_momentum.iloc[-1]
    momentum_trend = "bull" if current_momentum > 0.02 else ("bear" if current_momentum < -0.02 else "neutral")
    
    return {
        "trend": momentum_trend,
        "strength": abs(current_momentum) * 100,
        "signal": current_momentum
    }

def stochastic_rsi_enhanced(df, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """مؤشر RSI العشوائي المحسن"""
    if len(df) < max(rsi_period, stoch_period) + 10:
        return {"k": 50, "d": 50, "signal": "neutral", "oversold": False, "overbought": False}
    
    # حساب RSI
    rsi = compute_rsi(df['close'].astype(float), rsi_period)
    
    # حساب Stochastic للـ RSI
    rsi_low = rsi.rolling(stoch_period).min()
    rsi_high = rsi.rolling(stoch_period).max()
    
    stoch_k = 100 * (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, 100)
    stoch_k_smooth = stoch_k.rolling(k_period).mean()
    stoch_d = stoch_k_smooth.rolling(d_period).mean()
    
    current_k = stoch_k_smooth.iloc[-1]
    current_d = stoch_d.iloc[-1]
    
    # إشارات التداول
    signal = "neutral"
    if current_k < 20 and current_d < 20:
        signal = "bullish"
    elif current_k > 80 and current_d > 80:
        signal = "bearish"
    elif current_k > current_d and stoch_k_smooth.iloc[-2] <= stoch_d.iloc[-2]:
        signal = "bullish_cross"
    elif current_k < current_d and stoch_k_smooth.iloc[-2] >= stoch_d.iloc[-2]:
        signal = "bearish_cross"
    
    return {
        "k": current_k,
        "d": current_d,
        "signal": signal,
        "oversold": current_k < 20,
        "overbought": current_k > 80
    }

def dynamic_pivot_points(df, period=20):
    """نقاط محورية ديناميكية"""
    if len(df) < period:
        return {"pivot": 0, "r1": 0, "r2": 0, "s1": 0, "s2": 0, "bias": "neutral"}
    
    high = df['high'].astype(float).tail(period)
    low = df['low'].astype(float).tail(period)
    close = df['close'].astype(float).tail(period)
    
    pivot = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
    r1 = 2 * pivot - low.iloc[-1]
    r2 = pivot + (high.iloc[-1] - low.iloc[-1])
    s1 = 2 * pivot - high.iloc[-1]
    s2 = pivot - (high.iloc[-1] - low.iloc[-1])
    
    current_price = close.iloc[-1]
    
    # تحديد الانحياز
    if current_price > r1:
        bias = "strong_bullish"
    elif current_price > pivot:
        bias = "bullish"
    elif current_price < s1:
        bias = "strong_bearish"
    elif current_price < pivot:
        bias = "bearish"
    else:
        bias = "neutral"
    
    return {
        "pivot": pivot,
        "r1": r1, "r2": r2,
        "s1": s1, "s2": s2,
        "bias": bias
    }

def dynamic_trend_indicator(df, fast_period=10, slow_period=20, signal_period=9):
    """مؤشر الاتجاه الديناميكي"""
    if len(df) < slow_period + signal_period:
        return {"trend": "neutral", "momentum": 0, "signal": "hold"}
    
    close = df['close'].astype(float)
    
    # متوسطات متحركة متعددة
    ema_fast = close.ewm(span=fast_period).mean()
    ema_slow = close.ewm(span=slow_period).mean()
    ema_signal = ema_fast.ewm(span=signal_period).mean()
    
    # تقاطعات الاتجاه
    fast_above_slow = ema_fast.iloc[-1] > ema_slow.iloc[-1]
    fast_above_signal = ema_fast.iloc[-1] > ema_signal.iloc[-1]
    
    # زخم الاتجاه
    momentum = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100
    
    # تحديد الاتجاه
    if fast_above_slow and fast_above_signal and momentum > 0.1:
        trend = "strong_bull"
    elif fast_above_slow and momentum > 0:
        trend = "bull"
    elif not fast_above_slow and not fast_above_signal and momentum < -0.1:
        trend = "strong_bear"
    elif not fast_above_slow and momentum < 0:
        trend = "bear"
    else:
        trend = "neutral"
    
    # إشارة التداول
    signal = "hold"
    if trend == "strong_bull" and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
        signal = "strong_buy"
    elif trend == "bull" and ema_fast.iloc[-2] <= ema_signal.iloc[-2]:
        signal = "buy"
    elif trend == "strong_bear" and ema_fast.iloc[-2] >= ema_slow.iloc[-2]:
        signal = "strong_sell"
    elif trend == "bear" and ema_fast.iloc[-2] >= ema_signal.iloc[-2]:
        signal = "sell"
    
    return {
        "trend": trend,
        "momentum": momentum,
        "signal": signal,
        "ema_fast": ema_fast.iloc[-1],
        "ema_slow": ema_slow.iloc[-1]
    }

# =================== ADVANCED FOOTPRINT ANALYSIS ===================
def advanced_footprint_analysis(df, current_price):
    """
    تحليل بصمة السوق المتقدم لاكتشاف:
    - الامتصاص (Absorption)
    - الاندفاع الحقيقي (Real Momentum)
    - نقاط التوقف (Stops)
    - السيولة المخفية (Hidden Liquidity)
    """
    if len(df) < FOOTPRINT_PERIOD + 5:
        return {"ok": False, "reason": "لا توجد بيانات كافية"}
    
    try:
        # تحليل الحجم والسعر المتقدم
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)
        open_price = df['open'].astype(float)
        
        # المتوسطات الحجمية
        volume_ma = volume.rolling(FOOTPRINT_PERIOD).mean()
        volume_ratio = volume / volume_ma.replace(0, 1)
        
        # حساب دلتا الحجم (الفرق بين الشراء والبيع)
        up_volume = volume.where(close > open_price, 0)
        down_volume = volume.where(close < open_price, 0)
        volume_delta = (up_volume - down_volume).fillna(0)
        
        # كفاءة الحركة (Efficiency)
        body_size = abs(close - open_price)
        total_range = high - low
        efficiency = body_size / total_range.replace(0, 1)
        
        # تحليل الشمعة الحالية
        current_candle = {
            'high': float(high.iloc[-1]),
            'low': float(low.iloc[-1]),
            'close': float(close.iloc[-1]),
            'open': float(open_price.iloc[-1]),
            'volume': float(volume.iloc[-1]),
            'volume_ratio': float(volume_ratio.iloc[-1]),
            'delta': float(volume_delta.iloc[-1]),
            'efficiency': float(efficiency.iloc[-1])
        }
        
        # تحليل الامتصاص
        absorption_bullish = False
        absorption_bearish = False
        
        # امتصاص صاعد: حجم عالي + كفاءة منخفضة + دلتا إيجابية
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] < 0.4 and
            current_candle['delta'] > 0):
            absorption_bullish = True
        
        # امتصاص هابط: حجم عالي + كفاءة منخفضة + دلتا سلبية
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] < 0.4 and
            current_candle['delta'] < 0):
            absorption_bearish = True
        
        # اندفاع حقيقي
        real_momentum_bullish = False
        real_momentum_bearish = False
        
        # اندفاع صاعد حقيقي: حجم عالي + كفاءة عالية + دلتا إيجابية
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] > EFFICIENCY_THRESHOLD and
            current_candle['delta'] > DELTA_THRESHOLD):
            real_momentum_bullish = True
        
        # اندفاع هابط حقيقي: حجم عالي + كفاءة عالية + دلتا سلبية
        if (current_candle['volume_ratio'] >= FOOTPRINT_VOLUME_THRESHOLD and
            current_candle['efficiency'] > EFFICIENCY_THRESHOLD and
            current_candle['delta'] < -DELTA_THRESHOLD):
            real_momentum_bearish = True
        
        # نقاط التوقف (Stop Hunts)
        stop_hunt_bullish = False
        stop_hunt_bearish = False
        
        # صيد توقف صاعد: حركة سريعة هابطة ثم ارتداد سريع
        if len(df) >= 3:
            prev_low = float(low.iloc[-2])
            prev_high = float(high.iloc[-2])
            current_low = current_candle['low']
            current_high = current_candle['high']
            
            # صيد توقف هابط: اختراق قاع سابق ثم ارتداد
            if current_low < prev_low and current_candle['close'] > prev_low:
                stop_hunt_bullish = True
            
            # صيد توقف صاعد: اختراق قمة سابقة ثم انهيار
            if current_high > prev_high and current_candle['close'] < prev_high:
                stop_hunt_bearish = True
        
        # تحليل تجمعات السيولة (Liquidity Pools)
        liquidity_analysis = analyze_liquidity_pools(df, current_price)
        
        # حساب قوة الإشارة
        footprint_score_bull = 0.0
        footprint_score_bear = 0.0
        reasons = []
        
        if absorption_bullish:
            footprint_score_bull += 2.5
            reasons.append("امتصاص صاعد قوي")
        
        if absorption_bearish:
            footprint_score_bear += 2.5
            reasons.append("امتصاص هابط قوي")
        
        if real_momentum_bullish:
            footprint_score_bull += 3.0
            reasons.append("اندفاع صاعد حقيقي")
        
        if real_momentum_bearish:
            footprint_score_bear += 3.0
            reasons.append("اندفاع هابط حقيقي")
        
        if stop_hunt_bullish:
            footprint_score_bull += 2.0
            reasons.append("صيد توقف صاعد")
        
        if stop_hunt_bearish:
            footprint_score_bear += 2.0
            reasons.append("صيد توقف هابط")
        
        # إضافة تحليل السيولة
        if liquidity_analysis.get('buy_liquidity_above'):
            footprint_score_bull += 1.5
            reasons.append("سيولة شراء فوق السعر")
        
        if liquidity_analysis.get('sell_liquidity_below'):
            footprint_score_bear += 1.5
            reasons.append("سيولة بيع تحت السعر")
        
        return {
            "ok": True,
            "absorption_bullish": absorption_bullish,
            "absorption_bearish": absorption_bearish,
            "real_momentum_bullish": real_momentum_bullish,
            "real_momentum_bearish": real_momentum_bearish,
            "stop_hunt_bullish": stop_hunt_bullish,
            "stop_hunt_bearish": stop_hunt_bearish,
            "footprint_score_bull": footprint_score_bull,
            "footprint_score_bear": footprint_score_bear,
            "current_candle": current_candle,
            "liquidity_analysis": liquidity_analysis,
            "reasons": reasons
        }
        
    except Exception as e:
        return {"ok": False, "reason": f"خطأ في التحليل: {str(e)}"}

def analyze_liquidity_pools(df, current_price):
    """تحليل تجمعات السيولة المخفية"""
    if len(df) < 50:
        return {}
    
    try:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # البحث عن مناطق السيولة (نقاط الدعم والمقاومة)
        lookback = min(100, len(df))
        recent_highs = high.tail(lookback)
        recent_lows = low.tail(lookback)
        
        # تحديد المستويات الرئيسية
        resistance_levels = find_significant_highs(recent_highs)
        support_levels = find_significant_lows(recent_lows)
        
        # تحليل القرب من مستويات السيولة
        buy_liquidity_above = False
        sell_liquidity_below = False
        
        for level in resistance_levels:
            if abs(current_price - level) / current_price <= 0.02:  # ضمن 2%
                sell_liquidity_below = True
                break
        
        for level in support_levels:
            if abs(current_price - level) / current_price <= 0.02:  # ضمن 2%
                buy_liquidity_above = True
                break
        
        return {
            "resistance_levels": resistance_levels[-3:],  # آخر 3 مستويات مقاومة
            "support_levels": support_levels[-3:],        # آخر 3 مستويات دعم
            "buy_liquidity_above": buy_liquidity_above,
            "sell_liquidity_below": sell_liquidity_below
        }
        
    except Exception as e:
        return {}

def find_significant_highs(series, window=5):
    """إيجاد القمم الهامة"""
    highs = []
    for i in range(window, len(series) - window):
        if (series.iloc[i] == series.iloc[i-window:i+window].max() and 
            series.iloc[i] > series.iloc[i-1] and 
            series.iloc[i] > series.iloc[i+1]):
            highs.append(series.iloc[i])
    return highs

def find_significant_lows(series, window=5):
    """إيجاد القيعان الهامة"""
    lows = []
    for i in range(window, len(series) - window):
        if (series.iloc[i] == series.iloc[i-window:i+window].min() and 
            series.iloc[i] < series.iloc[i-1] and 
            series.iloc[i] < series.iloc[i+1]):
            lows.append(series.iloc[i])
    return lows

# =================== SMART GOLDEN ZONE DETECTION ===================
def _ema_gz(series, n):
    """المتوسط المتحرك الأسي للمنطقة الذهبية"""
    return series.ewm(span=n, adjust=False).mean()

def _rsi_fallback_gz(close, n=14):
    """RSI بديل محسّن"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def _body_wicks_gz(h, l, o, c):
    """حساب الجسم والفتائل بدقة"""
    rng = max(1e-9, h - l)
    body = abs(c - o) / rng
    up_wick = (h - max(c, o)) / rng
    low_wick = (min(c, o) - l) / rng
    return body, up_wick, low_wick

def _displacement_gz(closes):
    """قياس اندفاع السعر"""
    if len(closes) < 22:
        return 0.0
    recent_std = closes.tail(20).std()
    return abs(closes.iloc[-1] - closes.iloc[-2]) / max(recent_std, 1e-9)

def _last_impulse_gz(df):
    """اكتشاف آخر موجة دافعة بدقة"""
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    
    # البحث عن القمة والقاع في آخر 120 شمعة
    lookback = min(120, len(df))
    recent_highs = h.tail(lookback)
    recent_lows = l.tail(lookback)
    
    hh_idx = recent_highs.idxmax()
    ll_idx = recent_lows.idxmin()
    
    hh = recent_highs.max()
    ll = recent_lows.min()
    
    # تحديد اتجاه الدافع
    if hh_idx < ll_idx:  # قمة ثم قاع => دافع هابط
        return ("down", hh_idx, ll_idx, hh, ll)
    else:  # قاع ثم قمة => دافع صاعد
        return ("up", ll_idx, hh_idx, ll, hh)

def golden_zone_check(df, ind=None, side_hint=None):
    """اكتشاف المناطق الذهبية بدقة محسنة"""
    if len(df) < 60:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        # استخراج البيانات
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        v = df['volume'].astype(float)
        
        # اكتشاف الدافع الأخير
        impulse_data = _last_impulse_gz(df)
        if not impulse_data:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": ["no_clear_impulse"]}
            
        side, idx1, idx2, p1, p2 = impulse_data
        
        # حساب فيبوناتشي بناءً على اتجاه الدافع
        if side == "down":
            # دافع هابط: التصحيح الصاعد بين 0.618-0.786 من الهبوط
            swing_hi, swing_lo = p1, p2
            f618 = swing_lo + FIB_LOW * (swing_hi - swing_lo)
            f786 = swing_lo + FIB_HIGH * (swing_hi - swing_lo)
            zone_type = "golden_bottom"
        else:
            # دافع صاعد: التصحيح الهابط بين 0.618-0.786 من الصعود
            swing_lo, swing_hi = p1, p2
            f618 = swing_hi - FIB_HIGH * (swing_hi - swing_lo)
            f786 = swing_hi - FIB_LOW * (swing_hi - swing_lo)
            zone_type = "golden_top"
        
        last_close = float(c.iloc[-1])
        in_zone = (f618 <= last_close <= f786) if side == "down" else (f786 <= last_close <= f618)
        
        if not in_zone:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"price_not_in_zone {last_close:.6f} vs [{f618:.6f},{f786:.6f}]"]}
        
        # الشروط المساعدة
        current_high = float(h.iloc[-1])
        current_low = float(l.iloc[-1])
        current_open = float(o.iloc[-1])
        
        body, up_wick, low_wick = _body_wicks_gz(current_high, current_low, current_open, last_close)
        
        # حجم التداول
        vol_ma = v.rolling(VOL_MA_LEN).mean().iloc[-1]
        vol_ok = float(v.iloc[-1]) >= vol_ma * 0.9  # تخفيف الشرط قليلاً
        
        # RSI
        rsi_series = _rsi_fallback_gz(c, RSI_LEN_GZ)
        rsi_ma_series = _ema_gz(rsi_series, RSI_MA_LEN_GZ)
        rsi_last = float(rsi_series.iloc[-1])
        rsi_ma_last = float(rsi_ma_series.iloc[-1])
        
        # ADX من المؤشرات المحسوبة مسبقاً
        adx = ind.get('adx', 0) if ind else 0
        
        # اندفاع السعر
        disp = _displacement_gz(c)
        
        # فتيلة مناسبة حسب الاتجاه
        if side == "down":  # نبحث عن فتيلة سفلية للشراء
            wick_ok = low_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last > rsi_ma_last and rsi_last < 70
            candle_bullish = last_close > current_open
        else:  # نبحث عن فتيلة علوية للبيع
            wick_ok = up_wick >= MIN_WICK_PCT
            rsi_ok = rsi_last < rsi_ma_last and rsi_last > 30
            candle_bullish = last_close < current_open
        
        # حساب النقاط
        score = 0.0
        reasons = []
        
        # الشروط الأساسية
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
        
        # شرط المنطقة
        score += 2.0
        reasons.append("in_zone")
        
        # النتيجة النهائية
        ok = (score >= GZ_MIN_SCORE and in_zone and adx >= GZ_REQ_ADX)
        
        # تشخيص تفصيلي
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
    """تحديد نمط التداول: SCALP أم TREND"""
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

# =================== ULTIMATE COUNCIL WITH FOOTPRINT ===================
def ultimate_council_voting_with_footprint(df):
    """مجلس الإدارة المتقدم مع تحليل بصمة السوق"""
    try:
        # المؤشرات الأساسية
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)
        candles = compute_candles(df)
        current_price = float(df['close'].iloc[-1])
        
        # المؤشرات المتقدمة الجديدة
        volume_momentum = enhanced_volume_momentum(df)
        stoch_rsi = stochastic_rsi_enhanced(df)
        pivots = dynamic_pivot_points(df)
        trend_indicator = dynamic_trend_indicator(df)
        footprint = advanced_footprint_analysis(df, current_price)
        
        # نظام التصويت المتقدم مع Footprint
        votes_buy = 0
        votes_sell = 0
        confidence_buy = 0.0
        confidence_sell = 0.0
        detailed_logs = []
        
        # 1. تحليل Footprint (أعلى وزن)
        if footprint.get("ok"):
            fp_bull = footprint.get("footprint_score_bull", 0)
            fp_bear = footprint.get("footprint_score_bear", 0)
            
            if fp_bull > 0:
                votes_buy += 4
                confidence_buy += min(3.0, fp_bull)
                for reason in footprint.get("reasons", []):
                    if "صاعد" in reason:
                        detailed_logs.append(f"👣 Footprint: {reason}")
            
            if fp_bear > 0:
                votes_sell += 4
                confidence_sell += min(3.0, fp_bear)
                for reason in footprint.get("reasons", []):
                    if "هابط" in reason:
                        detailed_logs.append(f"👣 Footprint: {reason}")
        
        # 2. تحليل الاتجاه الأساسي
        if trend_indicator["signal"] in ["strong_buy", "buy"]:
            votes_buy += 3
            confidence_buy += 2.0
            detailed_logs.append(f"🎯 الاتجاه: {trend_indicator['signal']}")
        
        if trend_indicator["signal"] in ["strong_sell", "sell"]:
            votes_sell += 3
            confidence_sell += 2.0
            detailed_logs.append(f"🎯 الاتجاه: {trend_indicator['signal']}")
        
        # 3. الزخم الحجمي
        if volume_momentum["trend"] == "bull" and volume_momentum["strength"] > 1.5:
            votes_buy += 2
            confidence_buy += 1.5
            detailed_logs.append(f"📊 زخم حجمي صاعد")
        
        if volume_momentum["trend"] == "bear" and volume_momentum["strength"] > 1.5:
            votes_sell += 2
            confidence_sell += 1.5
            detailed_logs.append(f"📊 زخم حجمي هابط")
        
        # 4. RSI العشوائي
        if stoch_rsi["signal"] in ["bullish", "bullish_cross"] and not stoch_rsi["overbought"]:
            votes_buy += 2
            confidence_buy += 1.2
            detailed_logs.append(f"📈 Stoch RSI إيجابي")
        
        if stoch_rsi["signal"] in ["bearish", "bearish_cross"] and not stoch_rsi["oversold"]:
            votes_sell += 2
            confidence_sell += 1.2
            detailed_logs.append(f"📈 Stoch RSI سلبي")
        
        # 5. المناطق الذهبية
        if gz and gz.get("ok"):
            if gz['zone']['type'] == 'golden_bottom' and gz['score'] >= 7.0:
                votes_buy += 3
                confidence_buy += 2.5
                detailed_logs.append(f"🏆 منطقة ذهبية للشراء")
            elif gz['zone']['type'] == 'golden_top' and gz['score'] >= 7.0:
                votes_sell += 3
                confidence_sell += 2.5
                detailed_logs.append(f"🏆 منطقة ذهبية للبيع")
        
        # 6. شموع اليابان
        if candles["score_buy"] > 2.0:
            votes_buy += 2
            confidence_buy += min(2.0, candles["score_buy"])
            detailed_logs.append(f"🕯️ إشارة شموع شراء")
        
        if candles["score_sell"] > 2.0:
            votes_sell += 2
            confidence_sell += min(2.0, candles["score_sell"])
            detailed_logs.append(f"🕯️ إشارة شموع بيع")
        
        # 7. ADX والاتجاه
        if ind.get('adx', 0) > 25:
            if ind.get('plus_di', 0) > ind.get('minus_di', 0):
                votes_buy += 2
                confidence_buy += 1.5
                detailed_logs.append(f"📊 اتجاه صاعد قوي")
            else:
                votes_sell += 2
                confidence_sell += 1.5
                detailed_logs.append(f"📊 اتجاه هابط قوي")
        
        # 8. RSI مع المتوسط
        if rsi_ctx["cross"] == "bull" and rsi_ctx["rsi"] < 65:
            votes_buy += 1
            confidence_buy += 1.0
            detailed_logs.append(f"📈 RSI إيجابي")
        
        if rsi_ctx["cross"] == "bear" and rsi_ctx["rsi"] > 35:
            votes_sell += 1
            confidence_sell += 1.0
            detailed_logs.append(f"📈 RSI سلبي")
        
        # 9. نقاط المحورية
        if pivots["bias"] in ["bullish", "strong_bullish"]:
            votes_buy += 1
            confidence_buy += 0.8
        
        if pivots["bias"] in ["bearish", "strong_bearish"]:
            votes_sell += 1
            confidence_sell += 0.8
        
        # تصفية الإشارات المتضاربة مع أولوية Footprint
        if votes_buy > 0 and votes_sell > 0:
            # إذا كان Footprint يعطي إشارة قوية، نرجحها
            if footprint.get("ok") and footprint.get("footprint_score_bull", 0) > 3:
                votes_sell = 0
                confidence_sell = 0
                detailed_logs.append("🔄 ترجيح إشارة الشراء (Footprint قوي)")
            elif footprint.get("ok") and footprint.get("footprint_score_bear", 0) > 3:
                votes_buy = 0
                confidence_buy = 0
                detailed_logs.append("🔄 ترجيح إشارة البيع (Footprint قوي)")
            elif confidence_buy > confidence_sell:
                votes_sell = 0
                confidence_sell = 0
                detailed_logs.append("🔄 ترجيح إشارة الشراء (ثقة أعلى)")
            else:
                votes_buy = 0
                confidence_buy = 0
                detailed_logs.append("🔄 ترجيح إشارة البيع (ثقة أعلى)")
        
        # تحديث المؤشرات بالإضافات الجديدة
        ind.update({
            "volume_momentum": volume_momentum,
            "stoch_rsi": stoch_rsi,
            "pivots": pivots,
            "trend_indicator": trend_indicator,
            "footprint_analysis": footprint,
            "ultimate_votes_buy": votes_buy,
            "ultimate_votes_sell": votes_sell,
            "ultimate_confidence_buy": confidence_buy,
            "ultimate_confidence_sell": confidence_sell
        })
        
        return {
            "b": votes_buy, "s": votes_sell,
            "score_b": confidence_buy, "score_s": confidence_sell,
            "logs": detailed_logs, 
            "ind": ind, 
            "gz": gz, 
            "candles": candles,
            "advanced_indicators": {
                "volume_momentum": volume_momentum,
                "stoch_rsi": stoch_rsi,
                "pivots": pivots,
                "trend_indicator": trend_indicator,
                "footprint": footprint
            }
        }
        
    except Exception as e:
        log_w(f"ultimate_council_voting_with_footprint error: {e}")
        return {"b":0, "s":0, "score_b":0.0, "score_s":0.0, "logs":[], "ind":{}, "gz":None, "candles":{}}

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
def emit_snapshots_with_footprint(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    """لوحة تحكم محسنة مع عرض Footprint"""
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = ultimate_council_voting_with_footprint(df)  # استخدام المجلس مع Footprint
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"]["adx"]}, "buy" if cv["b"]>=cv["s"] else "sell")
        current_price = float(df['close'].iloc[-1])
        footprint = advanced_footprint_analysis(df, current_price)

        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        # عرض Footprint في الداشبورد
        footprint_note = ""
        if footprint.get("ok"):
            fp_bull = footprint.get("footprint_score_bull", 0)
            fp_bear = footprint.get("footprint_score_bear", 0)
            footprint_note = f" | 👣 FP({fp_bull:.1f}/{fp_bear:.1f})"
            
            # عرض إشارات Footprint الهامة
            if footprint.get("absorption_bullish"):
                print(f"🟢 FOOTPRINT: امتصاص صاعد قوي | حجم: {footprint['current_candle']['volume_ratio']:.1f}x", flush=True)
            if footprint.get("absorption_bearish"):
                print(f"🔴 FOOTPRINT: امتصاص هابط قوي | حجم: {footprint['current_candle']['volume_ratio']:.1f}x", flush=True)
            if footprint.get("real_momentum_bullish"):
                print(f"🚀 FOOTPRINT: اندفاع صاعد حقيقي | دلتا: {footprint['current_candle']['delta']:+.0f}", flush=True)
            if footprint.get("real_momentum_bearish"):
                print(f"💥 FOOTPRINT: اندفاع هابط حقيقي | دلتا: {footprint['current_candle']['delta']:+.0f}", flush=True)

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
                f"DI={cv['ind'].get('di_spread',0):.1f}{footprint_note}")

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
            
            # تحديث السناك شوت النهائي
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP ULTIMATE | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{footprint_note}{gz_note}", 
                  flush=True)
            
            print("✅ ULTIMATE ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "footprint": footprint, "wallet": wallet}
    except Exception as e:
        print(f"🟨 Ultimate AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "footprint": {}, "wallet": ""}

# =================== EXECUTION MANAGER ===================
def execute_trade_decision_with_footprint(side, price, qty, mode, council_data, gz_data):
    """تنفيذ متقدم مع تحليل Footprint"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f} | mode={mode}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    # تحليل Footprint للإدخال
    footprint = council_data.get("advanced_indicators", {}).get("footprint", {})
    footprint_note = ""
    
    if footprint.get("ok"):
        fp_score = footprint.get("footprint_score_bull", 0) if side == "buy" else footprint.get("footprint_score_bear", 0)
        footprint_note = f" | 👣 Footprint score={fp_score:.1f}"
        
        # تسجيل تفاصيل Footprint
        for reason in footprint.get("reasons", []):
            if ("صاعد" in reason and side == "buy") or ("هابط" in reason and side == "sell"):
                log_i(f"🎯 Footprint Signal: {reason}")
    
    gz_note = ""
    if gz_data and gz_data.get("ok"):
        gz_note = f" | 🟡 {gz_data['zone']['type']} s={gz_data['score']:.1f}"
    
    votes = council_data
    print(f"🎯 EXECUTE ULTIMATE: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"mode={mode} | votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{footprint_note}{gz_note}", flush=True)

    try:
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ EXECUTED ULTIMATE: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        # تسجيل تفاصيل Footprint في السجل
        if footprint.get("ok"):
            logging.info(f"FOOTPRINT_ENTRY: {side} | score={fp_score:.1f} | reasons={footprint.get('reasons', [])}")
        
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def setup_trade_management(mode):
    """تهيئة إدارة الصفقة حسب النمط"""
    if mode == "scalp":
        return {
            "tp1_pct": SCALP_TP1 / 100.0,
            "be_activate_pct": SCALP_BE_AFTER / 100.0,
            "trail_activate_pct": 0.8 / 100.0,
            "atr_trail_mult": SCALP_ATR_MULT,
            "close_aggression": "high"
        }
    else:
        return {
            "tp1_pct": TREND_TP1 / 100.0,
            "be_activate_pct": TREND_BE_AFTER / 100.0,
            "trail_activate_pct": 1.2 / 100.0,
            "atr_trail_mult": TREND_ATR_MULT,
            "close_aggression": "medium"
        }

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots_with_footprint(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision_with_footprint(side, price, qty, mode, votes, gz)
    
    if success:
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
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
}
compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== WAIT FOR NEXT SIGNAL ===================
def _arm_wait_after_close(prev_side):
    """تفعيل انتظار الإشارة التالية بعد الإغلاق"""
    global wait_for_next_signal_side
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    log_i(f"🛑 WAIT FOR NEXT SIGNAL: {wait_for_next_signal_side}")

def wait_gate_allow(df, info):
    """التحقق من بوابة الانتظار"""
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
    """إعادة تعيين الحالة بعد الإغلاق"""
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ADVANCED TRADE MANAGEMENT ===================
def advanced_trade_management(df, state, current_price):
    """إدارة متقدمة للصفقات مع جني الأرباح الديناميكي"""
    if not state["open"] or state["qty"] <= 0:
        return {"action": "hold", "reason": "لا توجد صفقة مفتوحة"}
    
    entry = state["entry"]
    side = state["side"]
    unrealized_pnl_pct = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
    
    # حساب التقلب الحالي
    atr = compute_indicators(df).get('atr', 0.001)
    volatility_ratio = atr / current_price * 100
    
    # تحديد أهداف الربح الديناميكية بناءً على التقلب
    if volatility_ratio > 2.0:  # سوق متقلب
        tp_levels = [0.8, 1.5, 2.5]  # أهداف أعلى
        tp_fractions = [0.3, 0.4, 0.3]
    elif volatility_ratio < 0.5:  # سوق هادئ
        tp_levels = [0.4, 0.8, 1.2]  # أهداف أقل
        tp_fractions = [0.4, 0.3, 0.3]
    else:  # سوق عادي
        tp_levels = [0.6, 1.2, 2.0]
        tp_fractions = [0.3, 0.3, 0.4]
    
    # جني الأرباح عند المستويات
    achieved_tps = state.get("profit_targets_achieved", 0)
    
    if achieved_tps < len(tp_levels) and unrealized_pnl_pct >= tp_levels[achieved_tps]:
        close_fraction = tp_fractions[achieved_tps]
        return {
            "action": "partial_close",
            "reason": f"جني أرباح عند المستوى {achieved_tps + 1}",
            "close_fraction": close_fraction,
            "tp_level": tp_levels[achieved_tps]
        }
    
    # وقف الخسارة المتحرك المحسن
    if unrealized_pnl_pct > 1.0:  # بدء التريل بعد تحقيق ربح 1%
        if side == "long":
            new_trail = current_price - (atr * 1.5)
            if state.get("trail") is None or new_trail > state["trail"]:
                state["trail"] = new_trail
        else:
            new_trail = current_price + (atr * 1.5)
            if state.get("trail") is None or new_trail < state["trail"]:
                state["trail"] = new_trail
        
        # التحقق من التريل
        if state.get("trail"):
            if (side == "long" and current_price <= state["trail"]) or \
               (side == "short" and current_price >= state["trail"]):
                return {
                    "action": "close", 
                    "reason": "وقف خسارة متحرك",
                    "trail_price": state["trail"]
                }
    
    return {"action": "hold", "reason": "الاستمرار في الصفقة"}

# =================== ULTIMATE TRADE MANAGEMENT ===================
def manage_after_entry_ultimate(df, ind, info):
    """الإدارة النهائية للصفقات مع النظام المتقدم"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    current_price = info["price"]
    
    # الإدارة المتقدمة
    management_signal = advanced_trade_management(df, STATE, current_price)
    
    if management_signal["action"] == "partial_close":
        close_fraction = management_signal["close_fraction"]
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                    log_g(f"✅ جني أرباح: {close_fraction*100}% عند {management_signal['tp_level']:.2f}%")
                    STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    STATE["profit_targets_achieved"] += 1
                except Exception as e:
                    log_e(f"❌ فشل جني الأرباح: {e}")
            else:
                log_i(f"DRY_RUN: جني أرباح {close_qty:.4f}")
    
    elif management_signal["action"] == "close":
        log_w(f"🚨 إغلاق بالوقف المتحرك: {management_signal['reason']}")
        close_market_strict(f"advanced_trail_{management_signal['reason']}")
        return
    
    # الاستمرار في الإدارة الأساسية المحسنة
    manage_after_entry_enhanced(df, ind, info)

def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع خروج ذكي حسب النمط"""
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

    snap = emit_snapshots_with_footprint(ex, SYMBOL, df)
    gz = snap["gz"]
    
    exit_signal = smart_exit_guard_with_footprint(STATE, df, ind, snap["flow"], snap["bm"], 
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
    tp1_pct = management.get("tp1_pct", TP1_PCT_BASE/100.0)
    be_activate_pct = management.get("be_activate_pct", BREAKEVEN_AFTER/100.0)
    trail_activate_pct = management.get("trail_activate_pct", TRAIL_ACTIVATE_PCT/100.0)
    atr_trail_mult = management.get("atr_trail_mult", ATR_TRAIL_MULT)

    if not STATE.get("tp1_done") and pnl_pct/100 >= tp1_pct:
        close_fraction = TP1_CLOSE_FRAC
        close_qty = safe_qty(STATE["qty"] * close_fraction)
        if close_qty > 0:
            close_side = "sell" if STATE["side"] == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                    log_g(f"✅ TP1 HIT: closed {close_fraction*100}%")
                except Exception as e:
                    log_e(f"❌ TP1 close failed: {e}")
            STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
            STATE["tp1_done"] = True
            STATE["profit_targets_achieved"] += 1

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

# استبدال إدارة الصفقات بالنظام المتقدم
manage_after_entry = manage_after_entry_ultimate

def smart_exit_guard_with_footprint(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    """خروج ذكي مع تحليل Footprint المتقدم"""
    # التحليل الأساسي
    basic_exit = smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz)
    
    # إضافة تحليل Footprint للخروج
    footprint = ind.get('footprint_analysis', {})
    
    if footprint.get("ok"):
        # اكتشاف انعكاس مبكر عبر Footprint
        if side == "long" and footprint.get("absorption_bearish"):
            return {
                "action": "close", 
                "why": "footprint_absorption_bearish",
                "log": f"🔴 خروج Footprint | امتصاص هابط مثير للشك | pnl={pnl_pct*100:.2f}%"
            }
        
        if side == "short" and footprint.get("absorption_bullish"):
            return {
                "action": "close", 
                "why": "footprint_absorption_bullish",
                "log": f"🔴 خروج Footprint | امتصاص صاعد مثير للشك | pnl={pnl_pct*100:.2f}%"
            }
        
        # اكتشاف فقدان الزخم عبر Footprint
        current_candle = footprint.get("current_candle", {})
        if current_candle.get('volume_ratio', 0) < 0.5 and pnl_pct > 0.5:
            return {
                "action": "partial",
                "why": "footprint_low_volume",
                "qty_pct": 0.4,
                "log": f"🟡 خروج جزئي | حجم منخفض مع ربح | pnl={pnl_pct*100:.2f}%"
            }
    
    return basic_exit

def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    """يقرر: Partial / Tighten / Strict Close مع لوج واضح."""
    atr = ind.get('atr', 0.0)
    adx = ind.get('adx', 0.0)
    rsi = ind.get('rsi', 50.0)
    rsi_ma = ind.get('rsi_ma', 50.0)
    
    if len(df) >= 3:
        adx_slope = adx - ind.get('adx_prev', adx)
    else:
        adx_slope = 0.0

    # حساب الفتائل
    wick_signal = False
    if len(df) > 0:
        c = df.iloc[-1]
        wick_up = float(c['high']) - max(float(c['close']), float(c['open']))
        wick_down = min(float(c['close']), float(c['open'])) - float(c['low'])
        wick_signal = (wick_up >= WICK_ATR_MULT * atr) if side == "long" else (wick_down >= WICK_ATR_MULT * atr)

    rsi_cross_down = (rsi < rsi_ma) if side == "long" else (rsi > rsi_ma)
    adx_falling = (adx_slope < 0)
    cvd_down = (flow and flow.get('ok') and flow.get('cvd_trend') == 'down')
    evx_spike = False  # يمكن إضافة حساب EVX لاحقًا
    
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

    # --- Golden Reversal بعد TP1 ---
    if state.get('tp1_done') and (gz and gz.get('ok')):
        # إغلاق صارم لو تقاطع Golden عكس اتجاهي بعد TP1
        opp = (gz['zone']['type']=='golden_top' and side=='long') or (gz['zone']['type']=='golden_bottom' and side=='short')
        if opp and gz.get('score',0) >= GOLDEN_REVERSAL_SCORE:
            return {
                "action": "close", 
                "why": "golden_reversal",
                "log": f"🔴 CLOSE STRONG | golden reversal after TP1 | score={gz['score']:.1f}"
            }

    tp1_target = TP1_SCALP_PCT if mode == 'scalp' else TP1_TREND_PCT
    if pnl_pct >= tp1_target and not state.get('tp1_done'):
        qty_pct = 0.35 if mode == 'scalp' else 0.25
        return {
            "action": "partial", 
            "why": f"TP1 hit {tp1_target*100:.2f}%",
            "qty_pct": qty_pct,
            "log": f"💰 TP1 جزئي {tp1_target*100:.2f}% | pnl={pnl_pct*100:.2f}% | mode={mode}"
        }

    # --- Wick exhaustion + Tighten عند إجهاد/تدفق/جدار ---
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

# =================== ULTIMATE DECISION LOGGING ===================
def log_ultimate_decision(council_data, decision):
    """تسجيل متقدم لقرارات المجلس"""
    votes = council_data
    advanced = votes.get("advanced_indicators", {})
    
    print(f"\n🎯 قرار المجلس المتقدم:", flush=True)
    print(f"📊 الأصوات: شراء {votes['b']} | بيع {votes['s']}", flush=True)
    print(f"⭐ الثقة: شراء {votes['score_b']:.1f} | بيع {votes['score_s']:.1f}", flush=True)
    print(f"📈 القرار النهائي: {decision}", flush=True)
    
    # تفاصيل المؤشرات المتقدمة
    if advanced.get("trend_indicator"):
        ti = advanced["trend_indicator"]
        print(f"🔄 مؤشر الاتجاه: {ti['trend']} - {ti['signal']}", flush=True)
    
    if advanced.get("volume_momentum"):
        vm = advanced["volume_momentum"]
        print(f"📊 الزخم الحجمي: {vm['trend']} (قوة: {vm['strength']:.1f})", flush=True)
    
    if advanced.get("stoch_rsi"):
        sr = advanced["stoch_rsi"]
        print(f"📈 Stoch RSI: {sr['signal']} (K: {sr['k']:.1f}, D: {sr['d']:.1f})", flush=True)
    
    print("─" * 80, flush=True)

# =================== ULTIMATE TRADE LOOP ===================
def trade_loop_ultimate_with_footprint():
    """حلقة تداول نهائية مع Footprint Analysis"""
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # تحديث الـ Snapshots مع Footprint
            snap = emit_snapshots_with_footprint(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة مع Footprint
            if STATE["open"]:
                manage_after_entry_ultimate(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    "footprint": snap.get("footprint", {}),
                    **info
                })
            
            # قرار الدخول باستخدام المجلس المتقدم مع Footprint
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"انتشار مرتفع ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = ultimate_council_voting_with_footprint(df)
            gz = council_data.get("gz")
            footprint = council_data.get("advanced_indicators", {}).get("footprint", {})
            
            # قرار التداول النهائي مع أولوية Footprint
            decision = None
            min_confidence = ULTIMATE_MIN_CONFIDENCE
            
            # تخفيض عتبة الثقة إذا كان Footprint قوي
            if footprint.get("ok") and (footprint.get("footprint_score_bull", 0) > 3 or 
                                      footprint.get("footprint_score_bear", 0) > 3):
                min_confidence = 6.0  # تخفيض العتبة لصالح Footprint القوي
            
            if (council_data["score_b"] >= min_confidence and 
                council_data["score_b"] > council_data["score_s"] + 2.0):
                decision = "BUY"
            elif (council_data["score_s"] >= min_confidence and 
                  council_data["score_s"] > council_data["score_b"] + 2.0):
                decision = "SELL"
            
            # تسجيل القرار المتقدم مع Footprint
            if decision:
                log_ultimate_decision(council_data, decision)
                
                # تسجيل تفاصيل Footprint
                if footprint.get("ok"):
                    fp_score = (footprint.get("footprint_score_bull", 0) if decision == "BUY" 
                               else footprint.get("footprint_score_bear", 0))
                    print(f"🎯 FOOTPRINT CONFIRMATION: {decision} | score={fp_score:.1f}", flush=True)
            
            if not STATE["open"] and decision and reason is None:
                # التحقق من سياسة الانتظار
                allow_wait, wait_reason = wait_gate_allow(df, info)
                if not allow_wait:
                    reason = wait_reason
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        side = "buy" if decision == "BUY" else "sell"
                        ok = open_market(side, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                    else:
                        reason = "الكمية غير صالحة"
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"خطأ في الحلقة: {e}\n{traceback.format_exc()}")
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# استبدال حلقة التداول النهائية
trade_loop = trade_loop_ultimate_with_footprint

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI Council ULTIMATE Bot — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Multi-Exchange"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "ULTIMATE_COUNCIL_WITH_FOOTPRINT", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "ULTIMATE_COUNCIL_WITH_FOOTPRINT", "wait_for_next_signal": wait_for_next_signal_side
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

# =================== INTEGRATION ===================
# استبدال الدوال الأساسية بالإصدارات المحسنة
ultimate_council_voting = ultimate_council_voting_with_footprint
smart_exit_guard = smart_exit_guard_with_footprint
execute_trade_decision = execute_trade_decision_with_footprint
emit_snapshots = emit_snapshots_with_footprint

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("SUI COUNCIL ULTIMATE BOT - FOOTPRINT ANALYSIS")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"🎯 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • ULTIMATE_COUNCIL=ENABLED", "yellow"))
    print(colored(f"🏆 ULTIMATE MIN CONFIDENCE: {ULTIMATE_MIN_CONFIDENCE}", "yellow"))
    print(colored(f"📊 ADVANCED INDICATORS: Volume Momentum + Stochastic RSI + Dynamic Pivots", "yellow"))
    print(colored(f"👣 FOOTPRINT ANALYSIS: Volume Analysis + Absorption + Real Momentum + Stop Hunts", "yellow"))
    print(colored(f"📈 DYNAMIC TREND INDICATOR: Fast={TREND_FAST_PERIOD} Slow={TREND_SLOW_PERIOD}", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
