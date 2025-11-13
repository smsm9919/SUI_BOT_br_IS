# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (Multi-Exchange: BingX & Bybit)
• Council ULTIMATE with Smart Money Concepts & Advanced Indicators
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Exit
• Dynamic TP ladder + ATR-trailing + Volume Momentum + Liquidity Analysis
• Professional Logging & Dashboard + Multi-Exchange Support
• ACTIVE COUNCIL SYSTEM - Enhanced Profit Maximization
• SUPER INTELLIGENT TRADE MANAGEMENT - Smart Profit Taking
• PROFESSIONAL TRADE CLASSIFICATION - Premium/Strong/Normal/Scalp
• ENHANCED PROTECTION SYSTEM - Adaptive Risk Management
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

# =================== HELPER FUNCTIONS ===================
def last_val(x):
    """يرجع آخر قيمة من Series أو ndarray أو list بأمان كـ float."""
    try:
        if hasattr(x, "iloc"):   # pandas Series
            return float(x.iloc[-1])
        elif hasattr(x, "__len__") and len(x) > 0:
            return float(x[-1])
        return float(x)
    except Exception:
        return 0.0

def safe_iloc(series, index=-1):
    """استخراج قيمة من Series أو array بأمان"""
    try:
        if hasattr(series, 'iloc'):
            return float(series.iloc[index])
        elif hasattr(series, '__getitem__'):
            return float(series[index])
        else:
            return float(series)
    except (IndexError, TypeError, ValueError):
        return 0.0

def safe_qty(qty):
    """كمية آمنة حسب خطوة التداول"""
    if LOT_STEP is None:
        return float(Decimal(str(qty)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
    step = float(LOT_STEP)
    if step == 0:
        return float(qty)
    return float(math.floor(float(qty) / step) * step)

def ensure_dataframe_compatibility(data):
    """تأكد من توافق البيانات مع pandas DataFrame"""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        # تحويل numpy array إلى DataFrame
        if data.ndim == 1:
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    else:
        # افتراض أنها قائمة أو أي نوع بيانات آخر
        try:
            return pd.DataFrame(data)
        except:
            return None

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
BOT_VERSION = f"SUI Council PROFESSIONAL v8.0 — {EXCHANGE_NAME.upper()} Multi-Exchange - ACTIVE COUNCIL"
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

# =================== SMART MONEY CONCEPTS SETTINGS ===================
FVG_THRESHOLD = 0.1  # Minimum FVG size percentage
OB_STRENGTH_THRESHOLD = 0.1  # Minimum OB strength percentage
LIQUIDITY_ZONE_PROXIMITY = 0.01  # 1% proximity to liquidity zone

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
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 2.0))
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
ULTIMATE_MIN_CONFIDENCE = 7.0  # Reduced slightly due to more indicators
VOLUME_MOMENTUM_PERIOD = 20
STOCH_RSI_PERIOD = 14
DYNAMIC_PIVOT_PERIOD = 20
TREND_FAST_PERIOD = 10
TREND_SLOW_PERIOD = 20
TREND_SIGNAL_PERIOD = 9

# ==== ACTIVE COUNCIL SETTINGS ====
ACTIVE_COUNCIL_ENABLED = True
MIN_HOLD_TIME_SCALP = 180  # 3 دقائق minimum for scalp
MIN_HOLD_TIME_TREND = 300  # 5 دقائق minimum for trend
PROFIT_TARGET_BOOST_FACTOR = 1.3  # زيادة الأهداف مع قوة المجلس

# ==== SUPER INTELLIGENT SYSTEM SETTINGS ====
SUPER_INTELLIGENT_MODE = True
ADAPTIVE_LEARNING_ENABLED = True
TRADE_CLASSIFICATION_ENABLED = True
MULTI_LEVEL_PROFIT_TAKING = True
INTELLIGENT_PROTECTION_SYSTEM = True

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

# =================== ENHANCED STATE MANAGEMENT ===================
def initialize_state():
    """تهيئة حالة البوت بشكل آمن"""
    global STATE
    default_state = {
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0,
        "profit_targets_achieved": 0, "trade_type": None,
        "trade_size_category": "small", "opened_at": None,
        "peak_profit": 0.0, "max_drawdown": 0.0,
        "adjustment_count": 0, "last_adjustment_time": 0
    }
    
    # دمج مع الحالة المحفوظة
    saved_state = load_state()
    if saved_state.get("in_position"):
        default_state.update({
            "open": True,
            "side": saved_state.get("side", "").lower(),
            "entry": saved_state.get("entry_price"),
            "qty": saved_state.get("position_qty", 0.0),
            "opened_at": saved_state.get("opened_at", time.time())
        })
    
    STATE.update(default_state)
    return STATE

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
    print(f"🎯 PROFESSIONAL COUNCIL: min_confidence={ULTIMATE_MIN_CONFIDENCE}", flush=True)
    print(f"📈 ADVANCED INDICATORS: SMC + MACD + VWAP + Volume Momentum", flush=True)
    print(f"👣 SMART MONEY CONCEPTS: BOS + Order Blocks + FVG + Liquidity Analysis", flush=True)
    print(f"⚡ RF SETTINGS: period={RF_PERIOD} | mult={RF_MULT} (SUI Optimized)", flush=True)
    print(f"🚀 ACTIVE COUNCIL: ENABLED - Enhanced Profit Maximization", flush=True)
    print(f"🧠 SUPER INTELLIGENT SYSTEM: ENABLED - Smart Trade Management", flush=True)
    print(f"🎯 PROFESSIONAL TRADE CLASSIFICATION: ENABLED - Premium/Strong/Normal/Scalp", flush=True)
    print(f"🛡️ ENHANCED PROTECTION SYSTEM: ENABLED - Adaptive Risk Management", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

# =================== ENHANCED INDICATORS ===================
def sma(series, n: int):
    """المتوسط المتحرك البسيط"""
    if hasattr(series, 'rolling'):
        return series.rolling(n, min_periods=1).mean()
    else:
        # إذا كان series عبارة عن numpy array
        return pd.Series(series).rolling(n, min_periods=1).mean()

def compute_rsi(close, n: int = 14):
    """حساب مؤشر RSI مع معالجة numpy arrays"""
    if not hasattr(close, 'diff'):
        close = pd.Series(close)
    
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    
    # استخدام min_periods=1 لتجنب القيم NaN
    roll_up = up.ewm(span=n, min_periods=1, adjust=False).mean()
    roll_down = down.ewm(span=n, min_periods=1, adjust=False).mean()
    
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def compute_indicators(df):
    """حساب المؤشرات الأساسية مع إصلاح مشكلة numpy arrays"""
    if len(df) < max(RF_PERIOD, ATR_LEN, ADX_LEN) + 5:
        return {}
    
    try:
        # تحويل إلى pandas Series إذا كانت numpy arrays
        if not hasattr(df['close'], 'iloc'):
            close = pd.Series(df['close'])
            high = pd.Series(df['high'])
            low = pd.Series(df['low'])
        else:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
        
        # ATR محسّن
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(ATR_LEN, min_periods=1).mean()
        
        # ADX محسّن
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        # استخدام ATR السلس
        atr_smooth = atr.rolling(ADX_LEN, min_periods=1).mean()
        
        plus_di = 100 * (plus_dm.rolling(ADX_LEN, min_periods=1).mean() / atr_smooth.replace(0, 1))
        minus_di = 100 * (minus_dm.rolling(ADX_LEN, min_periods=1).mean() / atr_smooth.replace(0, 1))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-12)
        adx = dx.rolling(ADX_LEN, min_periods=1).mean()
        
        return {
            'atr': float(last_val(atr)),
            'adx': float(last_val(adx)),
            'plus_di': float(last_val(plus_di)),
            'minus_di': float(last_val(minus_di))
        }
    except Exception as e:
        log_w(f"compute_indicators error: {e}")
        return {}

def rsi_ma_context(df):
    """سياق RSI مع معالجة numpy arrays"""
    if len(df) < max(RSI_MA_LEN, 14):
        return {"rsi": 50, "rsi_ma": 50, "cross": "none", "trendZ": "none", "in_chop": True}
    
    # تحويل إلى pandas Series إذا لزم الأمر
    if not hasattr(df['close'], 'iloc'):
        close_series = pd.Series(df['close'])
    else:
        close_series = df['close'].astype(float)
    
    rsi = compute_rsi(close_series, 14)
    rsi_ma = sma(rsi, RSI_MA_LEN)
    
    cross = "none"
    if len(rsi) >= 2:
        rsi_prev = float(rsi.iloc[-2]) if hasattr(rsi, 'iloc') else float(rsi[-2])
        rsi_curr = float(rsi.iloc[-1]) if hasattr(rsi, 'iloc') else float(rsi[-1])
        rsi_ma_prev = float(rsi_ma.iloc[-2]) if hasattr(rsi_ma, 'iloc') else float(rsi_ma[-2])
        rsi_ma_curr = float(rsi_ma.iloc[-1]) if hasattr(rsi_ma, 'iloc') else float(rsi_ma[-1])
        
        if (rsi_prev <= rsi_ma_prev) and (rsi_curr > rsi_ma_curr):
            cross = "bull"
        elif (rsi_prev >= rsi_ma_prev) and (rsi_curr < rsi_ma_curr):
            cross = "bear"
    
    # تحليل الاتجاه
    if hasattr(rsi, 'tail'):
        above = (rsi > rsi_ma).tail(RSI_TREND_PERSIST)
        below = (rsi < rsi_ma).tail(RSI_TREND_PERSIST)
        persist_bull = above.all() if len(above) >= RSI_TREND_PERSIST else False
        persist_bear = below.all() if len(below) >= RSI_TREND_PERSIST else False
    else:
        # Fallback for numpy arrays
        persist_bull = all(rsi[-RSI_TREND_PERSIST:] > rsi_ma[-RSI_TREND_PERSIST:])
        persist_bear = all(rsi[-RSI_TREND_PERSIST:] < rsi_ma[-RSI_TREND_PERSIST:])
    
    current_rsi = float(rsi.iloc[-1]) if hasattr(rsi, 'iloc') else float(rsi[-1])
    in_chop = RSI_NEUTRAL_BAND[0] <= current_rsi <= RSI_NEUTRAL_BAND[1]
    
    return {
        "rsi": current_rsi,
        "rsi_ma": float(rsi_ma.iloc[-1]) if hasattr(rsi_ma, 'iloc') else float(rsi_ma[-1]),
        "cross": cross,
        "trendZ": "bull" if persist_bull else ("bear" if persist_bear else "none"),
        "in_chop": in_chop
    }

# =================== SMART MONEY CONCEPTS (SMC) ===================
def detect_liquidity_zones(df, window=20):
    """اكتشاف مناطق السيولة (Liquidity Pools)"""
    if len(df) < window * 2:
        return {"buy_liquidity": [], "sell_liquidity": []}
    
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float)
    
    # اكتشاف القمم والقيعان الهامة
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(df) - window):
        # قمم
        if (high.iloc[i] == high.iloc[i-window:i+window].max() and 
            high.iloc[i] > high.iloc[i-1] and 
            high.iloc[i] > high.iloc[i+1]):
            resistance_levels.append({
                'price': high.iloc[i],
                'strength': volume.iloc[i],
                'time': df['time'].iloc[i]
            })
        
        # قيعان
        if (low.iloc[i] == low.iloc[i-window:i+window].min() and 
            low.iloc[i] < low.iloc[i-1] and 
            low.iloc[i] < low.iloc[i+1]):
            support_levels.append({
                'price': low.iloc[i],
                'strength': volume.iloc[i],
                'time': df['time'].iloc[i]
            })
    
    return {
        "buy_liquidity": sorted(support_levels, key=lambda x: x['price'])[-5:],  # آخر 5 مستويات دعم
        "sell_liquidity": sorted(resistance_levels, key=lambda x: x['price'])[:5]  # آخر 5 مستويات مقاومة
    }

def detect_fvg(df, threshold=0.1):
    """اكتشاف Fair Value Gaps (FVG)"""
    if len(df) < 3:
        return {"bullish_fvg": [], "bearish_fvg": []}
    
    fvg_bullish = []
    fvg_bearish = []
    
    for i in range(1, len(df) - 1):
        current_low = float(df['low'].iloc[i])
        current_high = float(df['high'].iloc[i])
        prev_high = float(df['high'].iloc[i-1])
        prev_low = float(df['low'].iloc[i-1])
        next_high = float(df['high'].iloc[i+1])
        next_low = float(df['low'].iloc[i+1])
        
        # FVG صاعد: قاع الشمعة الحالية > قمة الشمعة السابقة
        if current_low > prev_high and (current_low - prev_high) / current_low >= threshold/100:
            fvg_bullish.append({
                'low': prev_high,
                'high': current_low,
                'strength': (current_low - prev_high) / current_low * 100,
                'time': df['time'].iloc[i]
            })
        
        # FVG هابط: قمة الشمعة الحالية < قاع الشمعة السابقة
        if current_high < prev_low and (prev_low - current_high) / prev_low >= threshold/100:
            fvg_bearish.append({
                'low': current_high,
                'high': prev_low,
                'strength': (prev_low - current_high) / prev_low * 100,
                'time': df['time'].iloc[i]
            })
    
    return {
        "bullish_fvg": fvg_bullish[-3:],  # آخر 3 FVG صاعدة
        "bearish_fvg": fvg_bearish[-3:]   # آخر 3 FVG هابطة
    }

def detect_market_structure(df):
    """تحليل هيكل السوق (Market Structure)"""
    if len(df) < 20:
        return {"trend": "neutral", "bos_bullish": False, "bos_bearish": False, 
                "choch_bullish": False, "choch_bearish": False, "liquidity_sweep": False}
    
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    # تحديد الاتجاه
    higher_highs = high.rolling(5).apply(lambda x: x[-1] > x[-2] and x[-2] > x[-3], raw=True).fillna(0)
    higher_lows = low.rolling(5).apply(lambda x: x[-1] > x[-2] and x[-2] > x[-3], raw=True).fillna(0)
    lower_highs = high.rolling(5).apply(lambda x: x[-1] < x[-2] and x[-2] < x[-3], raw=True).fillna(0)
    lower_lows = low.rolling(5).apply(lambda x: x[-1] < x[-2] and x[-2] < x[-3], raw=True).fillna(0)
    
    # Break of Structure (BOS)
    bos_bullish = False
    bos_bearish = False
    
    if len(df) >= 10:
        # BOS صاعد: اختراق أعلى قمة سابقة
        recent_high = high.iloc[-10:-1].max()
        bos_bullish = high.iloc[-1] > recent_high
        
        # BOS هابط: اختراق أدنى قاع سابق
        recent_low = low.iloc[-10:-1].min()
        bos_bearish = low.iloc[-1] < recent_low
    
    # Change of Character (CHoCH)
    choch_bullish = higher_highs.iloc[-1] and lower_lows.iloc[-1]
    choch_bearish = lower_lows.iloc[-1] and higher_highs.iloc[-1]
    
    # Liquidity Sweep
    liquidity_sweep = False
    if len(df) >= 5:
        # مسح سيولة: حركة سريعة تجاه مستوى ثم ارتداد
        recent_extreme = high.iloc[-5:-1].max() if bos_bullish else low.iloc[-5:-1].min() if bos_bearish else None
        if recent_extreme:
            move_size = abs(close.iloc[-1] - recent_extreme) / recent_extreme * 100
            liquidity_sweep = move_size > 0.5  # حركة أكثر من 0.5%
    
    # تحديد الاتجاه
    if higher_highs.iloc[-1] and higher_lows.iloc[-1]:
        trend = "bullish"
    elif lower_highs.iloc[-1] and lower_lows.iloc[-1]:
        trend = "bearish"
    else:
        trend = "neutral"
    
    return {
        "trend": trend,
        "bos_bullish": bool(bos_bullish),
        "bos_bearish": bool(bos_bearish),
        "choch_bullish": bool(choch_bullish),
        "choch_bearish": bool(choch_bearish),
        "liquidity_sweep": bool(liquidity_sweep)
    }

def detect_order_blocks(df):
    """اكتشاف Order Blocks (OB)"""
    if len(df) < 10:
        return {"bullish_ob": [], "bearish_ob": []}
    
    bullish_ob = []
    bearish_ob = []
    
    for i in range(5, len(df) - 5):
        # Order Block صاعد: شمعة هابطة كبيرة تليها شمعة صاعدة
        if (df['close'].iloc[i] < df['open'].iloc[i] and  # شمعة هابطة
            df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # شمعة صاعدة تليها
            abs(df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i] > OB_STRENGTH_THRESHOLD/100):  # حجم مناسب
            
            bullish_ob.append({
                'high': max(float(df['high'].iloc[i]), float(df['high'].iloc[i+1])),
                'low': min(float(df['low'].iloc[i]), float(df['low'].iloc[i+1])),
                'strength': abs(df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i] * 100,
                'time': df['time'].iloc[i]
            })
        
        # Order Block هابط: شمعة صاعدة كبيرة تليها شمعة هابطة
        if (df['close'].iloc[i] > df['open'].iloc[i] and  # شمعة صاعدة
            df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # شمعة هابطة تليها
            abs(df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i] > OB_STRENGTH_THRESHOLD/100):  # حجم مناسب
            
            bearish_ob.append({
                'high': max(float(df['high'].iloc[i]), float(df['high'].iloc[i+1])),
                'low': min(float(df['low'].iloc[i]), float(df['low'].iloc[i+1])),
                'strength': abs(df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i] * 100,
                'time': df['time'].iloc[i]
            })
    
    return {
        "bullish_ob": bullish_ob[-5:],  # آخر 5 order blocks صاعدة
        "bearish_ob": bearish_ob[-5:]   # آخر 5 order blocks هابطة
    }

# =================== ADVANCED INDICATORS - PROFESSIONAL ===================
def compute_macd(df, fast=12, slow=26, signal=9):
    """حساب مؤشر MACD المتقدم"""
    if len(df) < slow + signal:
        return {"macd": 0, "signal": 0, "histogram": 0, "trend": "neutral", "crossover": "none", "above_zero": False}
    
    close = df['close'].astype(float)
    
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    # تحليل الاتجاه
    current_macd = last_val(macd_line)
    current_signal = last_val(signal_line)
    current_hist = last_val(histogram)
    
    # اتجاه MACD
    if current_macd > current_signal and current_hist > 0:
        trend = "bullish"
    elif current_macd < current_signal and current_hist < 0:
        trend = "bearish"
    else:
        trend = "neutral"
    
    # تقاطعات
    crossover = "none"
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        if (safe_iloc(macd_line, -2) <= safe_iloc(signal_line, -2) and 
            current_macd > current_signal):
            crossover = "bullish"
        elif (safe_iloc(macd_line, -2) >= safe_iloc(signal_line, -2) and 
              current_macd < current_signal):
            crossover = "bearish"
    
    return {
        "macd": current_macd,
        "signal": current_signal,
        "histogram": current_hist,
        "trend": trend,
        "crossover": crossover,
        "above_zero": current_macd > 0
    }

def compute_vwap(df):
    """حساب VWAP (Volume Weighted Average Price)"""
    if len(df) < 20:
        return {"vwap": 0, "deviation": 0, "signal": "neutral", "price_above_vwap": False}
    
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    current_vwap = last_val(vwap)
    current_price = last_val(close)
    deviation = (current_price - current_vwap) / current_vwap * 100
    
    # إشارات VWAP
    if deviation > 2.0:
        signal = "overbought"
    elif deviation < -2.0:
        signal = "oversold"
    elif deviation > 0.5:
        signal = "bullish"
    elif deviation < -0.5:
        signal = "bearish"
    else:
        signal = "neutral"
    
    return {
        "vwap": current_vwap,
        "deviation": deviation,
        "signal": signal,
        "price_above_vwap": current_price > current_vwap
    }

def enhanced_price_momentum(df):
    """زخم سعر محسن"""
    if len(df) < 20:
        return {"trend": "neutral", "momentum_strength": 0}
    
    close = df['close'].astype(float)
    
    # زخم متعدد الأطر الزمنية
    momentum_5 = (last_val(close) / safe_iloc(close, -5) - 1) * 100
    momentum_10 = (last_val(close) / safe_iloc(close, -10) - 1) * 100
    momentum_20 = (last_val(close) / safe_iloc(close, -20) - 1) * 100
    
    # تسارع الزخم
    acceleration = momentum_5 - momentum_10
    
    # قوة الزخم
    momentum_strength = (abs(momentum_5) + abs(momentum_10) + abs(momentum_20)) / 3
    
    # تحديد الاتجاه
    if momentum_5 > 0.5 and momentum_10 > 0.3 and acceleration > 0:
        trend = "bullish"
    elif momentum_5 < -0.5 and momentum_10 < -0.3 and acceleration < 0:
        trend = "bearish"
    else:
        trend = "neutral"
        
    return {
        "trend": trend,
        "momentum_strength": momentum_strength,
        "acceleration": acceleration,
        "short_term": momentum_5,
        "medium_term": momentum_10
    }

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
    
    current_momentum = last_val(volume_weighted_momentum)
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
    
    current_k = last_val(stoch_k_smooth)
    current_d = last_val(stoch_d)
    
    # إشارات التداول
    signal = "neutral"
    if current_k < 20 and current_d < 20:
        signal = "bullish"
    elif current_k > 80 and current_d > 80:
        signal = "bearish"
    elif current_k > current_d and len(stoch_k_smooth) >= 2 and len(stoch_d) >= 2 and safe_iloc(stoch_k_smooth, -2) <= safe_iloc(stoch_d, -2):
        signal = "bullish_cross"
    elif current_k < current_d and len(stoch_k_smooth) >= 2 and len(stoch_d) >= 2 and safe_iloc(stoch_k_smooth, -2) >= safe_iloc(stoch_d, -2):
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
    
    pivot = (last_val(high) + last_val(low) + last_val(close)) / 3
    r1 = 2 * pivot - last_val(low)
    r2 = pivot + (last_val(high) - last_val(low))
    s1 = 2 * pivot - last_val(high)
    s2 = pivot - (last_val(high) - last_val(low))
    
    current_price = last_val(close)
    
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
        return {"trend": "neutral", "momentum": 0, "signal": "hold", "ema_fast": 0, "ema_slow": 0}
    
    close = df['close'].astype(float)
    
    # متوسطات متحركة متعددة
    ema_fast = close.ewm(span=fast_period).mean()
    ema_slow = close.ewm(span=slow_period).mean()
    ema_signal = ema_fast.ewm(span=signal_period).mean()
    
    # تقاطعات الاتجاه
    fast_above_slow = last_val(ema_fast) > last_val(ema_slow)
    fast_above_signal = last_val(ema_fast) > last_val(ema_signal)
    
    # زخم الاتجاه
    momentum = (last_val(ema_fast) - last_val(ema_slow)) / last_val(ema_slow) * 100
    
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
    if len(ema_fast) >= 2 and len(ema_slow) >= 2:
        if trend == "strong_bull" and safe_iloc(ema_fast, -2) <= safe_iloc(ema_slow, -2):
            signal = "strong_buy"
        elif trend == "bull" and len(ema_signal) >= 2 and safe_iloc(ema_fast, -2) <= safe_iloc(ema_signal, -2):
            signal = "buy"
        elif trend == "strong_bear" and safe_iloc(ema_fast, -2) >= safe_iloc(ema_slow, -2):
            signal = "strong_sell"
        elif trend == "bear" and len(ema_signal) >= 2 and safe_iloc(ema_fast, -2) >= safe_iloc(ema_signal, -2):
            signal = "sell"
    
    return {
        "trend": trend,
        "momentum": momentum,
        "signal": signal,
        "ema_fast": last_val(ema_fast),
        "ema_slow": last_val(ema_slow)
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
            'high': last_val(high),
            'low': last_val(low),
            'close': last_val(close),
            'open': last_val(open_price),
            'volume': last_val(volume),
            'volume_ratio': last_val(volume_ratio),
            'delta': last_val(volume_delta),
            'efficiency': last_val(efficiency)
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
            prev_low = safe_iloc(low, -2)
            prev_high = safe_iloc(high, -2)
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
    return abs(last_val(closes) - safe_iloc(closes, -2)) / max(recent_std, 1e-9)

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
        
        last_close = last_val(c)
        in_zone = (f618 <= last_close <= f786) if side == "down" else (f786 <= last_close <= f618)
        
        if not in_zone:
            return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"price_not_in_zone {last_close:.6f} vs [{f618:.6f},{f786:.6f}]"]}
        
        # الشروط المساعدة
        current_high = last_val(h)
        current_low = last_val(l)
        current_open = last_val(o)
        
        body, up_wick, low_wick = _body_wicks_gz(current_high, current_low, current_open, last_close)
        
        # حجم التداول
        vol_ma = v.rolling(VOL_MA_LEN).mean().iloc[-1]
        vol_ok = last_val(v) >= vol_ma * 0.9  # تخفيف الشرط قليلاً
        
        # RSI
        rsi_series = _rsi_fallback_gz(c, RSI_LEN_GZ)
        rsi_ma_series = _ema_gz(rsi_series, RSI_MA_LEN_GZ)
        rsi_last = last_val(rsi_series)
        rsi_ma_last = last_val(rsi_ma_series)
        
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

# =================== ENHANCED TRADE CLASSIFICATION ===================
def classify_trade_professional(df, state, council_data):
    """تصنيف احترافي للصفقة بناءً على متعددة معايير"""
    if not state["open"]:
        return "unknown"
    
    current_price = last_val(df['close'])
    entry_price = state["entry"]
    current_pnl = (current_price - entry_price) / entry_price * 100 * (1 if state["side"] == "long" else -1)
    
    # تحليل متعدد الأبعاد للصفقة
    trend_analysis = analyze_trend_strength_pro(df, state["side"])
    momentum_quality = assess_momentum_quality_pro(df, state)
    volume_profile = analyze_volume_profile_pro(df)
    council_strength = max(council_data['score_b'], council_data['score_s'])
    time_in_trade = time.time() - state.get("opened_at", time.time())
    
    # نظام نقاط متقدم
    score = 0
    
    # 1. قوة الاتجاه (30 نقطة كحد أقصى)
    trend_score = min(30, trend_analysis["strength"] * 30)
    score += trend_score
    
    # 2. جودة الزخم (25 نقطة)
    momentum_score = min(25, momentum_quality * 25)
    score += momentum_score
    
    # 3. قوة المجلس (20 نقطة)
    council_score = min(20, (council_strength / 10.0) * 20)
    score += council_score
    
    # 4. تحليل الحجم (15 نقطة)
    volume_score = min(15, volume_profile["strength"] * 15)
    score += volume_score
    
    # 5. الوقت في الصفقة (10 نقطة)
    time_score = 0
    if time_in_trade > 1800:  # أكثر من 30 دقيقة
        time_score = 10
    elif time_in_trade > 600:  # أكثر من 10 دقائق
        time_score = 5
    score += time_score
    
    # تصنيف الصفقة
    if score >= 80:
        trade_type = "premium_trend"
        size_category = "large"
    elif score >= 65:
        trade_type = "strong_trend" 
        size_category = "medium"
    elif score >= 50:
        trade_type = "normal_trend"
        size_category = "medium"
    elif score >= 35:
        trade_type = "scalp_quality"
        size_category = "small"
    else:
        trade_type = "regular_scalp"
        size_category = "small"
    
    return {
        "trade_type": trade_type,
        "size_category": size_category,
        "confidence_score": score,
        "trend_strength": trend_analysis["strength"],
        "momentum_quality": momentum_quality,
        "council_strength": council_strength,
        "time_in_trade": time_in_trade
    }

def analyze_trend_strength_pro(df, trade_side):
    """تحليل قوة الاتجاه المتقدم"""
    if len(df) < 50:
        return {"strength": 0.5, "persistence": 0.5, "quality": 0.5}
    
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # متعددة الأطر الزمنية للاتجاه
    ema_8 = close.ewm(span=8).mean()
    ema_21 = close.ewm(span=21).mean()
    ema_50 = close.ewm(span=50).mean()
    ema_100 = close.ewm(span=100).mean()
    
    # محاذاة المتوسطات
    alignment_score = 0
    if (last_val(ema_8) > last_val(ema_21) > last_val(ema_50) > last_val(ema_100)):
        alignment_score = 1.0
    elif (last_val(ema_8) < last_val(ema_21) < last_val(ema_50) < last_val(ema_100)):
        alignment_score = 1.0
    elif (last_val(ema_8) > last_val(ema_21) > last_val(ema_50)):
        alignment_score = 0.7
    elif (last_val(ema_8) < last_val(ema_21) < last_val(ema_50)):
        alignment_score = 0.7
    else:
        alignment_score = 0.3
    
    # استمرارية الاتجاه
    persistence_score = 0
    lookback = min(20, len(ema_8))
    for i in range(1, lookback):
        if (safe_iloc(ema_8, -i) > safe_iloc(ema_21, -i) > safe_iloc(ema_50, -i)):
            persistence_score += 0.05
        elif (safe_iloc(ema_8, -i) < safe_iloc(ema_21, -i) < safe_iloc(ema_50, -i)):
            persistence_score += 0.05
    
    # جودة الحركة (نسبة الجسم إلى المدى)
    body_ratio = []
    for i in range(1, min(10, len(df))):
        candle = df.iloc[-i]
        body = abs(float(candle['close']) - float(candle['open']))
        range_val = float(candle['high']) - float(candle['low'])
        if range_val > 0:
            body_ratio.append(body / range_val)
    
    quality_score = np.mean(body_ratio) if body_ratio else 0.5
    
    # القوة النهائية
    strength = (alignment_score * 0.4 + persistence_score * 0.4 + quality_score * 0.2)
    
    return {
        "strength": min(1.0, strength),
        "persistence": min(1.0, persistence_score),
        "quality": min(1.0, quality_score)
    }

def assess_momentum_quality_pro(df, state):
    """تقييم جودة الزخم المتقدم"""
    momentum = enhanced_price_momentum(df)
    current_price = last_val(df['close'])
    entry_price = state["entry"]
    current_pnl = (current_price - entry_price) / entry_price * 100 * (1 if state["side"] == "long" else -1)
    
    score = 0.0
    
    # قوة الزخم الحالية
    momentum_strength = momentum.get("momentum_strength", 0)
    score += min(0.4, momentum_strength / 25.0)  # حتى 40%
    
    # استمرارية الزخم
    if momentum.get("trend") == ("bullish" if state["side"] == "long" else "bearish"):
        score += 0.3
    
    # تسارع الزخم
    acceleration = momentum.get("acceleration", 0)
    if (state["side"] == "long" and acceleration > 0.1) or (state["side"] == "short" and acceleration < -0.1):
        score += 0.2
    
    # استقرار الزخم
    if abs(acceleration) < 0.5:  # تسارع معتدل
        score += 0.1
    
    return min(1.0, score)

def analyze_volume_profile_pro(df):
    """تحليل متقدم لملف الحجم"""
    if len(df) < 20:
        return {"strength": 0.5, "consistency": 0.5, "participation": 0.5}
    
    volume = df['volume'].astype(float)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # قوة الحجم
    recent_volume = volume.tail(5).mean()
    avg_volume = volume.tail(20).mean()
    volume_strength = recent_volume / avg_volume if avg_volume > 0 else 1.0
    
    # اتساق الحجم
    volume_std = volume.tail(10).std()
    volume_mean = volume.tail(10).mean()
    volume_consistency = 1.0 - (volume_std / volume_mean) if volume_mean > 0 else 0.5
    
    # مشاركة الحجم في الاتجاه
    up_days = close > close.shift(1)
    volume_on_up = volume[up_days].tail(5).mean() if up_days.any() else 0
    volume_on_down = volume[~up_days].tail(5).mean() if (~up_days).any() else 0
    volume_participation = volume_on_up / volume_on_down if volume_on_down > 0 else 1.0
    
    # القوة النهائية
    strength = (min(2.0, volume_strength) * 0.4 + 
                volume_consistency * 0.3 + 
                min(2.0, volume_participation) * 0.3) / 1.5
    
    return {
        "strength": min(1.0, strength),
        "consistency": min(1.0, volume_consistency),
        "participation": min(2.0, volume_participation)
    }

# =================== PROFESSIONAL PROFIT MANAGEMENT ===================
def professional_profit_management(state, df, trade_classification, council_data):
    """إدارة أرباح احترافية بناءً على تصنيف الصفقة"""
    current_price = last_val(df['close'])
    entry_price = state["entry"]
    current_pnl = (current_price - entry_price) / entry_price * 100 * (1 if state["side"] == "long" else -1)
    
    trade_type = trade_classification["trade_type"]
    size_category = trade_classification["size_category"]
    confidence_score = trade_classification["confidence_score"]
    
    # تحديث ذروة الربح
    if current_pnl > state.get("peak_profit", 0):
        state["peak_profit"] = current_pnl
        state["max_drawdown"] = 0
    else:
        drawdown = state["peak_profit"] - current_pnl
        state["max_drawdown"] = max(state.get("max_drawdown", 0), drawdown)
    
    # استراتيجيات الجني حسب نوع الصفقة
    if trade_type == "premium_trend":
        return manage_premium_trend(state, current_pnl, confidence_score)
    elif trade_type == "strong_trend":
        return manage_strong_trend(state, current_pnl, confidence_score)
    elif trade_type == "normal_trend":
        return manage_normal_trend(state, current_pnl, confidence_score)
    elif trade_type == "scalp_quality":
        return manage_quality_scalp(state, current_pnl, confidence_score)
    else:
        return manage_regular_scalp(state, current_pnl, confidence_score)

def manage_premium_trend(state, current_pnl, confidence_score):
    """إدارة صفقات الترند الممتاز"""
    targets = [
        {"level": 1.0, "close_ratio": 0.15, "reason": "جني أولي - ترند قوي"},
        {"level": 2.0, "close_ratio": 0.20, "reason": "جني ثانوي - تعزيز الربح"},
        {"level": 3.5, "close_ratio": 0.25, "reason": "جني متقدم - استمرارية الترند"},
        {"level": 5.0, "close_ratio": 0.25, "reason": "جني شبه نهائي"},
        {"level": 7.0, "close_ratio": 0.15, "reason": "جني نهائي - تحقيق كامل"}
    ]
    
    # تعديل الأهداف بناءً على ثقة الصفقة
    confidence_boost = 1.0 + (confidence_score - 80) * 0.02
    achieved_targets = state.get("achieved_targets", [])
    
    for target in targets:
        adjusted_level = target["level"] * confidence_boost
        if current_pnl >= adjusted_level and target["level"] not in achieved_targets:
            return {
                "action": "partial_close",
                "close_ratio": target["close_ratio"],
                "target_level": target["level"],
                "adjusted_level": adjusted_level,
                "reason": f"{target['reason']} (ممتاز)",
                "strategy": "premium_trend"
            }
    
    return {"action": "hold", "reason": "لم يتم تحقيق أهداف جديدة"}

def manage_strong_trend(state, current_pnl, confidence_score):
    """إدارة صفقات الترند القوي"""
    targets = [
        {"level": 0.8, "close_ratio": 0.20, "reason": "جني أولي - ترند قوي"},
        {"level": 1.5, "close_ratio": 0.25, "reason": "جني ثانوي - استمرارية"},
        {"level": 2.5, "close_ratio": 0.30, "reason": "جني متقدم"},
        {"level": 4.0, "close_ratio": 0.25, "reason": "جني نهائي"}
    ]
    
    confidence_boost = 1.0 + (confidence_score - 65) * 0.015
    achieved_targets = state.get("achieved_targets", [])
    
    for target in targets:
        adjusted_level = target["level"] * confidence_boost
        if current_pnl >= adjusted_level and target["level"] not in achieved_targets:
            return {
                "action": "partial_close",
                "close_ratio": target["close_ratio"],
                "target_level": target["level"],
                "adjusted_level": adjusted_level,
                "reason": f"{target['reason']} (قوي)",
                "strategy": "strong_trend"
            }
    
    return {"action": "hold", "reason": "لم يتم تحقيق أهداف جديدة"}

def manage_normal_trend(state, current_pnl, confidence_score):
    """إدارة صفقات الترند العادي"""
    targets = [
        {"level": 0.6, "close_ratio": 0.30, "reason": "جني أولي - ترند عادي"},
        {"level": 1.2, "close_ratio": 0.40, "reason": "جني ثانوي"},
        {"level": 2.0, "close_ratio": 0.30, "reason": "جني نهائي"}
    ]
    
    confidence_boost = 1.0 + (confidence_score - 50) * 0.01
    achieved_targets = state.get("achieved_targets", [])
    
    for target in targets:
        adjusted_level = target["level"] * confidence_boost
        if current_pnl >= adjusted_level and target["level"] not in achieved_targets:
            return {
                "action": "partial_close",
                "close_ratio": target["close_ratio"],
                "target_level": target["level"],
                "adjusted_level": adjusted_level,
                "reason": f"{target['reason']} (عادي)",
                "strategy": "normal_trend"
            }
    
    return {"action": "hold", "reason": "لم يتم تحقيق أهداف جديدة"}

def manage_quality_scalp(state, current_pnl, confidence_score):
    """إدارة صفقات السكالب الجيدة"""
    targets = [
        {"level": 0.4, "close_ratio": 0.40, "reason": "جني سريع - سكالب جيد"},
        {"level": 0.8, "close_ratio": 0.60, "reason": "جني نهائي - سكالب"}
    ]
    
    confidence_boost = 1.0 + (confidence_score - 35) * 0.02
    achieved_targets = state.get("achieved_targets", [])
    
    for target in targets:
        adjusted_level = target["level"] * confidence_boost
        if current_pnl >= adjusted_level and target["level"] not in achieved_targets:
            return {
                "action": "partial_close",
                "close_ratio": target["close_ratio"],
                "target_level": target["level"],
                "adjusted_level": adjusted_level,
                "reason": f"{target['reason']} (سكالب جيد)",
                "strategy": "quality_scalp"
            }
    
    return {"action": "hold", "reason": "لم يتم تحقيق أهداف جديدة"}

def manage_regular_scalp(state, current_pnl, confidence_score):
    """إدارة صفقات السكالب العادية"""
    targets = [
        {"level": 0.3, "close_ratio": 0.50, "reason": "جني سريع - سكالب"},
        {"level": 0.6, "close_ratio": 0.50, "reason": "جني نهائي - سكالب"}
    ]
    
    achieved_targets = state.get("achieved_targets", [])
    
    for target in targets:
        if current_pnl >= target["level"] and target["level"] not in achieved_targets:
            return {
                "action": "partial_close",
                "close_ratio": target["close_ratio"],
                "target_level": target["level"],
                "reason": f"{target['reason']} (سكالب)",
                "strategy": "regular_scalp"
            }
    
    return {"action": "hold", "reason": "لم يتم تحقيق أهداف جديدة"}

# =================== ENHANCED PROTECTION SYSTEM ===================
def professional_protection_system(state, df, trade_classification, current_pnl):
    """نظام حماية احترافي يتكيف مع نوع الصفقة"""
    current_price = last_val(df['close'])
    entry_price = state["entry"]
    trade_type = trade_classification["trade_type"]
    max_drawdown = state.get("max_drawdown", 0)
    time_in_trade = time.time() - state.get("opened_at", time.time())
    
    # حماية من الخسارة الكبيرة
    if current_pnl <= -3.0:
        return {
            "action": "emergency_close", 
            "reason": "خسارة طارئة - تجاوز الحد المسموح",
            "severity": "high"
        }
    
    elif current_pnl <= -2.0 and max_drawdown >= 3.0:
        return {
            "action": "emergency_close",
            "reason": "خسارة مستمرة - حماية رأس المال",
            "severity": "high"
        }
    
    # استراتيجيات الحماية حسب نوع الصفقة
    if trade_type == "premium_trend":
        return protect_premium_trend(state, current_pnl, time_in_trade, max_drawdown)
    elif trade_type == "strong_trend":
        return protect_strong_trend(state, current_pnl, time_in_trade, max_drawdown)
    elif trade_type == "normal_trend":
        return protect_normal_trend(state, current_pnl, time_in_trade, max_drawdown)
    else:
        return protect_scalp_trades(state, current_pnl, time_in_trade, max_drawdown)

def protect_premium_trend(state, current_pnl, time_in_trade, max_drawdown):
    """حماية صفقات الترند الممتاز"""
    if current_pnl >= 2.0 and not state.get("breakeven_armed"):
        return {
            "action": "breakeven",
            "reason": "تفعيل نقطة التعادل - ترند ممتاز",
            "severity": "medium"
        }
    
    elif current_pnl >= 4.0 and not state.get("trail_activated"):
        return {
            "action": "activate_trail",
            "trail_type": "atr_wide",
            "reason": "تفعيل التريل الواسع - حماية الأرباح",
            "severity": "low"
        }
    
    elif current_pnl >= 6.0 and max_drawdown <= 1.0:
        return {
            "action": "tighten_trail",
            "trail_type": "atr_tight",
            "reason": "تشديد التريل - تأمين الأرباح العالية",
            "severity": "low"
        }
    
    return {"action": "hold", "reason": "لا توجد إجراءات حماية مطلوبة"}

def protect_strong_trend(state, current_pnl, time_in_trade, max_drawdown):
    """حماية صفقات الترند القوي"""
    if current_pnl >= 1.5 and not state.get("breakeven_armed"):
        return {
            "action": "breakeven",
            "reason": "تفعيل نقطة التعادل - ترند قوي",
            "severity": "medium"
        }
    
    elif current_pnl >= 3.0 and not state.get("trail_activated"):
        return {
            "action": "activate_trail",
            "trail_type": "atr_normal",
            "reason": "تفعيل التريل العادي",
            "severity": "low"
        }
    
    elif current_pnl >= 4.0 and time_in_trade > 1800:  # بعد 30 دقيقة
        return {
            "action": "consider_partial_close",
            "close_ratio": 0.2,
            "reason": "تفكير في جني جزئي - مكاسب جيدة مع وقت طويل",
            "severity": "low"
        }
    
    return {"action": "hold", "reason": "لا توجد إجراءات حماية مطلوبة"}

def protect_normal_trend(state, current_pnl, time_in_trade, max_drawdown):
    """حماية صفقات الترند العادي"""
    if current_pnl >= 1.0 and not state.get("breakeven_armed"):
        return {
            "action": "breakeven",
            "reason": "تفعيل نقطة التعادل - ترند عادي",
            "severity": "medium"
        }
    
    elif current_pnl >= 2.0 and not state.get("trail_activated"):
        return {
            "action": "activate_trail",
            "trail_type": "atr_normal",
            "reason": "تفعيل التريل للحماية",
            "severity": "low"
        }
    
    elif current_pnl <= -1.5 and time_in_trade > 1200:  # بعد 20 دقيقة
        return {
            "action": "consider_early_close",
            "reason": "تفكير في إغلاق مبكر - أداء ضعيف مع وقت كاف",
            "severity": "medium"
        }
    
    return {"action": "hold", "reason": "لا توجد إجراءات حماية مطلوبة"}

def protect_scalp_trades(state, current_pnl, time_in_trade, max_drawdown):
    """حماية صفقات السكالب"""
    if current_pnl >= 0.5 and not state.get("breakeven_armed"):
        return {
            "action": "breakeven",
            "reason": "تفعيل نقطة التعادل السريع - سكالب",
            "severity": "medium"
        }
    
    elif current_pnl >= 1.0:
        return {
            "action": "activate_trail",
            "trail_type": "atr_tight",
            "reason": "تفعيل التريل الضيق - سكالب",
            "severity": "low"
        }
    
    elif current_pnl <= -1.0 and time_in_trade > 600:  # بعد 10 دقائق
        return {
            "action": "early_close",
            "reason": "إغلاق مبكر - سكالب فاشل",
            "severity": "high"
        }
    
    return {"action": "hold", "reason": "لا توجد إجراءات حماية مطلوبة"}

# =================== ENHANCED SUPER INTELLIGENT SYSTEM ===================
def manage_after_entry_super_intelligent_enhanced(state, df, market_data):
    """نظام إدارة محسن فائق الذكاء"""
    if not state.get("open", False):
        return {"action": "hold", "reason": "لا توجد صفقة مفتوحة"}
    
    current_price = market_data.get("price", last_val(df['close']))
    council_data = enhanced_active_council(df, state)
    
    # تصنيف الصفقة بشكل احترافي
    trade_classification = classify_trade_professional(df, state, council_data)
    
    # تحديث حالة الصفقة
    state["trade_type"] = trade_classification["trade_type"]
    state["size_category"] = trade_classification["size_category"]
    
    # إدارة الأرباح الاحترافية
    profit_decision = professional_profit_management(state, df, trade_classification, council_data)
    
    # نظام الحماية المتقدم
    current_pnl = (current_price - state["entry"]) / state["entry"] * 100 * (1 if state["side"] == "long" else -1)
    protection_decision = professional_protection_system(state, df, trade_classification, current_pnl)
    
    # تحليل استمرارية الصفقة المحسن
    continuation_analysis = analyze_trade_continuation_enhanced_pro(state, council_data, current_pnl, trade_classification)
    
    # نظام الأسبقية الذكي للقرارات
    decision = resolve_professional_decisions(
        profit_decision, 
        protection_decision, 
        continuation_analysis,
        trade_classification
    )
    
    # تطبيق القرار
    apply_professional_decision(decision, state, df)
    
    # التسجيل المحترف
    log_professional_management_decision(state, decision, trade_classification, current_pnl)
    
    return decision

def resolve_professional_decisions(profit_decision, protection_decision, continuation_analysis, trade_classification):
    """حل تنازع القرارات بشكل احترافي"""
    # أولوية عالية: الحماية الطارئة
    if protection_decision["action"] in ["emergency_close", "early_close"]:
        return protection_decision
    
    # أولوية متوسطة: جني الأرباح
    if profit_decision["action"] == "partial_close":
        return profit_decision
    
    # أولوية منخفضة: إجراءات حماية وقائية
    if protection_decision["action"] in ["breakeven", "activate_trail", "tighten_trail"]:
        return protection_decision
    
    # قرارات الاستمرارية
    if not continuation_analysis["should_continue"]:
        return {
            "action": "close", 
            "reason": continuation_analysis["reason"],
            "strategy": "continuation_analysis"
        }
    
    return {"action": "hold", "reason": "استمرارية الصفقة مع المراقبة"}

def apply_professional_decision(decision, state, df):
    """تطبيق القرارات بشكل احترافي"""
    action = decision["action"]
    
    if action == "partial_close":
        execute_intelligent_partial_close(state, decision.get("close_ratio", 0.3), decision["reason"])
        
    elif action == "emergency_close":
        log_w(f"🚨 إغلاق طارئ: {decision['reason']}")
        close_market_strict(f"emergency_{decision['reason']}")
        
    elif action == "early_close":
        log_w(f"🟡 إغلاق مبكر: {decision['reason']}")
        close_market_strict(f"early_close_{decision['reason']}")
        
    elif action in ["breakeven", "activate_trail", "tighten_trail"]:
        apply_intelligent_protection(decision, state)
        
    elif action == "consider_partial_close":
        # تفكير في الجني - يمكن تنفيذه في الدورة القادمة إذا استمرت الظروف
        log_i(f"💭 تفكير في جني جزئي: {decision['reason']}")
        
    elif action == "consider_early_close":
        # تفكير في الإغلاق المبكر
        log_i(f"💭 تفكير في إغلاق مبكر: {decision['reason']}")

def analyze_trade_continuation_enhanced_pro(state, council_data, current_pnl, trade_classification):
    """تحليل استمرارية متقدم مع مراعاة تصنيف الصفقة"""
    side = state["side"]
    trade_type = trade_classification["trade_type"]
    confidence_score = trade_classification["confidence_score"]
    
    # عوامل الاستمرارية الإيجابية
    positive_factors = 0
    negative_factors = 0
    
    # 1. قوة المجلس (40% وزن)
    council_strength = max(council_data['score_b'], council_data['score_s'])
    if (side == "long" and council_data['score_b'] > council_data['score_s'] + 2) or \
       (side == "short" and council_data['score_s'] > council_data['score_b'] + 2):
        positive_factors += 4 * (council_strength / 10.0)
    
    # 2. الزخم والاتجاه (30% وزن)
    momentum = council_data.get("advanced_metrics", {}).get("price_momentum", {})
    if (side == "long" and momentum.get("trend") == "bullish") or \
       (side == "short" and momentum.get("trend") == "bearish"):
        positive_factors += 3
        
    # 3. تحليل SMC (20% وزن)
    smc_data = council_data.get("advanced_metrics", {}).get("smc_analysis", {})
    market_structure = smc_data.get("market_structure", {})
    
    if side == "long" and market_structure.get("bos_bullish"):
        positive_factors += 2
    elif side == "short" and market_structure.get("bos_bearish"):
        positive_factors += 2
        
    # 4. الأداء الحالي (10% وزن)
    if current_pnl > 0:
        positive_factors += 1
    else:
        negative_factors += 1
        
    # عوامل سلبية
    if current_pnl < -1.0:
        negative_factors += 2
    elif current_pnl < -0.5:
        negative_factors += 1
        
    # تعديل حسب نوع الصفقة
    if trade_type == "premium_trend":
        positive_factors *= 1.2  # تشجيع الاستمرارية في الصفقات الممتازة
    elif trade_type in ["regular_scalp", "scalp_quality"]:
        negative_factors *= 1.3  # تشديد في صفقات السكالب
        
    # قرار الاستمرارية
    should_continue = positive_factors > negative_factors
    confidence = positive_factors / max(positive_factors + negative_factors, 1)
    
    return {
        "should_continue": should_continue,
        "confidence": confidence,
        "reason": f"عوامل إيجابية: {positive_factors:.1f}, عوامل سلبية: {negative_factors:.1f}",
        "positive_factors": positive_factors,
        "negative_factors": negative_factors
    }

def log_professional_management_decision(state, decision, trade_classification, current_pnl):
    """تسجيل محترف لقرارات الإدارة"""
    if decision["action"] != "hold":
        print(f"\n🎯 إدارة احترافية - تقرير القرار:", flush=True)
        print(f"📊 نوع الصفقة: {trade_classification['trade_type']}", flush=True)
        print(f"⭐ ثقة الصفقة: {trade_classification['confidence_score']:.1f}", flush=True)
        print(f"💰 الربح الحالي: {current_pnl:.2f}%", flush=True)
        print(f"📈 القرار: {decision['action']} - {decision['reason']}", flush=True)
        
        if "strategy" in decision:
            print(f"🎯 الإستراتيجية: {decision['strategy']}", flush=True)
            
        print("─" * 80, flush=True)

# =================== SUPER INTELLIGENT TRADE MANAGEMENT ===================
def execute_intelligent_partial_close(state, close_ratio, reason):
    """تنفيذ جني أرباح ذكي"""
    close_qty = safe_qty(state["qty"] * close_ratio)
    
    if close_qty > 0:
        close_side = "sell" if state["side"] == "long" else "buy"
        
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            try:
                params = exchange_specific_params(close_side, is_close=True)
                ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                
                log_g(f"🧠 جني ذكي: {close_ratio*100:.1f}% - {reason}")
                
                # تحديث الحالة
                state["qty"] = safe_qty(state["qty"] - close_qty)
                state["profit_targets_achieved"] = state.get("profit_targets_achieved", 0) + 1
                
                # حفظ قرار الجني
                if "achieved_targets" not in state:
                    state["achieved_targets"] = []
                state["achieved_targets"].append(close_ratio)
                
            except Exception as e:
                log_e(f"❌ فشل الجني الذكي: {e}")
        else:
            log_i(f"DRY_RUN: جني ذكي {close_qty:.4f} - {reason}")

def apply_intelligent_protection(decision, state):
    """تطبيق إجراءات الحماية الذكية"""
    action = decision["action"]
    
    if action == "breakeven":
        state["breakeven_armed"] = True
        log_i(f"🛡️ تفعيل نقطة التعادل: {decision['reason']}")
        
    elif action == "activate_trail":
        state["trail_activated"] = True
        state["trail_type"] = decision.get("trail_type", "atr_normal")
        log_i(f"🛡️ تفعيل التريل: {decision['reason']}")
        
    elif action == "tighten_trail":
        state["trail_tightened"] = True
        log_i(f"🛡️ تشديد التريل: {decision['reason']}")
        
    elif action == "emergency_close":
        log_w(f"🚨 إغلاق طارئ: {decision['reason']}")
        close_market_strict(f"emergency_{decision['reason']}")

# =================== ACTIVE COUNCIL SYSTEMS ===================
class ActiveCouncilTracker:
    """نظام تتبع نشط للمجلس أثناء الصفقات"""
    
    def __init__(self):
        self.entry_time = None
        self.peak_profit = 0.0
        self.trend_momentum = 1.0
        self.council_strength = 0.0
        self.adjustment_history = []
        
    def start_trade(self, side, entry_price, council_data):
        """بدء تتبع الصفقة الجديدة"""
        self.entry_time = time.time()
        self.peak_profit = 0.0
        self.trend_momentum = 1.0
        self.council_strength = max(council_data['score_b'], council_data['score_s'])
        self.adjustment_history = []
        
    def update_tracking(self, current_pnl, council_data, market_conditions):
        """تحديث التتبع أثناء الصفقة"""
        # تحديث ذروة الربح
        if current_pnl > self.peak_profit:
            self.peak_profit = current_pnl
            
        # تحديث قوة المجلس
        current_strength = max(council_data['score_b'], council_data['score_s'])
        strength_change = current_strength - self.council_strength
        self.council_strength = current_strength
        
        # تحديث زخم الاتجاه
        self.update_trend_momentum(strength_change, market_conditions)
        
        return self.get_tracking_metrics()
    
    def update_trend_momentum(self, strength_change, market_conditions):
        """تحديث زخم الاتجاه"""
        momentum_factor = 1.0
        
        # زيادة الزخم مع قوة المجلس
        if strength_change > 0.5:
            momentum_factor += 0.1
        elif strength_change < -0.5:
            momentum_factor -= 0.1
            
        # تعديل حسب ظروف السوق
        volatility = market_conditions.get('volatility', 1.0)
        if volatility > 1.5:
            momentum_factor *= 1.2  # زيادة الزخم في السوق المتقلب
            
        self.trend_momentum *= momentum_factor
        self.trend_momentum = max(0.5, min(2.0, self.trend_momentum))
        
    def get_tracking_metrics(self):
        """الحصول على مقاييس التتبع"""
        return {
            'peak_profit': self.peak_profit,
            'trend_momentum': self.trend_momentum,
            'council_strength': self.council_strength,
            'time_in_trade': time.time() - self.entry_time if self.entry_time else 0
        }

# تهيئة أنظمة المجلس النشط
active_tracker = ActiveCouncilTracker()

# =================== ENHANCED PROFESSIONAL COUNCIL ===================
def enhanced_professional_council(df):
    """مجلس إدارة محسن بمؤشرات متقدمة"""
    try:
        current_price = last_val(df['close'])
        basic_indicators = compute_indicators(df)
        
        # المؤشرات المتقدمة الجديدة
        advanced_metrics = {
            "volume_analysis": advanced_volume_analysis(df),
            "price_momentum": enhanced_price_momentum(df),
            "market_cycle": detect_market_cycle(df),
            "liquidity_analysis": advanced_liquidity_detection(df),
            "sentiment_indicators": market_sentiment_analysis(df)
        }
        
        # نظام تصويت متقدم
        votes_buy = 0
        votes_sell = 0
        confidence_buy = 0.0
        confidence_sell = 0.0
        
        # 1. تحليل الحجم المتقدم (25% وزن)
        volume_signal = advanced_metrics["volume_analysis"]
        if volume_signal.get("bullish"):
            votes_buy += 3
            confidence_buy += volume_signal.get("strength", 0) * 2.5
        if volume_signal.get("bearish"):
            votes_sell += 3
            confidence_sell += volume_signal.get("strength", 0) * 2.5
            
        # 2. زخم السعر المحسن (20% وزن)
        momentum = advanced_metrics["price_momentum"]
        if momentum.get("trend") == "bullish":
            votes_buy += 2
            confidence_buy += momentum.get("momentum_strength", 0) * 2.0
        if momentum.get("trend") == "bearish":
            votes_sell += 2
            confidence_sell += momentum.get("momentum_strength", 0) * 2.0
            
        # 3. دورة السوق (15% وزن)
        market_cycle = advanced_metrics["market_cycle"]
        if market_cycle.get("phase") in ["accumulation", "uptrend"]:
            votes_buy += 2
            confidence_buy += 1.5
        if market_cycle.get("phase") in ["distribution", "downtrend"]:
            votes_sell += 2
            confidence_sell += 1.5
            
        # 4. تحليل السيولة (20% وزن)
        liquidity = advanced_metrics["liquidity_analysis"]
        if liquidity.get("buy_pressure"):
            votes_buy += 2
            confidence_buy += liquidity.get("pressure_strength", 0) * 2.0
        if liquidity.get("sell_pressure"):
            votes_sell += 2
            confidence_sell += liquidity.get("pressure_strength", 0) * 2.0
            
        # 5. المشاعر (10% وزن)
        sentiment = advanced_metrics["sentiment_indicators"]
        if sentiment.get("bullish"):
            votes_buy += 1
            confidence_buy += 1.0
        if sentiment.get("bearish"):
            votes_sell += 1
            confidence_sell += 1.0
            
        # 6. المؤشرات الأساسية (10% وزن)
        if basic_indicators.get('adx', 0) > 20:
            if basic_indicators.get('plus_di', 0) > basic_indicators.get('minus_di', 0):
                votes_buy += 1
                confidence_buy += 1.0
            else:
                votes_sell += 1
                confidence_sell += 1.0
                
        return {
            "b": votes_buy, "s": votes_sell,
            "score_b": confidence_buy, "score_s": confidence_sell,
            "advanced_metrics": advanced_metrics,
            "market_analysis": {
                "trade_mode": enhanced_trade_mode_detection(df),
                "volatility": calculate_advanced_volatility(df),
                "trend_strength": analyze_trend_strength(df)
            },
            "ind": basic_indicators  # إضافة المؤشرات الأساسية
        }
        
    except Exception as e:
        log_w(f"Enhanced council error: {e}")
        return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "ind": {}}

def advanced_volume_analysis(df):
    """تحليل حجم متقدم"""
    if len(df) < 30:
        return {"bullish": False, "bearish": False, "strength": 0}
    
    volume = df['volume'].astype(float)
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # حجم على القمم vs القيعان
    up_days = close > close.shift(1)
    volume_on_up = volume[up_days].mean() if up_days.any() else 0
    volume_on_down = volume[~up_days].mean() if (~up_days).any() else 0
    
    # نسبة الحجم
    volume_ratio = volume_on_up / volume_on_down if volume_on_down > 0 else 1
    
    # تركيز الحجم
    recent_volume = volume.tail(5).mean()
    avg_volume = volume.tail(20).mean()
    volume_concentration = recent_volume / avg_volume if avg_volume > 0 else 1
    
    bullish = volume_ratio > 1.2 and volume_concentration > 1.1
    bearish = volume_ratio < 0.8 and volume_concentration > 1.1
    
    strength = abs(volume_ratio - 1) * volume_concentration
    
    return {
        "bullish": bullish,
        "bearish": bearish,
        "strength": strength,
        "volume_ratio": volume_ratio,
        "concentration": volume_concentration
    }

def detect_market_cycle(df):
    """اكتشاف دورة السوق"""
    if len(df) < 50:
        return {"phase": "neutral", "confidence": 0}
    
    close = df['close'].astype(float)
    
    # المتوسطات المتحركة للدورة
    ema_short = close.ewm(span=10).mean()
    ema_medium = close.ewm(span=30).mean()
    ema_long = close.ewm(span=50).mean()
    
    # تحديد المرحلة
    price_vs_short = last_val(close) > last_val(ema_short)
    short_vs_medium = last_val(ema_short) > last_val(ema_medium)
    medium_vs_long = last_val(ema_medium) > last_val(ema_long)
    
    if price_vs_short and short_vs_medium and medium_vs_long:
        phase = "uptrend"
        confidence = 0.9
    elif not price_vs_short and not short_vs_medium and not medium_vs_long:
        phase = "downtrend"
        confidence = 0.9
    elif price_vs_short and not short_vs_medium:
        phase = "pullback"
        confidence = 0.6
    elif not price_vs_short and short_vs_medium:
        phase = "bounce"
        confidence = 0.6
    else:
        phase = "consolidation"
        confidence = 0.5
        
    return {"phase": phase, "confidence": confidence}

def advanced_liquidity_detection(df):
    """اكتشاف سيولة متقدم"""
    if len(df) < 20:
        return {"buy_pressure": False, "sell_pressure": False, "pressure_strength": 0}
    
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    # ضغط الشراء: سعر الإغلاق في أعلى النطاق مع حجم عالي
    range_position = (close - low) / (high - low).replace(0, 1)
    buy_pressure = (range_position > 0.7) & (volume > volume.rolling(20).mean() * 1.2)
    
    # ضغط البيع: سعر الإغلاق في أسفل النطاق مع حجم عالي
    sell_pressure = (range_position < 0.3) & (volume > volume.rolling(20).mean() * 1.2)
    
    # قوة الضغط
    pressure_strength = 0
    if buy_pressure.iloc[-1]:
        pressure_strength = range_position.iloc[-1] * (volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    elif sell_pressure.iloc[-1]:
        pressure_strength = (1 - range_position.iloc[-1]) * (volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])
    
    return {
        "buy_pressure": bool(buy_pressure.iloc[-1]),
        "sell_pressure": bool(sell_pressure.iloc[-1]),
        "pressure_strength": pressure_strength
    }

def market_sentiment_analysis(df):
    """تحليل مشاعر السوق"""
    if len(df) < 20:
        return {"bullish": False, "bearish": False, "sentiment_strength": 0}
    
    rsi = compute_rsi(df['close'].astype(float), 14)
    current_rsi = last_val(rsi)
    
    # مؤشرات المشاعر
    bullish_sentiment = current_rsi < 70 and current_rsi > 30  # ليست في مناطق ذروة الشراء/البيع
    bearish_sentiment = current_rsi > 30 and current_rsi < 70  # ليست في مناطق ذروة الشراء/البيع
    
    # قوة المشاعر
    sentiment_strength = 1 - abs(current_rsi - 50) / 50  # كلما كان RSI أقرب لـ 50، زادت قوة المشاعر
    
    return {
        "bullish": bullish_sentiment,
        "bearish": bearish_sentiment,
        "sentiment_strength": sentiment_strength
    }

def calculate_advanced_volatility(df):
    """حساب التقلب المتقدم"""
    if len(df) < 20:
        return 1.0
        
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * math.sqrt(365) * 100  # التقلب السنوي %
    
    # التقلب النسبي
    avg_volatility = 2.0  # افتراضي
    relative_vol = volatility / avg_volatility
    
    return relative_vol

def analyze_trend_strength(df):
    """تحليل قوة الاتجاه"""
    if len(df) < 30:
        return 0.5
        
    close = df['close'].astype(float)
    
    # اتجاه المتوسطات
    ema_short = close.ewm(span=8).mean()
    ema_medium = close.ewm(span=21).mean()
    ema_long = close.ewm(span=50).mean()
    
    # استمرارية الاتجاه
    trend_alignment = 0
    if (last_val(ema_short) > last_val(ema_medium) > last_val(ema_long)):
        trend_alignment = 1
    elif (last_val(ema_short) < last_val(ema_medium) < last_val(ema_long)):
        trend_alignment = -1
        
    # قياس الاستمرارية
    persistence = 0.0
    for i in range(1, min(10, len(ema_short))):
        if (safe_iloc(ema_short, -i) > safe_iloc(ema_medium, -i) > safe_iloc(ema_long, -i)):
            persistence += 0.1
        elif (safe_iloc(ema_short, -i) < safe_iloc(ema_medium, -i) < safe_iloc(ema_long, -i)):
            persistence += 0.1
            
    return min(persistence, 1.0)

def enhanced_trade_mode_detection(df):
    """كشف محسن لنمط التداول"""
    if len(df) < 50:
        return "scalp"
    
    # مؤشرات متعددة للتمييز
    adx = compute_indicators(df).get('adx', 0)
    volatility = calculate_advanced_volatility(df)
    trend_persistence = analyze_trend_strength(df)
    volume_profile = advanced_volume_analysis(df)
    
    # نظام نقاط للتمييز
    score = 0
    
    # قوة الاتجاه
    if adx > 25:
        score += 3
    elif adx > 20:
        score += 2
    elif adx > 15:
        score += 1
        
    # استمرارية الاتجاه
    if trend_persistence > 0.7:
        score += 2
    elif trend_persistence > 0.5:
        score += 1
        
    # نمط الحجم
    if volume_profile.get("bullish") or volume_profile.get("bearish"):
        score += 2
        
    # التقلب
    if volatility < 1.5:  # تقلب منخفض - مناسب للترند
        score += 1
        
    # تحديد النمط
    if score >= 6:
        return "trend"
    elif score >= 3:
        return "swing"
    else:
        return "scalp"

# =================== ACTIVE COUNCIL MANAGEMENT ===================
def enhanced_active_council(df, state):
    """مجلس إدارة معزز أثناء الصفقات النشطة"""
    if not state["open"]:
        return enhanced_professional_council(df)
    
    # البيانات الأساسية
    basic_council = enhanced_professional_council(df)
    current_price = last_val(df['close'])
    current_pnl = (current_price - state["entry"]) / state["entry"] * 100 * (1 if state["side"] == "long" else -1)
    
    # مؤشرات مخصصة للصفقات النشطة
    active_indicators = {
        "trade_health": analyze_trade_health(df, state),
        "momentum_quality": assess_momentum_quality(df, state),
        "exit_readiness": calculate_exit_readiness(df, state),
        "profit_potential": estimate_profit_potential(df, state)
    }
    
    # تعزيز قرارات المجلس بناءً على الصفقة النشطة
    enhanced_votes = basic_council.copy()
    
    # تعزيز الأصوات مع صحة الصفقة
    if active_indicators["trade_health"] > 0.7:
        if state["side"] == "long":
            enhanced_votes["score_b"] *= 1.2
        else:
            enhanced_votes["score_s"] *= 1.2
            
    # تعزيز مع جودة الزخم
    if active_indicators["momentum_quality"] > 0.8:
        momentum_boost = 1.3
        enhanced_votes["score_b"] *= momentum_boost
        enhanced_votes["score_s"] *= momentum_boost
        
    return {
        **enhanced_votes,
        "active_indicators": active_indicators,
        "current_pnl": current_pnl,
        "trade_duration": time.time() - active_tracker.entry_time if active_tracker.entry_time else 0
    }

def analyze_trade_health(df, state):
    """تحليل صحة الصفقة الحالية"""
    current_price = last_val(df['close'])
    entry = state["entry"]
    current_pnl = (current_price - entry) / entry * 100 * (1 if state["side"] == "long" else -1)
    
    # عوامل الصحة
    health_score = 0.0
    
    # الربح الحالي
    if current_pnl > 0:
        health_score += 0.4 * min(current_pnl / 2.0, 1.0)  # حتى 2% ربح
    
    # قوة الاتجاه
    trend_strength = analyze_trend_strength(df)
    health_score += 0.3 * trend_strength
    
    # استقرار السعر
    volatility = calculate_advanced_volatility(df)
    stability_bonus = max(0, 1.0 - volatility / 3.0)  # تقلب منخفض = صحة أفضل
    health_score += 0.3 * stability_bonus
    
    return min(1.0, health_score)

def assess_momentum_quality(df, state):
    """تقييم جودة الزخم للصفقة الحالية"""
    momentum = enhanced_price_momentum(df)
    trend_strength = analyze_trend_strength(df)
    
    quality_score = 0.0
    
    # قوة الزخم
    momentum_strength = momentum.get("momentum_strength", 0)
    quality_score += 0.4 * min(momentum_strength / 5.0, 1.0)
    
    # استمرارية الزخم
    if momentum.get("trend") == ("bullish" if state["side"] == "long" else "bearish"):
        quality_score += 0.3
        
    # تسارع الزخم
    acceleration = momentum.get("acceleration", 0)
    if (state["side"] == "long" and acceleration > 0) or (state["side"] == "short" and acceleration < 0):
        quality_score += 0.3
        
    return min(1.0, quality_score)

def calculate_exit_readiness(df, state):
    """حساب جاهزية الخروج من الصفقة"""
    current_price = last_val(df['close'])
    entry = state["entry"]
    current_pnl = (current_price - entry) / entry * 100 * (1 if state["side"] == "long" else -1)
    
    readiness = 0.0
    
    # تحقيق أهداف الربح
    if current_pnl >= 1.0:
        readiness += 0.4
        
    # ضعف الزخم
    momentum = enhanced_price_momentum(df)
    if momentum.get("trend") == "neutral":
        readiness += 0.3
        
    # إشارات انعكاس
    council_data = enhanced_professional_council(df)
    if (state["side"] == "long" and council_data["score_s"] > council_data["score_b"]) or \
       (state["side"] == "short" and council_data["score_b"] > council_data["score_s"]):
        readiness += 0.3
        
    return min(1.0, readiness)

def estimate_profit_potential(df, state):
    """تقدير إمكانية الربح المتبقية"""
    current_price = last_val(df['close'])
    entry = state["entry"]
    current_pnl = (current_price - entry) / entry * 100 * (1 if state["side"] == "long" else -1)
    
    potential = 0.0
    
    # قوة الاتجاه المستمر
    trend_strength = analyze_trend_strength(df)
    potential += 0.5 * trend_strength
    
    # المسافة إلى مستويات المقاومة/الدعم
    liquidity_zones = detect_liquidity_zones(df)
    if state["side"] == "long":
        next_resistance = find_next_resistance(current_price, liquidity_zones["sell_liquidity"])
        if next_resistance:
            potential_pnl = (next_resistance - current_price) / current_price * 100
            potential += 0.5 * min(potential_pnl / 3.0, 1.0)
    else:
        next_support = find_next_support(current_price, liquidity_zones["buy_liquidity"])
        if next_support:
            potential_pnl = (current_price - next_support) / current_price * 100
            potential += 0.5 * min(potential_pnl / 3.0, 1.0)
            
    return min(1.0, potential)

def find_next_resistance(current_price, resistance_levels):
    """إيجاد next مستوى مقاومة"""
    if not resistance_levels:
        return None
        
    higher_levels = [level['price'] for level in resistance_levels if level['price'] > current_price]
    return min(higher_levels) if higher_levels else None

def find_next_support(current_price, support_levels):
    """إيجاد next مستوى دعم"""
    if not support_levels:
        return None
        
    lower_levels = [level['price'] for level in support_levels if level['price'] < current_price]
    return max(lower_levels) if lower_levels else None

# =================== PROFESSIONAL TRADE EXECUTION ===================
def execute_professional_trade(side, price, qty, council_data):
    """تنفيذ صفقات محترف مع تحليل متقدم"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    # تحليل SMC للمدخل
    smc_data = council_data.get("advanced_metrics", {}).get("smc_analysis", {})
    market_structure = smc_data.get("market_structure", {})
    
    execution_note = ""
    if market_structure.get("bos_bullish") and side == "buy":
        execution_note = " | 🚀 BOS صاعد"
    elif market_structure.get("bos_bearish") and side == "sell":
        execution_note = " | 💥 BOS هابط"
    
    # تحليل Order Blocks
    order_blocks = smc_data.get("order_blocks", {})
    current_price = price
    ob_note = ""
    
    for ob in order_blocks.get("bullish_ob", []):
        if ob['low'] <= current_price <= ob['high']:
            ob_note = f" | 🟢 OB:{ob['strength']:.1f}%"
            break
    
    for ob in order_blocks.get("bearish_ob", []):
        if ob['low'] <= current_price <= ob['high']:
            ob_note = f" | 🔴 OB:{ob['strength']:.1f}%"
            break

    votes = council_data
    print(f"🎯 EXECUTE PROFESSIONAL: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"votes={votes['b']}/{votes['s']} score={votes['score_b']:.1f}/{votes['score_s']:.1f}"
          f"{execution_note}{ob_note}", flush=True)

    try:
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ EXECUTED PROFESSIONAL: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
        
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

# =================== POSITION MANAGEMENT ===================
def close_market_strict(reason=""):
    """إغلاق صارم للصفقة"""
    global STATE
    
    if not STATE["open"]:
        log_i("لا توجد صفقة مفتوحة للإغلاق")
        return True

    side = "sell" if STATE["side"] == "long" else "buy"
    qty = STATE["qty"]
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للإغلاق")
        return False

    log_i(f"🔄 إغلاق صارم: {qty:.4f} {reason}")

    for attempt in range(CLOSE_RETRY_ATTEMPTS):
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                params = exchange_specific_params(side, is_close=True)
                ex.create_order(SYMBOL, "market", side, qty, None, params)
            
            log_g(f"✅ تم الإغلاق: {qty:.4f} {reason}")
            
            # تحديث الحالة
            STATE.update({
                "open": False,
                "side": None,
                "entry": None,
                "qty": 0.0,
                "pnl": 0.0,
                "bars": 0,
                "trail": None,
                "breakeven": None,
                "tp1_done": False,
                "highest_profit_pct": 0.0,
                "profit_targets_achieved": 0,
                "trade_type": None,
                "trade_size_category": "small",
                "opened_at": None,
                "peak_profit": 0.0,
                "max_drawdown": 0.0,
                "adjustment_count": 0,
                "last_adjustment_time": 0
            })
            
            save_state(STATE)
            return True
            
        except Exception as e:
            log_w(f"⚠️ محاولة إغلاق فاشلة {attempt+1}: {e}")
            time.sleep(CLOSE_VERIFY_WAIT_S)
    
    log_e("❌ فشل جميع محاولات الإغلاق")
    return False

# =================== MAIN TRADING LOGIC ===================
STATE = {}
compound_pnl = 0.0
wait_for_next_signal_side = None
FAST_TRADE_ENABLED = True

def main_loop():
    """الحلقة الرئيسية للتداول مع إصلاحات التوافق"""
    global STATE, compound_pnl
    
    # التحقق من بيئة التنفيذ
    verify_execution_environment()
    
    # تهيئة الحالة
    STATE = initialize_state()
    
    last_hourly_check = 0
    last_trade_time = 0
    
    log_banner("بدء التشغيل - نظام التداول الاحترافي")
    
    while True:
        try:
            current_time = time.time()
            
            # جلب البيانات
            df = fetch_ohlcv()
            if df is None or len(df) < 100:
                log_w("بيانات غير كافية، إعادة المحاولة...")
                time.sleep(BASE_SLEEP)
                continue
            
            # تأكد من توافق البيانات
            df = ensure_dataframe_compatibility(df)
            if df is None:
                log_w("مشكلة في تنسيق البيانات، إعادة المحاولة...")
                time.sleep(BASE_SLEEP)
                continue
            
            current_price = last_val(df['close'])
            
            # تحديث حالة السوق
            market_data = {
                "price": current_price,
                "timestamp": current_time
            }
            
            # مجلس الإدارة المحسن
            council_data = enhanced_professional_council(df)
            
            if STATE["open"]:
                # إدارة الصفقة المفتوحة بالنظام فائق الذكاء
                management_decision = manage_after_entry_super_intelligent_enhanced(
                    STATE, df, market_data
                )
                
                # تحديث تتبع المجلس النشط
                current_pnl = (current_price - STATE["entry"]) / STATE["entry"] * 100 * (1 if STATE["side"] == "long" else -1)
                active_tracker.update_tracking(current_pnl, council_data, {})
                
            else:
                # فرص دخول جديدة
                if current_time - last_trade_time > COOLDOWN_SECS_AFTER_CLOSE:
                    entry_decision = evaluate_entry_opportunity(df, council_data)
                    
                    if entry_decision["action"] in ["buy", "sell"]:
                        execute_entry_decision(entry_decision, df, council_data)
                        last_trade_time = current_time
            
            # حفظ الحالة بشكل دوري
            if current_time % 30 < BASE_SLEEP:
                save_state(STATE)
            
            # النوم حسب الوضع
            sleep_time = NEAR_CLOSE_S if STATE["open"] else BASE_SLEEP
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            log_banner("إيقاف بطلب من المستخدم")
            break
        except Exception as e:
            log_e(f"خطأ في الحلقة الرئيسية: {e}")
            traceback.print_exc()
            time.sleep(BASE_SLEEP * 2)

def fetch_ohlcv():
    """جلب بيانات OHLCV مع معالجة محسنة"""
    try:
        since = ex.milliseconds() - 1000 * 60 * 60 * 24 * 3  # 3 أيام
        ohlcv = ex.fetch_ohlcv(SYMBOL, INTERVAL, since=since, limit=500)
        
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        # تحويل الأعمدة إلى float بشكل صريح
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # إزالة أي صفوف تحتوي على قيم NaN
        df = df.dropna()
        
        return df
    except Exception as e:
        log_e(f"فشل جلب البيانات: {e}")
        return None

def evaluate_entry_opportunity(df, council_data):
    """تقييم فرص الدخول"""
    current_price = last_val(df['close'])
    indicators = compute_indicators(df)
    
    # شروط الدخول الأساسية
    if indicators.get('adx', 0) < ADX_GATE:
        return {"action": "wait", "reason": "ADX منخفض"}
    
    # تحليل المناطق الذهبية
    gz_analysis = golden_zone_check(df, indicators)
    
    # قرارات المجلس
    votes = council_data
    confidence_threshold = ULTIMATE_MIN_CONFIDENCE
    
    entry_signal = None
    reason = ""
    
    if (votes['score_b'] >= confidence_threshold and 
        votes['score_b'] > votes['score_s'] + 1.0):
        entry_signal = "buy"
        reason = f"مجلس قوي: {votes['score_b']:.1f}/{votes['score_s']:.1f}"
        
    elif (votes['score_s'] >= confidence_threshold and 
          votes['score_s'] > votes['score_b'] + 1.0):
        entry_signal = "sell" 
        reason = f"مجلس قوي: {votes['score_s']:.1f}/{votes['score_b']:.1f}"
    
    # الجمع بين المناطق الذهبية وقرارات المجلس
    if gz_analysis["ok"] and entry_signal:
        reason += f" + منطقة ذهبية: {gz_analysis['score']:.1f}"
    
    return {
        "action": entry_signal or "wait",
        "reason": reason or "لا توجد إشارة قوية",
        "side": entry_signal,
        "price": current_price,
        "council_data": council_data,
        "golden_zone": gz_analysis
    }

def execute_entry_decision(decision, df, council_data):
    """تنفيذ قرار الدخول"""
    side = decision["side"]
    price = decision["price"]
    
    # حساب الكمية
    balance = fetch_balance()
    if balance <= 0:
        log_e("❌ رصيد غير كافٍ")
        return
    
    risk_amount = balance * RISK_ALLOC / 100.0
    atr = compute_indicators(df).get('atr', price * 0.01)
    stop_distance = atr * 1.5
    
    # كمية التداول
    qty = (risk_amount / stop_distance) * LEVERAGE
    qty = safe_qty(qty)
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة")
        return
    
    # تنفيذ الصفقة
    success = execute_professional_trade(side, price, qty, council_data)
    
    if success:
        # تحديث الحالة
        STATE.update({
            "open": True,
            "side": side,
            "entry": price,
            "qty": qty,
            "pnl": 0.0,
            "bars": 0,
            "trail": None,
            "breakeven": None,
            "tp1_done": False,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "trade_type": "pending_classification",
            "trade_size_category": "medium",
            "opened_at": time.time(),
            "peak_profit": 0.0,
            "max_drawdown": 0.0,
            "adjustment_count": 0,
            "last_adjustment_time": 0
        })
        
        # بدء التتبع النشط
        active_tracker.start_trade(side, price, council_data)
        
        log_g(f"🎯 صفقة جديدة: {side} {qty:.4f} @ {price:.6f}")
        save_state(STATE)

def fetch_balance():
    """جلب الرصيد"""
    try:
        if MODE_LIVE:
            balance = ex.fetch_balance()
            return float(balance['total']['USDT'])
        else:
            return 1000.0  # رصيد تجريبي
    except Exception as e:
        log_w(f"فشل جلب الرصيد: {e}")
        return 1000.0

def price_now():
    """الحصول على السعر الحالي"""
    try:
        ticker = ex.fetch_ticker(SYMBOL)
        return float(ticker['last'])
    except Exception as e:
        log_w(f"فشل جلب السعر: {e}")
        return 0.0

def resume_open_position_enhanced(exchange, symbol, state):
    """استئناف الصفقة المفتوحة"""
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if (abs(float(pos['contracts'])) > 0 and 
                pos['symbol'] == symbol.replace('/', '')):
                state.update({
                    "in_position": True,
                    "side": "long" if float(pos['contracts']) > 0 else "short",
                    "position_qty": abs(float(pos['contracts'])),
                    "entry_price": float(pos['entryPrice']),
                    "opened_at": time.time() - 3600  # تقديري
                })
                log_g(f"🔄 تم استئناف الصفقة: {state['side']} {state['position_qty']} @ {state['entry_price']}")
                break
    except Exception as e:
        log_w(f"استئناف الصفقة: {e}")
    return state

# =================== API / KEEPALIVE ===================
app = Flask(__name__)

@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ Council ELITE Bot ENHANCED — {SYMBOL} {INTERVAL} — {mode} — Fast Trading Mode"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "COUNCIL_ELITE_ENHANCED", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "fast_trading": FAST_TRADE_ENABLED
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "COUNCIL_ELITE_ENHANCED", "wait_for_next_signal": wait_for_next_signal_side,
        "fast_trading": FAST_TRADE_ENABLED
    }), 200

def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent": "rf-live-bot/keepalive"})
    log_i(f"KEEPALIVE every 50s → {url}")
    while True:
        try:
            sess.get(url, timeout=8)
        except Exception:
            pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("COUNCIL ELITE ENHANCED INIT")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position_enhanced(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  COUNCIL_ELITE_ENHANCED=ENABLED", "yellow"))
    print(colored(f"SMC/ICT: Golden Zones + FVG + BOS + Sweeps + Order Blocks", "yellow"))
    print(colored(f"MANAGEMENT: Smart TP + Smart Exit + Trail Adaptation", "yellow"))
    print(colored(f"FAST TRADING: {'ENABLED' if FAST_TRADE_ENABLED else 'DISABLED'}", "yellow"))
    print(colored(f"EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("Council ELITE ENHANCED service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=main_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
