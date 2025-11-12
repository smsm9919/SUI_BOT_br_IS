# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (Multi-Exchange: BingX & Bybit)
• Council ULTIMATE with Smart Money Concepts & Advanced Indicators
• Golden Entry + Golden Reversal + Wick Exhaustion + Smart Exit
• Dynamic TP ladder + ATR-trailing + Volume Momentum + Liquidity Analysis
• Professional Logging & Dashboard + Multi-Exchange Support
• Footprint Management for Advanced Trade Management
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

# ================== STATIC CONFIG (overrides ENV) ==================
CONFIG = {
    # تشغيل/تعطيل Override (لو False هيقرأ من ENV كالعادة)
    "OVERRIDE": True,

    # السوق والمنصة
    "EXCHANGE": "bingx",           
    "SYMBOL": "SUI/USDT:USDT",   
    "INTERVAL": "15m",

    # إدارة رأس المال
    "LEVERAGE": 10,
    "RISK_ALLOC": 0.60,

    # وضع المراكز
    "POSITION_MODE": "oneway",

    # Range Filter
    "RF_PERIOD": 20,
    "RF_MULT": 3.5,
    "RF_LIVE_ONLY": True,
    "RF_HYST_BPS": 8.0,

    # Footprint Management
    "FP_MGMT_ENABLE": 1,
    "FP_MGMT_WEIGHT": 1.6,
    "FP_MGMT_DELTA_Z": 1.2,
    "FP_MGMT_VOL_Z": 1.0,
    "FP_MGMT_ABSORB_WICK_PCT": 55,
    "FP_MGMT_TRAP_TOL_BPS": 8,
    "FP_MGMT_HOLD_SCORE": 1.8,
    "FP_MGMT_TIGHTEN_SCORE": 1.6,
    "FP_MGMT_PARTIAL_SCORE": 2.0,
    "FP_MGMT_STRICT_SCORE": 2.6,

    # عتبات المجلس
    "ULTIMATE_MIN_CONFIDENCE": 7.0,

    # أخرى
    "FINAL_CHUNK_QTY": 2.0,
    "MAX_SPREAD_BPS": 6.0,
}
# ===================================================================

# دالة القراءة من CONFIG أولاً ثم ENV
def cfg(name, env_key=None, cast=lambda v: v, default=None):
    """CONFIG > ENV > default"""
    if CONFIG.get("OVERRIDE", False) and (name in CONFIG):
        val = CONFIG[name]
        return cast(val) if val is not None else cast(default)
    
    if env_key:
        env_val = os.getenv(env_key)
        if env_val is not None and env_val != "":
            try:
                return cast(env_val)
            except Exception:
                pass
    return cast(default)

# =================== FOOTPRINT MANAGEMENT PATCH ===================
def _zscore(s: pd.Series, win=20):
    mu = s.rolling(win, min_periods=1).mean()
    sd = s.rolling(win, min_periods=1).std().replace(0, np.nan)
    return ((s - mu) / sd).fillna(0)

def _prep_ranges(df: pd.DataFrame):
    rng  = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    df["uw_pct"] = ((df["high"] - df[["close","open"]].max(axis=1)) / rng * 100).fillna(0)
    df["lw_pct"] = ((df[["close","open"]].min(axis=1) - df["low"]) / rng * 100).fillna(0)
    df["dir"]    = np.where(df["close"] >= df["open"], "up", "down")
    return df

def footprint_mgmt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """ يحسب قياسات Footprint على OHLCV مغلق — لا يحتاج Level2 """
    df = _prep_ranges(df.copy())
    df["delta_proxy"] = np.sign(df["close"] - df["open"]) * df["volume"]
    df["delta_z"]     = _zscore(df["delta_proxy"], win=20)
    df["vol_z"]       = _zscore(df["volume"], win=20)

    wick    = float(os.getenv("FP_MGMT_ABSORB_WICK_PCT","55"))
    vol_gate= float(os.getenv("FP_MGMT_VOL_Z","1.0"))
    df["abs_top"] = ((df["uw_pct"]>=wick) & (df["close"]<df["open"]) & (df["vol_z"]>=vol_gate)).astype(int)
    df["abs_bot"] = ((df["lw_pct"]>=wick) & (df["close"]>df["open"]) & (df["vol_z"]>=vol_gate)).astype(int)

    tol     = float(os.getenv("FP_MGMT_TRAP_TOL_BPS","8"))/10000.0
    hh      = df["high"].rolling(30, min_periods=2).max().shift(1)
    ll      = df["low"].rolling(30, min_periods=2).min().shift(1)
    df["trap_buy"]  = ((df["low"] < ll) & (df["close"] > (ll*(1+tol)))).astype(int)   # فخ بائعين → BUY
    df["trap_sell"] = ((df["high"]> hh) & (df["close"] < (hh*(1-tol)))).astype(int)   # فخ مشترين → SELL
    return df

def footprint_mgmt_signal(row: pd.Series, side: str):
    """ يرجّع (action, score, tags) لإدارة الصفقة فقط """
    if int(os.getenv("FP_MGMT_ENABLE","1")) != 1:
        return ("none", 0.0, ["fp_mgmt_off"])

    W   = float(os.getenv("FP_MGMT_WEIGHT","1.6"))
    dz  = float(row.get("delta_z",0.0))
    vz  = float(row.get("vol_z",0.0))
    dz_gate = float(os.getenv("FP_MGMT_DELTA_Z","1.2"))
    vz_gate = float(os.getenv("FP_MGMT_VOL_Z","1.0"))

    score_for, score_against = 0.0, 0.0
    tags = []

    # دلتا + حجم (قوة تدفّق)
    if abs(dz) >= dz_gate and vz >= vz_gate:
        if (dz>0 and side=="buy") or (dz<0 and side=="sell"):
            score_for += (1.0 + min(abs(dz)/3.0, 1.0)) * W;  tags.append("delta_with")
        else:
            score_against += (1.0 + min(abs(dz)/3.0, 1.0)) * W; tags.append("delta_against")

    # امتصاص
    if side=="buy"  and row.get("abs_top",0)==1: score_against += 1.0*W; tags.append("absorption_top")
    if side=="sell" and row.get("abs_bot",0)==1: score_against += 1.0*W; tags.append("absorption_bot")
    if side=="buy"  and row.get("abs_bot",0)==1: score_for     += 0.8*W; tags.append("absorption_support")
    if side=="sell" and row.get("abs_top",0)==1: score_for     += 0.8*W; tags.append("absorption_resist")

    # فِخاخ
    if side=="buy"  and row.get("trap_sell",0)==1: score_against += 0.9*W; tags.append("trap_sell")
    if side=="sell" and row.get("trap_buy",0)==1:  score_against += 0.9*W; tags.append("trap_buy")
    if side=="buy"  and row.get("trap_buy",0)==1:  score_for     += 0.6*W; tags.append("trap_buy_support")
    if side=="sell" and row.get("trap_sell",0)==1: score_for     += 0.6*W; tags.append("trap_sell_resist")

    # خريطة القرار
    hold_thr    = float(os.getenv("FP_MGMT_HOLD_SCORE","1.8"))
    tighten_thr = float(os.getenv("FP_MGMT_TIGHTEN_SCORE","1.6"))
    partial_thr = float(os.getenv("FP_MGMT_PARTIAL_SCORE","2.0"))
    strict_thr  = float(os.getenv("FP_MGMT_STRICT_SCORE","2.6"))

    action = "none"
    score  = round(max(score_for, score_against), 2)

    if score_against >= strict_thr:   action = "close"
    elif score_against >= partial_thr: action = "partial"
    elif score_against >= tighten_thr: action = "tighten"
    elif score_for    >= hold_thr:     action = "hold"

    return (action, score, tags)

# إعدادات Footprint Management
FP_MGMT_SETTINGS = {
    "enable": int(os.getenv("FP_MGMT_ENABLE", "1")),
    "weight": float(os.getenv("FP_MGMT_WEIGHT", "1.6")),
    "delta_z": float(os.getenv("FP_MGMT_DELTA_Z", "1.2")),
    "vol_z": float(os.getenv("FP_MGMT_VOL_Z", "1.0")),
    "absorb_wick_pct": float(os.getenv("FP_MGMT_ABSORB_WICK_PCT", "55")),
    "trap_tol_bps": float(os.getenv("FP_MGMT_TRAP_TOL_BPS", "8")),
    "hold_score": float(os.getenv("FP_MGMT_HOLD_SCORE", "1.8")),
    "tighten_score": float(os.getenv("FP_MGMT_TIGHTEN_SCORE", "1.6")),
    "partial_score": float(os.getenv("FP_MGMT_PARTIAL_SCORE", "2.0")),
    "strict_score": float(os.getenv("FP_MGMT_STRICT_SCORE", "2.6"))
}

FP_MGMT_ENABLED = FP_MGMT_SETTINGS["enable"] == 1

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

def apply_footprint_management(df, current_side, state):
    """تطبيق Footprint Management على الصفقة الحالية"""
    if not FP_MGMT_ENABLED or not state["open"]:
        return {"action": "none", "reason": "غير مفعل أو لا توجد صفقة"}
    
    try:
        # استخدام البيانات المغلقة فقط (بدون الشمعة الحية)
        df_closed = df.iloc[:-1].copy() if len(df) > 1 else df.copy()
        
        if len(df_closed) < 30:
            return {"action": "none", "reason": "بيانات غير كافية"}
        
        # حساب مقاييس Footprint
        df_fp = footprint_mgmt_metrics(df_closed)
        row = df_fp.iloc[-1]
        
        # الحصول على الإشارة
        action, score, tags = footprint_mgmt_signal(row=row, side=current_side)
        
        # التسجيل
        log_i(f"🧠 FP-MGMT | side={current_side} | action={action} | score={score} | "
              f"Δz={row.get('delta_z',0):.2f} | vol_z={row.get('vol_z',0):.2f} | "
              f"tags={','.join(tags) if tags else '—'}")
        
        return {
            "action": action,
            "score": score,
            "tags": tags,
            "delta_z": row.get('delta_z', 0),
            "vol_z": row.get('vol_z', 0),
            "abs_top": row.get('abs_top', 0),
            "abs_bot": row.get('abs_bot', 0),
            "trap_buy": row.get('trap_buy', 0),
            "trap_sell": row.get('trap_sell', 0)
        }
        
    except Exception as e:
        log_w(f"FP-MGMT error: {e}")
        return {"action": "none", "reason": f"خطأ: {str(e)}"}

# =================== ENV / MODE ===================
# Exchange Selection
EXCHANGE_NAME = cfg("EXCHANGE", "EXCHANGE", str, "bingx").lower()

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
BOT_VERSION = f"SUI Council PROFESSIONAL v7.0 — {EXCHANGE_NAME.upper()} Multi-Exchange"
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

# === Footprint Management Settings ===
FP_MGMT_ENABLE = cfg("FP_MGMT_ENABLE", "FP_MGMT_ENABLE", int, 1)
FP_MGMT_WEIGHT = cfg("FP_MGMT_WEIGHT", "FP_MGMT_WEIGHT", float, 1.6)
FP_MGMT_DELTA_Z = cfg("FP_MGMT_DELTA_Z", "FP_MGMT_DELTA_Z", float, 1.2)
FP_MGMT_VOL_Z = cfg("FP_MGMT_VOL_Z", "FP_MGMT_VOL_Z", float, 1.0)
FP_MGMT_ABSORB_WICK_PCT = cfg("FP_MGMT_ABSORB_WICK_PCT", "FP_MGMT_ABSORB_WICK_PCT", float, 55)
FP_MGMT_TRAP_TOL_BPS = cfg("FP_MGMT_TRAP_TOL_BPS", "FP_MGMT_TRAP_TOL_BPS", float, 8)
FP_MGMT_HOLD_SCORE = cfg("FP_MGMT_HOLD_SCORE", "FP_MGMT_HOLD_SCORE", float, 1.8)
FP_MGMT_TIGHTEN_SCORE = cfg("FP_MGMT_TIGHTEN_SCORE", "FP_MGMT_TIGHTEN_SCORE", float, 1.6)
FP_MGMT_PARTIAL_SCORE = cfg("FP_MGMT_PARTIAL_SCORE", "FP_MGMT_PARTIAL_SCORE", float, 2.0)
FP_MGMT_STRICT_SCORE = cfg("FP_MGMT_STRICT_SCORE", "FP_MGMT_STRICT_SCORE", float, 2.6)

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
SYMBOL     = cfg("SYMBOL", "SYMBOL", str, "SUI/USDT:USDT")
INTERVAL   = cfg("INTERVAL", "INTERVAL", str, "15m")
LEVERAGE   = cfg("LEVERAGE", "LEVERAGE", int, 10)
RISK_ALLOC = cfg("RISK_ALLOC", "RISK_ALLOC", float, 0.60)
POSITION_MODE = cfg("POSITION_MODE", "POSITION_MODE", str, "oneway")

# RF Settings - Optimized for SUI
RF_SOURCE = "close"
RF_PERIOD = cfg("RF_PERIOD", "RF_PERIOD", int, 18)
RF_MULT   = cfg("RF_MULT", "RF_MULT", float, 3.0)
RF_LIVE_ONLY = cfg("RF_LIVE_ONLY", "RF_LIVE_ONLY", lambda v: str(v).lower() in ("1","true","yes"), True)
RF_HYST_BPS  = cfg("RF_HYST_BPS", "RF_HYST_BPS", float, 6.0)

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = cfg("MAX_SPREAD_BPS", "MAX_SPREAD_BPS", float, 6.0)

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard
FINAL_CHUNK_QTY = cfg("FINAL_CHUNK_QTY", "FINAL_CHUNK_QTY", float, 2.0)
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
ULTIMATE_MIN_CONFIDENCE = cfg("ULTIMATE_MIN_CONFIDENCE", "ULTIMATE_MIN_CONFIDENCE", float, 7.0)
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

def json_sanitize(obj):
    """تحويل الكائنات غير القابلة للتسلسل JSON إلى أنواع قابلة للتسلسل"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    
    # numpy types
    if hasattr(obj, 'dtype'):
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
    
    # pandas Series
    if hasattr(obj, 'iloc'):
        return float(obj.iloc[-1]) if len(obj) > 0 else 0.0
    
    # dict
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    
    # default: convert to string
    return str(obj)

def save_state_safe(state: dict):
    """حفظ آمن للحالة مع معالجة جميع أنواع البيانات"""
    try:
        state["ts"] = int(time.time())
        sanitized_state = json_sanitize(state)
        
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(sanitized_state, f, ensure_ascii=False, indent=2)
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

# استبدال دالة save_state القديمة
save_state = save_state_safe

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
    print(f"⚡ RF SETTINGS: period={RF_PERIOD} | mult={RF_MULT}", flush=True)
    
    # عرض إعدادات CONFIG
    print(f"🎛️  CONFIG OVERRIDE: {CONFIG.get('OVERRIDE', False)}", flush=True)
    if CONFIG.get("OVERRIDE", False):
        print(f"   ↳ MODE: STATIC CONFIG", flush=True)
    else:
        print(f"   ↳ MODE: ENVIRONMENT VARIABLES", flush=True)
    
    # Footprint Management
    if FP_MGMT_ENABLED:
        print(f"👣 FOOTPRINT MANAGEMENT: ENABLED", flush=True)
        print(f"   ↳ الوزن: {FP_MGMT_WEIGHT} | Delta-Z: {FP_MGMT_DELTA_Z}", flush=True)
    else:
        print(f"👣 FOOTPRINT MANAGEMENT: DISABLED", flush=True)
    
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
        if (safe_iloc(rsi, -2) <= safe_iloc(rsi_ma, -2)) and (safe_iloc(rsi) > safe_iloc(rsi_ma)):
            cross = "bull"
        elif (safe_iloc(rsi, -2) >= safe_iloc(rsi_ma, -2)) and (safe_iloc(rsi) < safe_iloc(rsi_ma)):
            cross = "bear"
    
    above = (rsi > rsi_ma)
    below = (rsi < rsi_ma)
    persist_bull = above.tail(RSI_TREND_PERSIST).all() if len(above) >= RSI_TREND_PERSIST else False
    persist_bear = below.tail(RSI_TREND_PERSIST).all() if len(below) >= RSI_TREND_PERSIST else False
    
    current_rsi = safe_iloc(rsi)
    in_chop = RSI_NEUTRAL_BAND[0] <= current_rsi <= RSI_NEUTRAL_BAND[1]
    
    return {
        "rsi": current_rsi,
        "rsi_ma": safe_iloc(rsi_ma),
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

def compute_advanced_momentum(df):
    """زخم متقدم مع تحليل السرعة والتسارع"""
    if len(df) < 30:
        return {"momentum": 0, "acceleration": 0, "velocity": 0, "trend": "neutral", "strength": 0}
    
    close = df['close'].astype(float)
    
    # السرعة (التغير في السعر)
    velocity = close.pct_change(5).iloc[-1] * 100
    
    # التسارع (التغير في السرعة)
    acceleration = 0
    if len(close) >= 6:
        acceleration = (close.pct_change(5).iloc[-1] - close.pct_change(5).iloc[-2]) * 100
    
    # الزخم المرجح بالحجم
    volume = df['volume'].astype(float)
    volume_weighted_momentum = (close.pct_change(3) * volume.rolling(3).mean()).iloc[-1] * 100
    
    # تحليل الاتجاه
    if velocity > 0.5 and acceleration > 0.1:
        trend = "strong_bullish"
    elif velocity > 0.2:
        trend = "bullish"
    elif velocity < -0.5 and acceleration < -0.1:
        trend = "strong_bearish"
    elif velocity < -0.2:
        trend = "bearish"
    else:
        trend = "neutral"
    
    return {
        "momentum": volume_weighted_momentum,
        "acceleration": acceleration,
        "velocity": velocity,
        "trend": trend,
        "strength": abs(volume_weighted_momentum)
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

# =================== PROFESSIONAL COUNCIL WITH SMC ===================
def ultimate_council_professional(df):
    """مجلس الإدارة المحترف مع SMC والمؤشرات المتقدمة"""
    try:
        # البيانات الأساسية
        current_price = last_val(df['close'])
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        candles = compute_candles(df)
        
        # المؤشرات المتقدمة الحالية
        volume_momentum = enhanced_volume_momentum(df)
        stoch_rsi = stochastic_rsi_enhanced(df)
        pivots = dynamic_pivot_points(df)
        trend_indicator = dynamic_trend_indicator(df)
        footprint = advanced_footprint_analysis(df, current_price)
        
        # المؤشرات الجديدة المتقدمة
        macd = compute_macd(df)
        vwap = compute_vwap(df)
        advanced_momentum = compute_advanced_momentum(df)
        
        # تحليل SMC المتقدم - مع معالجة الأخطاء
        try:
            liquidity_zones = detect_liquidity_zones(df)
        except Exception as e:
            log_w(f"SMC liquidity_zones error: {e}")
            liquidity_zones = {"buy_liquidity": [], "sell_liquidity": []}
            
        try:
            fvg_zones = detect_fvg(df)
        except Exception as e:
            log_w(f"SMC fvg_zones error: {e}")
            fvg_zones = {"bullish_fvg": [], "bearish_fvg": []}
            
        try:
            market_structure = detect_market_structure(df)
        except Exception as e:
            log_w(f"SMC market_structure error: {e}")
            market_structure = {"trend": "neutral", "bos_bullish": False, "bos_bearish": False, 
                               "choch_bullish": False, "choch_bearish": False, "liquidity_sweep": False}
            
        try:
            order_blocks = detect_order_blocks(df)
        except Exception as e:
            log_w(f"SMC order_blocks error: {e}")
            order_blocks = {"bullish_ob": [], "bearish_ob": []}
        
        # نظام التصويت المحترف
        votes_buy = 0
        votes_sell = 0
        confidence_buy = 0.0
        confidence_sell = 0.0
        detailed_logs = []
        
        # === 1. تحليل SMC وهيكل السوق (أعلى وزن) ===
        
        # تحليل الاتجاه الهيكلي
        if market_structure["trend"] == "bullish":
            votes_buy += 3
            confidence_buy += 2.5
            detailed_logs.append("📊 هيكل السوق صاعد")
        
        if market_structure["trend"] == "bearish":
            votes_sell += 3
            confidence_sell += 2.5
            detailed_logs.append("📊 هيكل السوق هابط")
        
        # Break of Structure
        if market_structure["bos_bullish"]:
            votes_buy += 4
            confidence_buy += 3.0
            detailed_logs.append("🚀 BOS صاعد - اختراق هيكل")
        
        if market_structure["bos_bearish"]:
            votes_sell += 4
            confidence_sell += 3.0
            detailed_logs.append("💥 BOS هابط - اختراق هيكل")
        
        # Liquidity Sweep
        if market_structure.get("liquidity_sweep"):
            # مسح السيولة عادةً يسبق الانعكاس
            sweep_direction = "buy" if market_structure.get("bos_bullish") else "sell"
            if sweep_direction == "buy":
                votes_buy += 2
                confidence_buy += 1.5
                detailed_logs.append("💰 مسح سيولة شراء")
            else:
                votes_sell += 2
                confidence_sell += 1.5
                detailed_logs.append("💰 مسح سيولة بيع")
        
        # === 2. تحليل Order Blocks ===
        current_time = df['time'].iloc[-1] if hasattr(df['time'], 'iloc') else df['time'][-1]
        
        # تحقق من Order Blocks القريبة
        for ob in order_blocks["bullish_ob"]:
            if (ob['low'] <= current_price <= ob['high'] and 
                (current_time - ob['time']) / 1000 < 86400):  # within 24 hours
                votes_buy += 3
                confidence_buy += 2.0
                detailed_logs.append(f"🟢 Order Block شراء قوي: {ob['strength']:.1f}%")
        
        for ob in order_blocks["bearish_ob"]:
            if (ob['low'] <= current_price <= ob['high'] and 
                (current_time - ob['time']) / 1000 < 86400):  # within 24 hours
                votes_sell += 3
                confidence_sell += 2.0
                detailed_logs.append(f"🔴 Order Block بيع قوي: {ob['strength']:.1f}%")
        
        # === 3. تحليل FVG ===
        for fvg in fvg_zones["bullish_fvg"]:
            if fvg['low'] <= current_price <= fvg['high']:
                votes_buy += 2
                confidence_buy += 1.5
                detailed_logs.append(f"📈 FVG صاعد: {fvg['strength']:.2f}%")
        
        for fvg in fvg_zones["bearish_fvg"]:
            if fvg['low'] <= current_price <= fvg['high']:
                votes_sell += 2
                confidence_sell += 1.5
                detailed_logs.append(f"📉 FVG هابط: {fvg['strength']:.2f}%")
        
        # === 4. تحليل مناطق السيولة ===
        for level in liquidity_zones["buy_liquidity"]:
            if abs(current_price - level['price']) / current_price <= LIQUIDITY_ZONE_PROXIMITY:  # within proximity
                votes_buy += 2
                confidence_buy += 1.0
                detailed_logs.append(f"🏦 دعم قوي: {level['price']:.6f}")
        
        for level in liquidity_zones["sell_liquidity"]:
            if abs(current_price - level['price']) / current_price <= LIQUIDITY_ZONE_PROXIMITY:  # within proximity
                votes_sell += 2
                confidence_sell += 1.0
                detailed_logs.append(f"🏦 مقاومة قوية: {level['price']:.6f}")
        
        # === 5. المؤشرات التقنية المتقدمة ===
        
        # MACD
        if macd["crossover"] == "bullish" and macd["above_zero"]:
            votes_buy += 2
            confidence_buy += 1.5
            detailed_logs.append("📈 MACD إيجابي قوي")
        elif macd["crossover"] == "bullish":
            votes_buy += 1
            confidence_buy += 1.0
            detailed_logs.append("📈 MACD إيجابي")
        
        if macd["crossover"] == "bearish" and not macd["above_zero"]:
            votes_sell += 2
            confidence_sell += 1.5
            detailed_logs.append("📉 MACD سلبي قوي")
        elif macd["crossover"] == "bearish":
            votes_sell += 1
            confidence_sell += 1.0
            detailed_logs.append("📉 MACD سلبي")
        
        # VWAP
        if vwap["signal"] == "bullish" and vwap["price_above_vwap"]:
            votes_buy += 1
            confidence_buy += 0.8
            detailed_logs.append("⚡ VWAP إيجابي")
        
        if vwap["signal"] == "bearish" and not vwap["price_above_vwap"]:
            votes_sell += 1
            confidence_sell += 0.8
            detailed_logs.append("⚡ VWAP سلبي")
        
        # الزخم المتقدم
        if advanced_momentum["trend"] == "strong_bullish":
            votes_buy += 2
            confidence_buy += 1.5
            detailed_logs.append(f"🚀 زخم صاعد قوي: {advanced_momentum['momentum']:.2f}%")
        elif advanced_momentum["trend"] == "bullish":
            votes_buy += 1
            confidence_buy += 1.0
        
        if advanced_momentum["trend"] == "strong_bearish":
            votes_sell += 2
            confidence_sell += 1.5
            detailed_logs.append(f"💥 زخم هابط قوي: {advanced_momentum['momentum']:.2f}%")
        elif advanced_momentum["trend"] == "bearish":
            votes_sell += 1
            confidence_sell += 1.0
        
        # === 6. المؤشرات الحالية المعززة ===
        
        # Footprint Analysis
        if footprint.get("ok"):
            fp_bull = footprint.get("footprint_score_bull", 0)
            fp_bear = footprint.get("footprint_score_bear", 0)
            
            if fp_bull > 2.0:
                votes_buy += 3
                confidence_buy += min(2.5, fp_bull)
                detailed_logs.append(f"👣 Footprint صاعد: {fp_bull:.1f}")
            
            if fp_bear > 2.0:
                votes_sell += 3
                confidence_sell += min(2.5, fp_bear)
                detailed_logs.append(f"👣 Footprint هابط: {fp_bear:.1f}")
        
        # Trend Indicator
        if trend_indicator["signal"] in ["strong_buy", "buy"]:
            votes_buy += 2
            confidence_buy += 1.5
        
        if trend_indicator["signal"] in ["strong_sell", "sell"]:
            votes_sell += 2
            confidence_sell += 1.5
        
        # Volume Momentum
        if volume_momentum["trend"] == "bull" and volume_momentum["strength"] > 2.0:
            votes_buy += 2
            confidence_buy += 1.2
        
        if volume_momentum["trend"] == "bear" and volume_momentum["strength"] > 2.0:
            votes_sell += 2
            confidence_sell += 1.2
        
        # === 7. تصفية الإشارات المتضاربة ===
        
        if votes_buy > 0 and votes_sell > 0:
            # أولوية SMC وهيكل السوق
            smc_strength_buy = any("هيكل السوق صاعد" in log or "BOS صاعد" in log for log in detailed_logs)
            smc_strength_sell = any("هيكل السوق هابط" in log or "BOS هابط" in log for log in detailed_logs)
            
            if smc_strength_buy and not smc_strength_sell:
                votes_sell = 0
                confidence_sell = 0
                detailed_logs.append("🔄 ترجيح الشراء (SMC قوي)")
            elif smc_strength_sell and not smc_strength_buy:
                votes_buy = 0
                confidence_buy = 0
                detailed_logs.append("🔄 ترجيح البيع (SMC قوي)")
            elif confidence_buy > confidence_sell:
                votes_sell = 0
                confidence_sell = 0
                detailed_logs.append("🔄 ترجيح الشراء (ثقة أعلى)")
            else:
                votes_buy = 0
                confidence_buy = 0
                detailed_logs.append("🔄 ترجيح البيع (ثقة أعلى)")
        
        # تحديث البيانات بالمؤشرات الجديدة
        ind.update({
            "macd": macd,
            "vwap": vwap,
            "advanced_momentum": advanced_momentum,
            "liquidity_zones": liquidity_zones,
            "fvg_zones": fvg_zones,
            "market_structure": market_structure,
            "order_blocks": order_blocks,
            "professional_votes_buy": votes_buy,
            "professional_votes_sell": votes_sell,
            "professional_confidence_buy": confidence_buy,
            "professional_confidence_sell": confidence_sell
        })
        
        return {
            "b": votes_buy, "s": votes_sell,
            "score_b": confidence_buy, "score_s": confidence_sell,
            "logs": detailed_logs, 
            "ind": ind, 
            "candles": candles,
            "advanced_indicators": {
                "volume_momentum": volume_momentum,
                "stoch_rsi": stoch_rsi,
                "pivots": pivots,
                "trend_indicator": trend_indicator,
                "footprint": footprint,
                "macd": macd,
                "vwap": vwap,
                "advanced_momentum": advanced_momentum,
                "smc_analysis": {
                    "liquidity_zones": liquidity_zones,
                    "fvg_zones": fvg_zones,
                    "market_structure": market_structure,
                    "order_blocks": order_blocks
                }
            }
        }
        
    except Exception as e:
        log_w(f"ultimate_council_professional error: {e}")
        return {"b":0, "s":0, "score_b":0.0, "score_s":0.0, "logs":[], "ind":{}, "candles":{}}

# =================== PROFESSIONAL TRADE MANAGEMENT ===================
def _candle_fp_post_entry_boost(position_side, df, features, last_price, trail, tp_layers, log_func):
    """إدارة الصفقة بناءً على تحليل الشموع و Footprint بعد الدخول"""
    if len(df) < 3:
        return trail, tp_layers, False
    
    try:
        # تحليل الشموع الحالية
        current_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        o, h, l, c = (float(current_candle['open']), float(current_candle['high']), 
                      float(current_candle['low']), float(current_candle['close']))
        po, ph, pl, pc = (float(prev_candle['open']), float(prev_candle['high']), 
                         float(prev_candle['low']), float(prev_candle['close']))
        
        # حساب قوة الشمعة
        body = abs(c - o)
        range_candle = h - l
        body_ratio = body / range_candle if range_candle > 0 else 0
        
        # تحليل Footprint للإدارة
        fp_metrics = footprint_mgmt_metrics(df.iloc[:-1])  # بدون الشمعة الحية
        last_fp = fp_metrics.iloc[-1] if len(fp_metrics) > 0 else {}
        
        # قرارات الإدارة
        hold_tp = False
        tighten_trail = False
        
        # تأجيل جني الأرباح عند الشموع القوية في اتجاه الصفقة
        if body_ratio > 0.7:  # شمعة ماروبوزو
            if (position_side == "long" and c > o) or (position_side == "short" and c < o):
                hold_tp = True
                log_func("🟡 تأجيل جني الأرباح - شمعة قوية في اتجاه الصفقة")
        
        # تشديد وقف الخسارة عند علامات الامتصاص
        if position_side == "long" and last_fp.get('abs_top', 0) == 1:
            tighten_trail = True
            log_func("🟡 تشديد الوقف - امتصاص عند المقاومة")
        elif position_side == "short" and last_fp.get('abs_bot', 0) == 1:
            tighten_trail = True
            log_func("🟡 تشديد الوقف - امتصاص عند الدعم")
        
        # تطبيق القرارات
        if hold_tp and tp_layers:
            # تأجيل جني الأرباح بزيادة المستويات
            tp_layers = [tp * 1.2 for tp in tp_layers]  # زيادة الأهداف 20%
        
        if tighten_trail and trail is not None:
            # تشديد وقف الخسارة
            if position_side == "long":
                trail = max(trail, last_price * 0.995)  # أقرب بـ 0.5%
            else:
                trail = min(trail, last_price * 1.005)
        
        return trail, tp_layers, hold_tp or tighten_trail
        
    except Exception as e:
        log_func(f"⚠️ خطأ في إدارة الصفقة بالشموع: {e}")
        return trail, tp_layers, False

def professional_trade_management(df, state, current_price):
    """إدارة صفقات محترفة مع جني أرباح ديناميكي"""
    if not state["open"] or state["qty"] <= 0:
        return {"action": "hold", "reason": "لا توجد صفقة مفتوحة"}
    
    entry = state["entry"]
    side = state["side"]
    qty = state["qty"]
    
    # حساب الربح/الخسارة
    unrealized_pnl_pct = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
    
    # تحليل السوق الحالي
    market_structure = detect_market_structure(df)
    advanced_momentum = compute_advanced_momentum(df)
    macd = compute_macd(df)
    
    # تحديد أهداف الربح الديناميكية بناءً على ظروف السوق
    if market_structure["trend"] == "strong_bullish" and side == "long":
        # في اتجاه صاعد قوي - أهداف أعلى
        tp_levels = [1.0, 2.0, 3.5]
        tp_fractions = [0.25, 0.35, 0.40]
    elif market_structure["trend"] == "strong_bearish" and side == "short":
        # في اتجاه هابط قوي - أهداف أعلى
        tp_levels = [1.0, 2.0, 3.5]
        tp_fractions = [0.25, 0.35, 0.40]
    elif advanced_momentum["strength"] > 2.0:
        # زخم قوي - أهداف متوسطة
        tp_levels = [0.8, 1.5, 2.5]
        tp_fractions = [0.3, 0.3, 0.4]
    else:
        # ظروف عادية - أهداف محافظة
        tp_levels = [0.6, 1.2, 2.0]
        tp_fractions = [0.4, 0.3, 0.3]
    
    # جني الأرباح عند المستويات
    achieved_tps = state.get("profit_targets_achieved", 0)
    
    if achieved_tps < len(tp_levels) and unrealized_pnl_pct >= tp_levels[achieved_tps]:
        close_fraction = tp_fractions[achieved_tps]
        return {
            "action": "partial_close",
            "reason": f"جني أرباح عند المستوى {achieved_tps + 1} ({tp_levels[achieved_tps]}%)",
            "close_fraction": close_fraction,
            "tp_level": tp_levels[achieved_tps],
            "new_achieved_tps": achieved_tps + 1
        }
    
    # وقف الخسارة المتحرك الذكي
    if unrealized_pnl_pct > 0.5:  # بدء التريل بعد تحقيق ربح 0.5%
        atr = compute_indicators(df).get('atr', 0.001)
        
        if side == "long":
            # في الاتجاه الصاعد، استخدم تريل أكثر مرونة
            trail_distance = atr * (1.2 if unrealized_pnl_pct < 1.5 else 0.8)
            new_trail = current_price - trail_distance
        else:
            # في الاتجاه الهابط، استخدم تريل أكثر مرونة
            trail_distance = atr * (1.2 if unrealized_pnl_pct < 1.5 else 0.8)
            new_trail = current_price + trail_distance
        
        # تحديث التريل فقط إذا كان أفضل
        if state.get("trail") is None:
            state["trail"] = new_trail
        elif (side == "long" and new_trail > state["trail"]) or \
             (side == "short" and new_trail < state["trail"]):
            state["trail"] = new_trail
        
        # التحقق من التريل
        if state.get("trail"):
            if (side == "long" and current_price <= state["trail"]) or \
               (side == "short" and current_price >= state["trail"]):
                return {
                    "action": "close", 
                    "reason": "وقف خسارة متحرك محسن",
                    "trail_price": state["trail"],
                    "pnl_pct": unrealized_pnl_pct
                }
    
    # إغلاق استباقي عند علامات الانعكاس
    if unrealized_pnl_pct > 1.0:  # فقط إذا كان هناك ربح
        reversal_signals = 0
        
        # MACD انعكاس
        if (side == "long" and macd["crossover"] == "bearish") or \
           (side == "short" and macd["crossover"] == "bullish"):
            reversal_signals += 1
        
        # تغير هيكل السوق
        if (side == "long" and market_structure["choch_bearish"]) or \
           (side == "short" and market_structure["choch_bullish"]):
            reversal_signals += 1
        
        # فقدان الزخم
        if (side == "long" and advanced_momentum.get("trend") == "bearish") or \
           (side == "short" and advanced_momentum.get("trend") == "bullish"):
            reversal_signals += 1
        
        if reversal_signals >= 2:
            return {
                "action": "close", 
                "reason": "انعكاس متعدد المؤشرات",
                "reversal_signals": reversal_signals,
                "pnl_pct": unrealized_pnl_pct
            }
    
    return {"action": "hold", "reason": "الاستمرار في الصفقة"}

# =================== PROFESSIONAL EXECUTION ===================
def execute_professional_trade(side, price, qty, council_data):
    """تنفيذ صفقات محترف مع تحليل متقدم"""
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    # تحليل SMC للمدخل
    smc_data = council_data.get("advanced_indicators", {}).get("smc_analysis", {})
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
            ob_note = f" | 🟢 OB قوي: {ob['strength']:.1f}%"
            break
    
    for ob in order_blocks.get("bearish_ob", []):
        if ob['low'] <= current_price <= ob['high']:
            ob_note = f" | 🔴 OB قوي: {ob['strength']:.1f}%"
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
        
        # تسجيل تفاصيل SMC في السجل
        if market_structure.get("bos_bullish") or market_structure.get("bos_bearish"):
            logging.info(f"SMC_ENTRY: {side} | BOS={market_structure.get('bos_bullish') or market_structure.get('bos_bearish')}")
        
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

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
        z = float((last_val(wnd) - mu) / sd)
        trend = "up" if (last_val(cvd_ma) - safe_iloc(cvd_ma, -min(CVD_SMOOTH, len(cvd_ma)))) >= 0 else "down"
        return {"ok": True, "delta_last": last_val(delta), "delta_mean": mu, "delta_z": z,
                "cvd_last": last_val(cvd), "cvd_trend": trend, "spike": abs(z) >= FLOW_SPIKE_Z}
    except Exception as e:
        return {"ok": False, "why": str(e)}

# ========= Unified snapshot emitter =========
def emit_snapshots_with_smc(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    """لوحة تحكم محسنة مع عرض SMC"""
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = ultimate_council_professional(df)  # استخدام المجلس المحترف
        mode = decide_strategy_mode(df)
        
        # تمرير ind بشكل صحيح لـ golden_zone_check
        gz = golden_zone_check(df, cv.get("ind", {}), "buy" if cv["b"]>=cv["s"] else "sell")
        current_price = last_val(df['close'])
        footprint = advanced_footprint_analysis(df, current_price)

        bal = None; cpnl = None
        if callable(balance_fn):
            try: bal = balance_fn()
            except: bal = None
        if callable(pnl_fn):
            try: cpnl = pnl_fn()
            except: cpnl = None

        # عرض SMC في الداشبورد
        smc_note = ""
        smc_data = cv.get("advanced_indicators", {}).get("smc_analysis", {})
        market_structure = smc_data.get("market_structure", {})
        
        if market_structure.get("bos_bullish"):
            smc_note = f" | 🚀 BOS صاعد"
        elif market_structure.get("bos_bearish"):
            smc_note = f" | 💥 BOS هابط"
        
        # عرض Order Blocks
        ob_note = ""
        for ob in smc_data.get("order_blocks", {}).get("bullish_ob", []):
            if ob['low'] <= current_price <= ob['high']:
                ob_note = f" | 🟢 OB:{ob['strength']:.1f}%"
                break
        
        for ob in smc_data.get("order_blocks", {}).get("bearish_ob", []):
            if ob['low'] <= current_price <= ob['high']:
                ob_note = f" | 🔴 OB:{ob['strength']:.1f}%"
                break

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
                f"DI={cv['ind'].get('di_spread',0):.1f}{smc_note}{ob_note}")

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
            
            print(f"🧠 SNAP PROFESSIONAL | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={cv['ind'].get('adx',0):.1f} DI={cv['ind'].get('di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{smc_note}{ob_note}{gz_note}", 
                  flush=True)
            
            print("✅ PROFESSIONAL TRADING BOT LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "footprint": footprint, "wallet": wallet}
    except Exception as e:
        print(f"🟨 Professional AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "footprint": {}, "wallet": ""}

# =================== EXECUTION MANAGER ===================
def open_market_enhanced(side, qty, price):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    df = fetch_ohlcv()
    snap = emit_snapshots_with_smc(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=votes["ind"].get("adx"),
                                   di_plus=votes["ind"].get("plus_di"),
                                   di_minus=votes["ind"].get("minus_di"),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    success = execute_professional_trade(side, price, qty, votes)
    
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
            "hold_tp_until": 0  # حالة تأجيل جني الأرباح
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "gz_snapshot": gz if isinstance(gz, dict) else {},
            "cv_snapshot": votes if isinstance(votes, dict) else {},
            "opened_at": int(time.time()),
            "partial_taken": False,
            "breakeven_armed": False,
            "trail_active": False,
            "trail_tightened": False,
            "hold_tp_until": 0
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

    return {
        "rsi": last_val(rsi), 
        "plus_di": last_val(plus_di),
        "minus_di": last_val(minus_di), 
        "dx": last_val(dx),
        "adx": last_val(adx), 
        "atr": last_val(atr),
        "di_spread": abs(last_val(plus_di) - last_val(minus_di))
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
    p_now = last_val(src); p_prev = safe_iloc(src, -2)
    f_now = last_val(filt); f_prev = safe_iloc(filt, -2)
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": last_val(hi), "lo": last_val(lo)
    }

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    "hold_tp_until": 0  # حالة تأجيل جني الأرباح
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
        "trail_tightened": False, "partial_taken": False,
        "hold_tp_until": 0  # إعادة تعيين حالة التأجيل
    })
    save_state({"in_position": False, "position_qty": 0})
    
    # تفعيل انتظار الإشارة التالية
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_professional(df, ind, info):
    """الإدارة المحترفة للصفقات مع Footprint Management"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    current_price = info["price"]
    current_side = STATE["side"]
    
    # ===== FP-MGMT: تحليل Footprint للإدارة فقط =====
    fp_signal = apply_footprint_management(df, current_side, STATE)
    
    # تطبيق قرارات Footprint Management
    if fp_signal["action"] != "none":
        if fp_signal["action"] == "hold":
            # تأجيل جني الأرباح مؤقتًا
            STATE["hold_tp_until"] = int(time.time() + 180)  # 3 دقائق
            log_i("🟡 FP-MGMT: تأجيل جني الأرباح - قوة اتجاه")
            
        elif fp_signal["action"] == "tighten":
            # تشديد وقف الخسارة المتحرك
            if STATE.get("trail_active"):
                trail_mult = TRAIL_TIGHT_MULT * 0.8  # أكثر تشديدًا
                if current_side == "long":
                    new_trail = current_price - (ind.get("atr", 0) * trail_mult)
                    if STATE.get("trail") is None or new_trail > STATE["trail"]:
                        STATE["trail"] = new_trail
                else:
                    new_trail = current_price + (ind.get("atr", 0) * trail_mult)
                    if STATE.get("trail") is None or new_trail < STATE["trail"]:
                        STATE["trail"] = new_trail
                log_i("🟡 FP-MGMT: تشديد وقف الخسارة المتحرك")
            
        elif fp_signal["action"] == "partial":
            # جني جزئي تحوطي
            close_fraction = 0.10  # 10%
            close_qty = safe_qty(STATE["qty"] * close_fraction)
            
            if close_qty > 0:
                close_side = "sell" if current_side == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"✅ FP-MGMT: جني جزئي تحوطي {close_fraction*100}%")
                        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    except Exception as e:
                        log_e(f"❌ فشل الجني الجزئي: {e}")
                else:
                    log_i(f"DRY_RUN: FP-MGMT جني جزئي {close_qty:.4f}")
            
        elif fp_signal["action"] == "close":
            # إغلاق فوري
            log_w(f"🚨 FP-MGMT: إغلاق صارم - {fp_signal.get('tags', ['إشارة قوية'])[0]}")
            close_market_strict(f"fp_mgmt_{fp_signal.get('tags', ['strong_signal'])[0]}")
            return
    
    # ===== إدارة الصفقة بالشموع =====
    if CONFIG.get("HOLD_TP_ON_STRONG_CANDLE", True):
        trail = STATE.get("trail")
        tp_layers = TREND_TPS if STATE.get("mode") == "trend" else [SCALP_TP1]
        
        # استدعاء إدارة الصفقة بالشموع
        new_trail, new_tp_layers, managed = _candle_fp_post_entry_boost(
            current_side, df, {}, current_price, trail, tp_layers, log_i
        )
        
        if managed:
            STATE["trail"] = new_trail
            # يمكن تحديث tp_layers هنا إذا كنت تستخدمها
    
    # ===== الاستمرار في الإدارة الأساسية =====
    management_signal = professional_trade_management(df, STATE, current_price)
    
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
                    STATE["profit_targets_achieved"] = management_signal.get("new_achieved_tps", STATE["profit_targets_achieved"] + 1)
                except Exception as e:
                    log_e(f"❌ فشل جني الأرباح: {e}")
            else:
                log_i(f"DRY_RUN: جني أرباح {close_qty:.4f}")
    
    elif management_signal["action"] == "close":
        log_w(f"🚨 إغلاق محترف: {management_signal['reason']}")
        close_market_strict(f"professional_{management_signal['reason']}")
        return
    
    # الاستمرار في الإدارة الأساسية المحسنة
    manage_after_entry_enhanced(df, ind, info)

def manage_after_entry_enhanced(df, ind, info):
    """إدارة محسنة للمركز مع مراعاة Footprint Management"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    # التحقق من حالة تأجيل جني الأرباح
    hold_until = STATE.get("hold_tp_until", 0)
    if hold_until > time.time():
        # تأجيل تنفيذ جني الأرباح
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    snap = emit_snapshots_with_smc(ex, SYMBOL, df)
    gz = snap["gz"]
    
    exit_signal = smart_exit_guard_with_smc(STATE, df, ind, snap["flow"], snap["bm"], 
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
    tp1_pct = TP1_PCT_BASE/100.0
    be_activate_pct = BREAKEVEN_AFTER/100.0
    trail_activate_pct = TRAIL_ACTIVATE_PCT/100.0
    atr_trail_mult = ATR_TRAIL_MULT

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

def smart_exit_guard_with_smc(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    """خروج ذكي مع تحليل SMC المتقدم"""
    # التحليل الأساسي
    basic_exit = smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz)
    
    # إضافة تحليل SMC للخروج
    smc_data = ind.get('smc_analysis', {})
    market_structure = smc_data.get('market_structure', {})
    
    if market_structure:
        # اكتشاف انعكاس هيكلي
        if side == "long" and market_structure.get("choch_bearish"):
            return {
                "action": "close", 
                "why": "smc_choch_bearish",
                "log": f"🔴 خروج SMC | تغير هيكل السوق | pnl={pnl_pct*100:.2f}%"
            }
        
        if side == "short" and market_structure.get("choch_bullish"):
            return {
                "action": "close", 
                "why": "smc_choch_bullish",
                "log": f"🔴 خروج SMC | تغير هيكل السوق | pnl={pnl_pct*100:.2f}%"
            }
        
        # BOS عكسي
        if side == "long" and market_structure.get("bos_bearish"):
            return {
                "action": "close", 
                "why": "smc_bos_bearish",
                "log": f"🔴 خروج SMC | BOS عكسي | pnl={pnl_pct*100:.2f}%"
            }
        
        if side == "short" and market_structure.get("bos_bullish"):
            return {
                "action": "close", 
                "why": "smc_bos_bullish",
                "log": f"🔴 خروج SMC | BOS عكسي | pnl={pnl_pct*100:.2f}%"
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

# =================== PROFESSIONAL DECISION LOGGING ===================
def log_professional_decision(council_data, decision):
    """تسجيل متقدم لقرارات المجلس المحترف"""
    votes = council_data
    advanced = votes.get("advanced_indicators", {})
    smc_data = advanced.get("smc_analysis", {})
    
    print(f"\n🎯 قرار المجلس المحترف:", flush=True)
    print(f"📊 الأصوات: شراء {votes['b']} | بيع {votes['s']}", flush=True)
    print(f"⭐ الثقة: شراء {votes['score_b']:.1f} | بيع {votes['score_s']:.1f}", flush=True)
    print(f"📈 القرار النهائي: {decision}", flush=True)
    
    # تفاصيل SMC
    market_structure = smc_data.get("market_structure", {})
    if market_structure.get("bos_bullish"):
        print(f"🚀 هيكل السوق: BOS صاعد", flush=True)
    elif market_structure.get("bos_bearish"):
        print(f"💥 هيكل السوق: BOS هابط", flush=True)
    
    # تفاصيل المؤشرات المتقدمة
    if advanced.get("macd"):
        macd = advanced["macd"]
        print(f"📈 MACD: {macd['crossover']} | فوق الصفر: {macd['above_zero']}", flush=True)
    
    if advanced.get("vwap"):
        vwap = advanced["vwap"]
        print(f"⚡ VWAP: {vwap['signal']} | انحراف: {vwap['deviation']:.2f}%", flush=True)
    
    if advanced.get("advanced_momentum"):
        momentum = advanced["advanced_momentum"]
        print(f"🚀 الزخم: {momentum['trend']} | قوة: {momentum['strength']:.1f}", flush=True)
    
    print("─" * 80, flush=True)

# =================== PROFESSIONAL TRADE LOOP ===================
def trade_loop_professional_with_smc():
    """حلقة تداول محترفة مع SMC Analysis"""
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
            
            # تحديث الـ Snapshots مع SMC
            snap = emit_snapshots_with_smc(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            # تحديث حالة الربح/الخسارة
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # إدارة الصفقة المفتوحة مع SMC
            if STATE["open"]:
                manage_after_entry_professional(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    "smc": snap.get("cv", {}).get("advanced_indicators", {}).get("smc_analysis", {}),
                    **info
                })
            
            # قرار الدخول باستخدام المجلس المحترف مع SMC
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"انتشار مرتفع ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = ultimate_council_professional(df)
            smc_data = council_data.get("advanced_indicators", {}).get("smc_analysis", {})
            
            # قرار التداول النهائي مع أولوية SMC
            decision = None
            min_confidence = ULTIMATE_MIN_CONFIDENCE
            
            # تخفيض عتبة الثقة إذا كان SMC قوي
            market_structure = smc_data.get("market_structure", {})
            if market_structure.get("bos_bullish") or market_structure.get("bos_bearish"):
                min_confidence = 6.0  # تخفيض العتبة لصالح SMC القوي
            
            if (council_data["score_b"] >= min_confidence and 
                council_data["score_b"] > council_data["score_s"] + 2.0):
                decision = "BUY"
            elif (council_data["score_s"] >= min_confidence and 
                  council_data["score_s"] > council_data["score_b"] + 2.0):
                decision = "SELL"
            
            # تسجيل القرار المحترف مع SMC
            if decision:
                log_professional_decision(council_data, decision)
                
                # تسجيل تفاصيل SMC
                if market_structure.get("bos_bullish") and decision == "BUY":
                    print(f"🎯 SMC CONFIRMATION: BOS صاعد يدعم الشراء", flush=True)
                elif market_structure.get("bos_bearish") and decision == "SELL":
                    print(f"🎯 SMC CONFIRMATION: BOS هابط يدعم البيع", flush=True)
            
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

# استبدال حلقة التداول المحترفة
trade_loop = trade_loop_professional_with_smc

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    fp_status = 'ENABLED' if FP_MGMT_ENABLED else 'DISABLED'
    return f"✅ SUI Council PROFESSIONAL Bot — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Footprint: {fp_status}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "PROFESSIONAL_COUNCIL_WITH_SMC", "wait_for_next_signal": wait_for_next_signal_side,
        "footprint_management": {
            "enabled": FP_MGMT_ENABLED,
            "settings": FP_MGMT_SETTINGS
        },
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "PROFESSIONAL_COUNCIL_WITH_SMC", "wait_for_next_signal": wait_for_next_signal_side,
        "footprint_management": {
            "enabled": FP_MGMT_ENABLED,
            "current_action": STATE.get("hold_tp_until", 0) > time.time() and "HOLD_TP" or "NORMAL"
        }
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
ultimate_council_voting_with_footprint = ultimate_council_professional
manage_after_entry_ultimate = manage_after_entry_professional
execute_trade_decision_with_footprint = execute_professional_trade
emit_snapshots = emit_snapshots_with_smc

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("SUI COUNCIL PROFESSIONAL BOT - SMART MONEY CONCEPTS + FOOTPRINT MANAGEMENT")
    
    # طباعة إعدادات CONFIG
    print(f"\n🎛️  CONFIG SUMMARY:", flush=True)
    print(f"   OVERRIDE: {CONFIG.get('OVERRIDE', False)}", flush=True)
    print(f"   EXCHANGE: {EXCHANGE_NAME} | SYMBOL: {SYMBOL} | INTERVAL: {INTERVAL}", flush=True)
    print(f"   LEVERAGE: {LEVERAGE}x | RISK: {RISK_ALLOC*100:.0f}%", flush=True)
    print(f"   RF: period={RF_PERIOD} | mult={RF_MULT}", flush=True)
    print(f"   COUNCIL: min_confidence={ULTIMATE_MIN_CONFIDENCE}", flush=True)
    print(f"   FP_MGMT: {'ENABLED' if FP_MGMT_ENABLE else 'DISABLED'}", flush=True)
    
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"🎯 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • PROFESSIONAL_COUNCIL=ENABLED", "yellow"))
    print(colored(f"🏆 PROFESSIONAL MIN CONFIDENCE: {ULTIMATE_MIN_CONFIDENCE}", "yellow"))
    print(colored(f"📊 SMART MONEY CONCEPTS: BOS + Order Blocks + FVG + Liquidity Analysis", "yellow"))
    print(colored(f"📈 ADVANCED INDICATORS: MACD + VWAP + Volume Momentum + Dynamic Pivots", "yellow"))
    print(colored(f"👣 FOOTPRINT MANAGEMENT: {'ENABLED' if FP_MGMT_ENABLED else 'DISABLED'} (للإدارة فقط)", "yellow"))
    if FP_MGMT_ENABLED:
        print(colored(f"   ↳ الوزن: {FP_MGMT_WEIGHT} | Delta-Z: {FP_MGMT_DELTA_Z} | Vol-Z: {FP_MGMT_VOL_Z}", "yellow"))
    print(colored(f"🚀 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
