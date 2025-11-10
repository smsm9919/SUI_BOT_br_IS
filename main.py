# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم
• مجلس الإدارة الفائق الذكي مع 13 استراتيجية متقدمة
• نظام ركوب الترند الذكي لتحقيق أقصى ربح متتالي
• السكالب الفائق بأهداف متعددة ذكية
• إدارة صفقات ذكية متكيفة مع قوة الترند
• Multi-Exchange Support: BingX & Bybit
• نظام Footprint + Diagonal Order-Flow المتقدم
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
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()

if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:
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
BOT_VERSION = f"SUI ULTRA PRO AI v6.0 — {EXCHANGE_NAME.upper()}"
print("🚀 Booting:", BOT_VERSION, flush=True)

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

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80, 2.50, 3.50, 5.00, 7.00]
TREND_TP_FRACS  = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]

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

MAX_TRADES_PER_HOUR = 8
COOLDOWN_SECS_AFTER_CLOSE = 45
ADX_GATE = 17

# ===== SUPER SCALP ENGINE =====
SCALP_MODE            = True
SCALP_EXECUTE         = True
SCALP_SIZE_FACTOR     = 0.35
SCALP_ADX_GATE        = 12.0
SCALP_MIN_SCORE       = 3.5
SCALP_IMB_THRESHOLD   = 1.00
SCALP_VOL_MA_FACTOR   = 1.20
SCALP_COOLDOWN_SEC    = 8
SCALP_RESPECT_WAIT    = False
SCALP_TP_SINGLE_PCT   = 0.35
SCALP_BE_AFTER_PCT    = 0.15
SCALP_ATR_TRAIL_MULT  = 1.0

# ==== SUPER COUNCIL ENHANCEMENTS ====
COUNCIL_AI_MODE = True
TREND_EARLY_DETECTION = True
MOMENTUM_ACCELERATION = True
VOLUME_CONFIRMATION = True
PRICE_ACTION_INTELLIGENCE = True

# أوزان التصويت الذكية
WEIGHT_ADX = 1.5
WEIGHT_RSI = 1.2
WEIGHT_MACD = 1.3
WEIGHT_VOLUME = 1.1
WEIGHT_FLOW = 1.4
WEIGHT_GOLDEN = 1.6
WEIGHT_CANDLES = 1.2
WEIGHT_MOMENTUM = 1.3
WEIGHT_FOOTPRINT = 1.5
WEIGHT_DIAGONAL = 1.4

# ===== INTELLIGENT TREND MANAGEMENT =====
TREND_RIDING_AI = True
DYNAMIC_TP_ADJUSTMENT = True
ADAPTIVE_TRAILING = True
TREND_STRENGTH_ANALYSIS = True

# إعدادات ركوب الترند الذكية
TREND_FOLLOW_MULTIPLIER = 1.5
WEAK_TREND_EARLY_EXIT = True
STRONG_TREND_HOLD = True
TREND_REENTRY_STRATEGY = True

# ===== FLOW/FOOTPRINT Council Boost =====
FLOW_IMB_RATIO          = 1.6
FLOW_STACK_DEPTH        = 4
FLOW_ABSORB_PCTL        = 0.95
FLOW_ABSORB_MAX_TICKS   = 2
FP_WINDOW               = 3
FP_SCORE_BUY            = (2, 1.0)
FP_SCORE_SELL           = (2, 1.0)
FP_SCORE_ABSORB_PENALTY = (-1, -0.5)
DIAG_SCORE_BUY          = (2, 1.0)
DIAG_SCORE_SELL         = (2, 1.0)

# ===== PROFIT ACCUMULATION SYSTEM =====
COMPOUND_PROFIT_REINVEST = True
PROFIT_REINVEST_RATIO = 0.3  # 30% من الأرباح يعاد استثمارها
MIN_COMPOUND_BALANCE = 50.0  # الحد الأدنى للرصيد قبل البدء في المراكبة

# ===== SNAPSHOT & MARK SYSTEM =====
GREEN="🟢"; RED="🔴"
RESET="\x1b[0m"; BOLD="\x1b[1m"
FG_G="\x1b[32m"; FG_R="\x1b[31m"; FG_C="\x1b[36m"; FG_Y="\x1b[33m"; FG_M="\x1b[35m"

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
    exchange_config = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    }
    
    if EXCHANGE_NAME == "bybit":
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bybit(exchange_config)
    else:
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bingx(exchange_config)

ex = make_ex()

# =================== EXCHANGE-SPECIFIC ADAPTERS ===================
def exchange_specific_params(side, is_close=False):
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    try:
        if EXCHANGE_NAME == "bybit":
            exchange.set_leverage(leverage, symbol)
        else:
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

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== LOGGING SETUP ===================
def setup_file_logging():
    """إعداد التسجيل المهني مع قمع رسائل Werkzeug"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s [%(filename)s:%(lineno)d]"))
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.INFO)
    
    log_i("🔄 Professional logging ready - File rotation + Werkzeug suppression")

setup_file_logging()

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _fmt(x,n=6):
    try: return f"{float(x):.{n}f}"
    except: return str(x)

def _pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)

def last_scalar(x, default=0.0):
    """يرجع float من آخر عنصر; يقبل Series/np.ndarray/list/float."""
    try:
        if isinstance(x, pd.Series): return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)): return float(x[-1])
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_get(ind: dict, key: str, default=0.0):
    """يقرأ مؤشر من dict ويحوّله scalar أخير."""
    if ind is None: 
        return float(default)
    val = ind.get(key, default)
    return last_scalar(val, default=default)

def _ind_brief(ind):
    if not ind: return "n/a"
    
    # استخراج قيم scalar بأمان
    adx = safe_get(ind, 'adx', 0)
    di_spread = safe_get(ind, 'di_spread', 0)
    rsi = safe_get(ind, 'rsi', 0)
    rsi_ma = safe_get(ind, 'rsi_ma', 0)
    atr = safe_get(ind, 'atr', 0)
    
    return (f"ADX={adx:.1f} DI={di_spread:.1f} | "
            f"RSI={rsi:.1f}/{rsi_ma:.1f} | "
            f"ATR={atr:.4f}")

def _council_brief(c):
    if not c: return "n/a"
    return f"B:{c.get('b',0)}/{_fmt(c.get('score_b',0),1)} | S:{c.get('s',0)}/{_fmt(c.get('score_s',0),1)}"

def _flow_brief(f):
    if not f: return "n/a"
    parts=[f"Δz={_fmt(f.get('delta_z','n/a'),2)}", f"CVD={_fmt(f.get('cvd_last','n/a'),0)}", f"trend={f.get('cvd_trend','?')}"]
    if f.get("spike"): parts.append("SPIKE")
    return " ".join(parts)

def print_position_snapshot(reason="OPEN", color=None):
    try:
        side   = STATE.get("side")
        open_f = STATE.get("open",False)
        qty    = STATE.get("qty"); px = STATE.get("entry")
        mode   = STATE.get("mode","trend")
        lev    = globals().get("LEVERAGE",0)
        tp1    = globals().get("TP1_PCT_BASE",0)
        be_a   = globals().get("BREAKEVEN_AFTER",0)
        trailA = globals().get("TRAIL_ACTIVATE_PCT",0)
        atrM   = globals().get("ATR_TRAIL_MULT",0)
        bal    = balance_usdt()
        spread = STATE.get("last_spread_bps")
        council= STATE.get("last_council")
        ind    = STATE.get("last_ind")
        flow   = STATE.get("last_flow")

        if color is None:
            icon = GREEN if side=="buy" else RED
            ccol = FG_G if side=="buy" else FG_R
        else:
            icon = GREEN if str(color).lower()=="green" else RED
            ccol = FG_G if icon==GREEN else FG_R

        log_i(f"{ccol}{BOLD}{icon} {reason} — POSITION SNAPSHOT{RESET}")
        log_i(f"{BOLD}SIDE:{RESET} {side} | {BOLD}QTY:{RESET} {_fmt(qty)} | {BOLD}ENTRY:{RESET} {_fmt(px)} | "
              f"{BOLD}LEV:{RESET} {lev}× | {BOLD}MODE:{RESET} {mode} | {BOLD}OPEN:{RESET} {open_f}")
        log_i(f"{BOLD}TP1:{RESET} {_pct(tp1)} | {BOLD}BE@:{RESET} {_pct(be_a)} | "
              f"{BOLD}TRAIL:{RESET} act≥{_pct(trailA)}, ATR×{atrM} | {BOLD}SPREAD:{RESET} {_fmt(spread,2)} bps")
        log_i(f"{FG_C}IND:{RESET} {_ind_brief(ind)}")
        log_i(f"{FG_M}COUNCIL:{RESET} {_council_brief(council)}")
        log_i(f"{FG_Y}FLOW:{RESET} {_flow_brief(flow)}")
        log_i("—"*72)
    except Exception as e:
        log_w(f"SNAPSHOT ERR: {e}")

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

# =================== FOOTPRINT & DIAGONAL FLOW SYSTEMS ===================
def analyze_footprint_fallback(df: pd.DataFrame, window: int = FP_WINDOW):
    """
    نسخة بديلة تعتمد على علاقة الحجم بالسعر عندما لا تتوفر بيانات Footprint مباشرة
    """
    try:
        sub = df.tail(window)
        votes_b = votes_s = 0
        score_b = score_s = 0.0
        tag = "balanced"

        for _, row in sub.iterrows():
            close = float(row["close"])
            open_ = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            volume = float(row["volume"])
            
            candle_up = close > open_
            body_size = abs(close - open_)
            total_range = high - low
            
            if total_range <= 0:
                continue
                
            # نسبة الجسم إلى المدى (تشير إلى قوة الاتجاه)
            body_ratio = body_size / total_range
            
            # حجم التداول بالنسبة للمدى (كثافة التداول)
            volume_density = volume / total_range if total_range > 0 else 0
            
            # شمعة قوية صاعدة: جسم كبير + حجم عالي
            if candle_up and body_ratio > 0.6 and volume_density > np.percentile([v/(h-l) for v,h,l in zip(sub['volume'], sub['high'], sub['low']) if (h-l)>0], 70):
                vb, sb = FP_SCORE_BUY
                votes_b += vb; score_b += sb; tag = "aggressive_buy"
            
            # شمعة قوية هابطة: جسم كبير + حجم عالي  
            elif not candle_up and body_ratio > 0.6 and volume_density > np.percentile([v/(h-l) for v,h,l in zip(sub['volume'], sub['high'], sub['low']) if (h-l)>0], 70):
                vs, ss = FP_SCORE_SELL
                votes_s += vs; score_s += ss; tag = "aggressive_sell"
                
            # دوجي مع حجم عالي (امتصاص)
            elif body_ratio < 0.3 and volume_density > np.percentile([v/(h-l) for v,h,l in zip(sub['volume'], sub['high'], sub['low']) if (h-l)>0], 80):
                if close > open_:  # دوجي مع إغلاق أعلى (امتصاص بيع)
                    vs, ss = FP_SCORE_ABSORB_PENALTY
                    votes_s += vs; score_s += ss; tag = "absorb_bid"
                else:  # دوجي مع إغلاق أقل (امتصاص شراء)
                    vb, sb = FP_SCORE_ABSORB_PENALTY
                    votes_b += vb; score_b += sb; tag = "absorb_ask"

        return {"votes_b": votes_b, "votes_s": votes_s,
                "score_b": score_b, "score_s": score_s, "tag": tag}
    except Exception as e:
        return {"votes_b":0,"votes_s":0,"score_b":0.0,"score_s":0.0,"tag":f"err:{e}"}

def analyze_diagonal_flow(orderbook: dict, depth: int = FLOW_STACK_DEPTH, imb_ratio: float = FLOW_IMB_RATIO):
    try:
        bids = orderbook.get("bids", []) or []
        asks = orderbook.get("asks", []) or []
        n = min(len(bids), len(asks), depth)
        buy_strength = sell_strength = 0

        for i in range(n):
            b_qty = float(bids[i][1]); a_qty = float(asks[i][1])
            if b_qty <= 0 or a_qty <= 0: 
                continue
            r = b_qty / a_qty
            if r >= imb_ratio: buy_strength += 1
            elif r <= 1/imb_ratio: sell_strength += 1

        if buy_strength > sell_strength:
            bias = "buy"; votes, score = DIAG_SCORE_BUY
        elif sell_strength > buy_strength:
            bias = "sell"; votes, score = DIAG_SCORE_SELL
        else:
            bias = "neutral"; votes, score = 0, 0.0

        return {"bias": bias, "votes": votes, "score": score,
                "buy_strength": buy_strength, "sell_strength": sell_strength}
    except Exception as e:
        return {"bias":"neutral","votes":0,"score":0.0,"err":str(e)}

def council_boost_from_flow(df: pd.DataFrame, orderbook: dict):
    fp = analyze_footprint_fallback(df)
    dg = analyze_diagonal_flow(orderbook)

    # تجميع التصويت
    votes_b = fp["votes_b"] + (dg["votes"] if dg["bias"]=="buy" else 0)
    votes_s = fp["votes_s"] + (dg["votes"] if dg["bias"]=="sell" else 0)
    score_b = fp["score_b"] + (dg["score"] if dg["bias"]=="buy" else 0.0)
    score_s = fp["score_s"] + (dg["score"] if dg["bias"]=="sell" else 0.0)

    tag = f"FP:{fp['tag']} | DIAG:{dg['bias']}(B{dg.get('buy_strength',0)}/S{dg.get('sell_strength',0)})"
    return {"votes_b":votes_b, "votes_s":votes_s, "score_b":score_b, "score_s":score_s, "tag":tag}

# ========= Unified snapshot emitter =========
def emit_snapshots(exchange, symbol, df, balance_fn=None, pnl_fn=None):
    try:
        bm = bookmap_snapshot(exchange, symbol)
        flow = compute_flow_metrics(df)
        cv = council_votes_pro(df)
        mode = decide_strategy_mode(df)
        gz = golden_zone_check(df, {"adx": cv["ind"].get("adx", 0)}, "buy" if cv["b"]>=cv["s"] else "sell")

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
                f"RSI={safe_get(cv['ind'],'rsi',0):.1f} ADX={safe_get(cv['ind'],'adx',0):.1f} "
                f"DI={safe_get(cv['ind'],'di_spread',0):.1f} | Confidence: {cv.get('confidence',0):.1f}")

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
            
            gz_snap_note = ""
            if gz and gz.get("ok"):
                zone_type = gz["zone"]["type"]
                zone_score = gz["score"]
                gz_snap_note = f" | 🟡{zone_type} s={zone_score:.1f}"
            
            flow_z = flow['delta_z'] if flow and flow.get('ok') else 0.0
            bm_imb = bm['imbalance'] if bm and bm.get('ok') else 1.0
            
            print(f"🧠 SNAP | {side_hint} | votes={cv['b']}/{cv['s']} score={cv['score_b']:.1f}/{cv['score_s']:.1f} "
                  f"| ADX={safe_get(cv['ind'],'adx',0):.1f} DI={safe_get(cv['ind'],'di_spread',0):.1f} | "
                  f"z={flow_z:.2f} | imb={bm_imb:.2f}{gz_snap_note}", 
                  flush=True)
            
            print("✅ ADDONS LIVE", flush=True)

        return {"bm": bm, "flow": flow, "cv": cv, "mode": mode, "gz": gz, "wallet": wallet}
    except Exception as e:
        print(f"🟨 AddonLog error: {e}", flush=True)
        return {"bm": None, "flow": None, "cv": {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"ind":{}},
                "mode": {"mode":"n/a"}, "gz": None, "wallet": ""}

# =================== ADVANCED INDICATORS ===================
def sma(series, n: int):
    return series.rolling(n, min_periods=1).mean()

def ema(series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def compute_rsi(close, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(span=n, adjust=False).mean()
    roll_down = down.ewm(span=n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, 1e-12)
    rsi = 100 - (100/(1+rs))
    return rsi.fillna(50)

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = ema(macd, signal)
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def compute_bollinger_bands(close, n=20, k=2):
    sma_val = sma(close, n)
    std = close.rolling(n).std()
    upper = sma_val + (std * k)
    lower = sma_val - (std * k)
    return upper, sma_val, lower

def compute_stochastic(high, low, close, n=14, d=3):
    lowest_low = low.rolling(n).min()
    highest_high = high.rolling(n).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_line = sma(k, d)
    return k, d_line

def compute_volume_profile(df, period=20):
    volume = df['volume'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    price_range = high - low
    volume_per_price = volume / (price_range.replace(0, 1e-12))
    
    return {
        'volume_ma': sma(volume, period),
        'volume_spike': volume > sma(volume, period) * 1.5,
        'volume_trend': 'up' if volume.iloc[-1] > volume.iloc[-2] else 'down'
    }

def compute_momentum_indicators(df):
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    
    roc = ((close - close.shift(5)) / close.shift(5)) * 100
    price_accel = close.diff().diff()
    volatility = high - low
    
    return {
        'roc': roc.iloc[-1] if len(roc) > 0 else 0,
        'price_accel': price_accel.iloc[-1] if len(price_accel) > 0 else 0,
        'volatility': volatility.iloc[-1] if len(volatility) > 0 else 0,
        'volatility_ma': sma(volatility, 20).iloc[-1] if len(volatility) >= 20 else 0
    }

def compute_trend_strength(df, ind):
    close = df['close'].astype(float)
    adx = safe_get(ind, 'adx', 0)
    plus_di = safe_get(ind, 'plus_di', 0)
    minus_di = safe_get(ind, 'minus_di', 0)
    
    momentum_5 = ((close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]) * 100 if len(close) >= 5 else 0
    momentum_10 = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(close) >= 10 else 0
    
    trend_consistency = 0
    if len(close) >= 10:
        up_days = sum(close.diff().tail(10) > 0)
        down_days = sum(close.diff().tail(10) < 0)
        trend_consistency = max(up_days, down_days) / 10.0
    
    if adx > 40 and abs(momentum_5) > 3.0 and trend_consistency > 0.7:
        strength = "very_strong"
        multiplier = 2.0
    elif adx > 30 and abs(momentum_5) > 2.0 and trend_consistency > 0.6:
        strength = "strong"
        multiplier = 1.5
    elif adx > 25 and abs(momentum_5) > 1.0:
        strength = "moderate"
        multiplier = 1.2
    elif adx > 20:
        strength = "weak"
        multiplier = 1.0
    else:
        strength = "no_trend"
        multiplier = 0.8
    
    direction = "up" if plus_di > minus_di else "down"
    
    return {
        "strength": strength,
        "direction": direction,
        "multiplier": multiplier,
        "adx": adx,
        "momentum_5": momentum_5,
        "momentum_10": momentum_10,
        "consistency": trend_consistency
    }

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

    rng1 = _rng(h1,l1); up = _upper_wick(h1,o1,c1); dn = _lower_wick(l1,o1,c1)
    wick_up_big = (up >= 1.2*_body(o1,c1)) and (up >= 0.4*rng1)
    wick_dn_big = (dn >= 1.2*_body(o1,c1)) and (dn >= 0.4*rng1)

    if is_doji:
        strength_b *= 0.8; strength_s *= 0.8

    return {
        "buy": strength_b>0, "sell": strength_s>0,
        "score_buy": round(strength_b,2), "score_sell": round(strength_s,2),
        "wick_up_big": bool(wick_up_big), "wick_dn_big": bool(wick_dn_big),
        "doji": bool(is_doji), "pattern": ",".join(tags) if tags else None
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
        
        adx = safe_get(ind, 'adx', 0) if ind else 0
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
        adx = safe_get(ind, 'adx', 0)
        di_plus = safe_get(ind, 'plus_di', 0)
        di_minus = safe_get(ind, 'minus_di', 0)
    
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

# =================== SUPER COUNCIL AI ===================
def super_council_ai_enhanced(df):
    try:
        if len(df) < 50:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "confidence": 0.0}
        
        ind = compute_indicators(df)
        
        # استخراج قيم scalar بأمان
        adx = safe_get(ind, "adx", 0.0)
        plus_di = safe_get(ind, "plus_di", 0.0)
        minus_di = safe_get(ind, "minus_di", 0.0)
        di_spread = abs(plus_di - minus_di)
        rsi_val = safe_get(ind, "rsi", 50.0)
        atr_val = safe_get(ind, "atr", 0.0)
        
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)
        candles = compute_candles(df)
        flow = compute_flow_metrics(df)
        volume_profile = compute_volume_profile(df)
        momentum = compute_momentum_indicators(df)
        trend_strength = compute_trend_strength(df, ind)
        
        macd, macd_signal, macd_hist = compute_macd(df['close'].astype(float))
        macd_bullish = last_scalar(macd) > last_scalar(macd_signal) and last_scalar(macd_hist) > 0
        macd_bearish = last_scalar(macd) < last_scalar(macd_signal) and last_scalar(macd_hist) < 0
        
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(df['close'].astype(float))
        current_price = float(df['close'].iloc[-1])
        bb_position = (current_price - last_scalar(bb_lower)) / (last_scalar(bb_upper) - last_scalar(bb_lower)) if last_scalar(bb_upper) != last_scalar(bb_lower) else 0.5
        
        stoch_k, stoch_d = compute_stochastic(df['high'].ast(float), df['low'].astype(float), df['close'].astype(float))
        stoch_bullish = last_scalar(stoch_k) > last_scalar(stoch_d) and last_scalar(stoch_k) < 80
        stoch_bearish = last_scalar(stoch_k) < last_scalar(stoch_d) and last_scalar(stoch_k) > 20
        
        votes_b = 0; votes_s = 0
        score_b = 0.0; score_s = 0.0
        logs = []
        confidence_factors = []

        # ===== FLOW/FOOTPRINT BOOST =====
        try:
            # جلب orderbook حديث إذا لم يكن محفوظًا
            current_orderbook = STATE.get("last_orderbook", {})
            if not current_orderbook:
                current_orderbook = ex.fetch_order_book(SYMBOL, limit=FLOW_STACK_DEPTH)
                STATE["last_orderbook"] = current_orderbook
            
            boost = council_boost_from_flow(df, current_orderbook)
            
            # دمج النتائج مع التصويت الأساسي
            votes_b += boost["votes_b"]
            votes_s += boost["votes_s"]
            score_b += boost["score_b"] * WEIGHT_FOOTPRINT
            score_s += boost["score_s"] * WEIGHT_FOOTPRINT
            
            logs.append(f"🧭 FLOW-BOOST → {boost['tag']}  "
                       f"Δvotes: B+{boost['votes_b']} S+{boost['votes_s']} | "
                       f"Δscore: B+{boost['score_b']:.1f} S+{boost['score_s']:.1f}")
        except Exception as e:
            logs.append(f"🟨 FLOW-BOOST error: {e}")

        # 1. تحليل الزخم المبكر
        if TREND_EARLY_DETECTION:
            momentum_accel = safe_get(momentum, 'price_accel', 0.0)
            momentum_roc = safe_get(momentum, 'roc', 0.0)
            
            if momentum_accel > 0 and momentum_roc > 0.5:
                score_b += WEIGHT_MOMENTUM * 1.5
                votes_b += 2
                logs.append("🚀 تسارع صاعد قوي")
                confidence_factors.append(1.2)
            
            if momentum_accel < 0 and momentum_roc < -0.5:
                score_s += WEIGHT_MOMENTUM * 1.5
                votes_s += 2
                logs.append("💥 تسارع هابط قوي")
                confidence_factors.append(1.2)

        # 2. تأكيد الحجم
        if VOLUME_CONFIRMATION:
            volume_spike = volume_profile.get('volume_spike', False)
            volume_trend = volume_profile.get('volume_trend', '')
            
            if volume_spike and volume_trend == 'up':
                if current_price > float(df['open'].iloc[-1]):
                    score_b += WEIGHT_VOLUME * 1.2
                    votes_b += 1
                    logs.append("📊 حجم صاعد مؤكد")
                else:
                    score_s += WEIGHT_VOLUME * 1.2
                    votes_s += 1
                    logs.append("📊 حجم هابط مؤكد")

        # 3. مؤشر الاتجاه المتقدم
        if adx > ADX_TREND_MIN:
            if plus_di > minus_di and di_spread > DI_SPREAD_TREND:
                score_b += WEIGHT_ADX * 2.0
                votes_b += 3
                logs.append(f"📈 ترند صاعد قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.5)
            elif minus_di > plus_di and di_spread > DI_SPREAD_TREND:
                score_s += WEIGHT_ADX * 2.0
                votes_s += 3
                logs.append(f"📉 ترند هابط قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.5)

        # 4. مؤشر RSI المتقدم
        rsi_cross = rsi_ctx.get("cross", "none")
        rsi_trendz = rsi_ctx.get("trendZ", "none")
        
        if rsi_cross == "bull" and rsi_val < 70:
            score_b += WEIGHT_RSI * 1.5
            votes_b += 2
            logs.append("🟢 RSI إيجابي قوي")
        elif rsi_cross == "bear" and rsi_val > 30:
            score_s += WEIGHT_RSI * 1.5
            votes_s += 2
            logs.append("🔴 RSI سلبي قوي")

        if rsi_trendz == "bull":
            score_b += WEIGHT_RSI * 2.0
            votes_b += 3
            logs.append("🚀 RSI ترند صاعد مستمر")
        elif rsi_trendz == "bear":
            score_s += WEIGHT_RSI * 2.0
            votes_s += 3
            logs.append("💥 RSI ترند هابط مستمر")

        # 5. المناطق الذهبية المحسنة
        if gz and gz.get("ok"):
            gz_score = gz.get("score", 0.0)
            zone_type = gz.get("zone", {}).get("type", "")
            
            if zone_type == 'golden_bottom' and gz_score >= 6.0:
                score_b += WEIGHT_GOLDEN * 2.5
                votes_b += 4
                logs.append(f"🏆 قاع ذهبي فائق (قوة: {gz_score:.1f})")
                confidence_factors.append(1.8)
            elif zone_type == 'golden_top' and gz_score >= 6.0:
                score_s += WEIGHT_GOLDEN * 2.5
                votes_s += 4
                logs.append(f"🏆 قمة ذهبية فائقة (قوة: {gz_score:.1f})")
                confidence_factors.append(1.8)

        # 6. تحليل الشموع اليابانية المتقدم
        candles_buy_score = candles.get("score_buy", 0.0)
        candles_sell_score = candles.get("score_sell", 0.0)
        
        if candles_buy_score > 0:
            enhanced_candle_score = min(3.0, candles_buy_score * 1.2)
            score_b += WEIGHT_CANDLES * enhanced_candle_score
            votes_b += int(enhanced_candle_score)
            logs.append(f"🕯️ شموع BUY قوية ({candles.get('pattern', '')}) +{enhanced_candle_score:.1f}")
        
        if candles_sell_score > 0:
            enhanced_candle_score = min(3.0, candles_sell_score * 1.2)
            score_s += WEIGHT_CANDLES * enhanced_candle_score
            votes_s += int(enhanced_candle_score)
            logs.append(f"🕯️ شموع SELL قوية ({candles.get('pattern', '')}) +{enhanced_candle_score:.1f}")

        # 7. تحليل التدفق والطلب المتقدم
        if flow.get("ok"):
            delta_z = flow.get("delta_z", 0.0)
            cvd_trend = flow.get("cvd_trend", "")
            
            if delta_z >= 2.0 and cvd_trend == "up":
                score_b += WEIGHT_FLOW * 1.8
                votes_b += 2
                logs.append(f"🌊 تدفق شرائي قوي (z: {delta_z:.2f})")
            elif delta_z <= -2.0 and cvd_trend == "down":
                score_s += WEIGHT_FLOW * 1.8
                votes_s += 2
                logs.append(f"🌊 تدفق بيعي قوي (z: {delta_z:.2f})")

        # 8. مؤشر MACD المتقدم
        if macd_bullish and last_scalar(macd_hist) > last_scalar(macd_hist.shift(1)):
            score_b += WEIGHT_MACD * 1.5
            votes_b += 2
            logs.append("📈 MACD صاعد متسارع")
        elif macd_bearish and last_scalar(macd_hist) < last_scalar(macd_hist.shift(1)):
            score_s += WEIGHT_MACD * 1.5
            votes_s += 2
            logs.append("📉 MACD هابط متسارع")

        # 9. بولنجر باندز لاكتشاف الانعكاسات
        if bb_position < 0.2 and current_price > last_scalar(bb_lower):
            score_b += 1.2
            votes_b += 1
            logs.append("🔄 ارتداد من نطاق بولنجر سفلي")
        elif bb_position > 0.8 and current_price < last_scalar(bb_upper):
            score_s += 1.2
            votes_s += 1
            logs.append("🔄 ارتداد من نطاق بولنجر علوي")

        # 10. ستوكاستيك للمدى القصير
        stoch_k_val = last_scalar(stoch_k)
        if stoch_bullish and stoch_k_val < 30:
            score_b += 1.0
            votes_b += 1
            logs.append("🎯 ستوكاستيك في منطقة شراء")
        elif stoch_bearish and stoch_k_val > 70:
            score_s += 1.0
            votes_s += 1
            logs.append("🎯 ستوكاستيك في منطقة بيع")

        # 11. قوة الترند
        trend_strength_val = trend_strength.get("strength", "")
        trend_direction = trend_strength.get("direction", "")
        trend_multiplier = trend_strength.get("multiplier", 1.0)
        
        if trend_strength_val in ["strong", "very_strong"]:
            if trend_direction == "up":
                score_b += trend_multiplier * 1.5
                votes_b += 2
                logs.append(f"💪 ترند صاعد {trend_strength_val} (مضاعف: {trend_multiplier})")
            else:
                score_s += trend_multiplier * 1.5
                votes_s += 2
                logs.append(f"💪 ترند هابط {trend_strength_val} (مضاعف: {trend_multiplier})")

        # تطبيق عوامل الثقة
        if confidence_factors:
            confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
            score_b *= confidence_multiplier
            score_s *= confidence_multiplier

        # تخفيف في النطاق المحايد
        if rsi_ctx.get("in_chop", False):
            score_b *= 0.7
            score_s *= 0.7
            logs.append("⚖️ RSI محايد — تخفيض ثقة")

        # حارس ADX العام
        if adx < ADX_GATE:
            score_b *= 0.8
            score_s *= 0.8
            logs.append(f"🛡️ ADX Gate ({adx:.1f} < {ADX_GATE})")

        # حساب الثقة النهائية
        total_score = score_b + score_s
        confidence = min(1.0, total_score / 25.0) if total_score > 0 else 0.0

        # تحديث المؤشرات الإضافية
        ind.update({
            "rsi_ma": rsi_ctx.get("rsi_ma", 50.0),
            "rsi_trendz": rsi_trendz,
            "di_spread": di_spread,
            "gz": gz,
            "candle_buy_score": candles_buy_score,
            "candle_sell_score": candles_sell_score,
            "wick_up_big": candles.get("wick_up_big", False),
            "wick_dn_big": candles.get("wick_dn_big", False),
            "candle_tags": candles.get("pattern", ""),
            "macd_bullish": macd_bullish,
            "macd_bearish": macd_bearish,
            "bb_position": bb_position,
            "momentum": momentum,
            "volume_profile": volume_profile,
            "trend_strength": trend_strength
        })

        return {
            "b": votes_b, "s": votes_s,
            "score_b": round(score_b, 2), "score_s": round(score_s, 2),
            "logs": logs, "ind": ind, "gz": gz, "candles": candles,
            "confidence": round(confidence, 2),
            "momentum": momentum,
            "volume": volume_profile,
            "trend_strength": trend_strength
        }
    except Exception as e:
        log_w(f"super_council_ai_enhanced error: {e}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"confidence":0.0}

council_votes_pro_enhanced = super_council_ai_enhanced
council_votes_pro = super_council_ai_enhanced

# =================== SUPER SCALP AI ===================
_last_scalp_ts = 0
_scalp_profit_total = 0.0

def detect_super_scalp_opportunity(df, ind, flow, volume_profile, momentum, spread_bps):
    try:
        if not SCALP_MODE or not SCALP_EXECUTE:
            return (None, "scalp_off")

        if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
            return (None, f"spread>{MAX_SPREAD_BPS}bps")

        current_price = float(df['close'].iloc[-1])
        volume_ok = volume_profile['volume_spike'] and volume_profile['volume_trend'] == 'up'
        momentum_ok = abs(momentum['roc']) > 0.3
        volatility_ok = momentum['volatility'] > momentum['volatility_ma'] * 0.8
        
        scalp_council = {
            'b': 0, 's': 0,
            'score_b': 0.0, 'score_s': 0.0
        }
        
        rsi = safe_get(ind, 'rsi', 50)
        if 30 <= rsi <= 45:
            scalp_council['score_b'] += 1.5
            scalp_council['b'] += 1
        elif 55 <= rsi <= 70:
            scalp_council['score_s'] += 1.5
            scalp_council['s'] += 1
        
        if flow and flow.get('ok'):
            if flow['delta_z'] > 1.5 and volume_ok:
                scalp_council['score_b'] += 2.0
                scalp_council['b'] += 2
            elif flow['delta_z'] < -1.5 and volume_ok:
                scalp_council['score_s'] += 2.0
                scalp_council['s'] += 2
        
        if momentum_ok and volatility_ok:
            if momentum['price_accel'] > 0 and momentum['roc'] > 0:
                scalp_council['score_b'] += 1.5
                scalp_council['b'] += 1
            elif momentum['price_accel'] < 0 and momentum['roc'] < 0:
                scalp_council['score_s'] += 1.5
                scalp_council['s'] += 1
        
        candles = compute_candles(df)
        if candles['score_buy'] > 1.0 and candles['wick_dn_big']:
            scalp_council['score_b'] += 1.2
            scalp_council['b'] += 1
        if candles['score_sell'] > 1.0 and candles['wick_up_big']:
            scalp_council['score_s'] += 1.2
            scalp_council['s'] += 1
        
        min_scalp_score = 4.0
        
        if scalp_council['score_b'] >= min_scalp_score and scalp_council['b'] > scalp_council['s']:
            reason = f"SCALP-BUY | score={scalp_council['score_b']:.1f} | vol={volume_ok} | mom={momentum_ok}"
            return ("buy", reason)
        
        if scalp_council['score_s'] >= min_scalp_score and scalp_council['s'] > scalp_council['b']:
            reason = f"SCALP-SELL | score={scalp_council['score_s']:.1f} | vol={volume_ok} | mom={momentum_ok}"
            return ("sell", reason)
        
        return (None, f"low_score_b={scalp_council['score_b']:.1f}_s={scalp_council['score_s']:.1f}")
        
    except Exception as e:
        return (None, f"scalp_err:{e}")

def execute_super_scalp(px_now, balance, df, ind, flow, volume_profile, momentum, spread_bps):
    global _last_scalp_ts, _scalp_profit_total
    
    if not SCALP_MODE or not SCALP_EXECUTE:
        return False
        
    if time.time() - _last_scalp_ts < SCALP_COOLDOWN_SEC:
        return False

    direction, reason = detect_super_scalp_opportunity(df, ind, flow, volume_profile, momentum, spread_bps)
    if direction is None:
        return False

    base_qty = compute_size(balance, px_now) * SCALP_SIZE_FACTOR
    volatility_factor = min(2.0, max(0.5, momentum['volatility'] / max(momentum['volatility_ma'], 1e-9)))
    smart_scalp_qty = base_qty * volatility_factor
    
    if smart_scalp_qty <= 0:
        log_w("SUPER SCALP: skip qty<=0")
        return False

    opened = open_market_enhanced(direction, smart_scalp_qty, px_now)
    if opened:
        _last_scalp_ts = time.time()
        STATE["mode"] = "super_scalp"
        
        STATE["scalp_multi_tp"] = True
        STATE["scalp_tp_levels"] = [0.15, 0.25, 0.35]
        STATE["scalp_tp_weights"] = [0.4, 0.3, 0.3]
        STATE["scalp_tp_achieved"] = [False, False, False]
        
        log_i(f"🔥 SUPER SCALP {direction.upper()} qty={smart_scalp_qty:.4f} px={px_now:.6f}")
        log_i(f"   Reason: {reason}")
        log_i(f"   Volatility Factor: {volatility_factor:.2f}")
        log_i(f"   Multi-TP: {STATE['scalp_tp_levels']}")
        
        try:
            print_position_snapshot(reason="SUPER_SCALP", color=("green" if direction=="buy" else "red"))
        except Exception as e:
            log_w(f"Snapshot error: {e}")
            
        return True
    
    return False

# =================== INTELLIGENT TREND RIDING SYSTEM ===================
def manage_trend_ride_intelligently(df, ind, info, trend_strength):
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    if mode != "trend":
        return
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    current_trend_strength = compute_trend_strength(df, ind)
    trend_multiplier = current_trend_strength["multiplier"]
    
    dynamic_tp_levels = [tp * trend_multiplier for tp in TREND_TPS]
    dynamic_tp_fractions = [frac * (2.0 if trend_multiplier > 1.5 else 1.0) for frac in TREND_TP_FRACS]
    
    for i, (tp_level, tp_frac) in enumerate(zip(dynamic_tp_levels, dynamic_tp_fractions)):
        tp_key = f"tp_{i+1}_done"
        if not STATE.get(tp_key, False) and pnl_pct >= tp_level:
            close_qty = safe_qty(STATE["qty"] * tp_frac)
            if close_qty > 0:
                close_side = "sell" if STATE["side"] == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🎯 TP{i+1} HIT: {tp_level:.2f}% | closed {tp_frac*100}% | Trend Strength: {current_trend_strength['strength']}")
                        STATE["profit_targets_achieved"] += 1
                    except Exception as e:
                        log_e(f"❌ TP{i+1} close failed: {e}")
                STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                STATE[tp_key] = True
                
                if current_trend_strength["strength"] in ["strong", "very_strong"] and i == len(dynamic_tp_levels) - 1:
                    log_i(f"💎 ترند قوي مستمر - الاحتفاظ بجزء من المركز للربح الإضافي")

    manage_intelligent_trailing_stop(px, side, ind, current_trend_strength)
    
    if TREND_REENTRY_STRATEGY and current_trend_strength["strength"] in ["strong", "very_strong"]:
        consider_trend_reentry(df, ind, px, side, current_trend_strength)

def manage_intelligent_trailing_stop(current_price, side, ind, trend_strength):
    if not STATE.get("trail_active", False):
        return
    
    atr = safe_get(ind, "atr", 0.0)
    pnl_pct = STATE.get("pnl", 0.0)
    
    if trend_strength["strength"] == "very_strong":
        trail_mult = ATR_TRAIL_MULT * 0.7
    elif trend_strength["strength"] == "strong":
        trail_mult = ATR_TRAIL_MULT * 0.8
    elif trend_strength["strength"] == "weak":
        trail_mult = ATR_TRAIL_MULT * 1.2
    else:
        trail_mult = ATR_TRAIL_MULT
    
    if pnl_pct > 2.0:
        trail_mult *= 0.9
    elif pnl_pct > 1.0:
        trail_mult *= 0.95
    
    if side == "long":
        new_trail = current_price - (atr * trail_mult)
        if STATE.get("trail") is None or new_trail > STATE["trail"]:
            STATE["trail"] = new_trail
            if STATE["trail"] > STATE.get("entry", 0):
                log_i(f"🔼 وقف متحرك محدث: {STATE['trail']:.6f} (قوة الترند: {trend_strength['strength']})")
    else:
        new_trail = current_price + (atr * trail_mult)
        if STATE.get("trail") is None or new_trail < STATE["trail"]:
            STATE["trail"] = new_trail
            if STATE["trail"] < STATE.get("entry", float('inf')):
                log_i(f"🔽 وقف متحرك محدث: {STATE['trail']:.6f} (قوة الترند: {trend_strength['strength']})")
    
    if STATE.get("trail"):
        if (side == "long" and current_price <= STATE["trail"]) or (side == "short" and current_price >= STATE["trail"]):
            log_w(f"🛑 وقف متحرك: {current_price} vs trail {STATE['trail']}")
            close_market_strict("intelligent_trailing_stop")

def consider_trend_reentry(df, ind, current_price, current_side, trend_strength):
    if STATE["qty"] > FINAL_CHUNK_QTY * 2:
        return
    
    council_data = super_council_ai_enhanced(df)
    new_side = "buy" if council_data["score_b"] > council_data["score_s"] else "sell"
    
    if (new_side == current_side and 
        trend_strength["strength"] in ["strong", "very_strong"] and
        council_data["confidence"] > 0.7):
        
        reentry_qty = compute_size(balance_usdt(), current_price) * 0.3
        
        if reentry_qty > 0:
            log_i(f"🔄 إعادة دخول في الترند {current_side.upper()} | قوة: {trend_strength['strength']}")
            open_market_enhanced(new_side, reentry_qty, current_price)

# =================== EXECUTION MANAGER ===================
def execute_trade_decision(side, price, qty, mode, council_data, gz_data):
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

def setup_trade_management(mode):
    if mode == "scalp":
        return {
            "tp1_pct": SCALP_TP_SINGLE_PCT / 100.0,
            "be_activate_pct": SCALP_BE_AFTER_PCT / 100.0,
            "trail_activate_pct": 0.8 / 100.0,
            "atr_trail_mult": SCALP_ATR_TRAIL_MULT,
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
    snap = emit_snapshots(ex, SYMBOL, df)
    
    votes = snap["cv"]
    mode_data = decide_strategy_mode(df, 
                                   adx=safe_get(votes["ind"], "adx", 0),
                                   di_plus=safe_get(votes["ind"], "plus_di", 0),
                                   di_minus=safe_get(votes["ind"], "minus_di", 0),
                                   rsi_ctx=rsi_ma_context(df))
    
    mode = mode_data["mode"]
    gz = snap["gz"]
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, mode, votes, gz)
    
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
        
        STATE["last_ind"] = votes["ind"] if isinstance(votes,dict) else {}
        STATE["last_council"] = votes
        STATE["last_flow"] = compute_flow_metrics(df)
        STATE["last_spread_bps"] = orderbook_spread_bps()
        
        log_g(f"✅ POSITION OPENED: {side.upper()} | mode={mode}")
        print_position_snapshot(reason="OPEN")
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
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "trail_tightened": False, "partial_taken": False
    })
    save_state({"in_position": False, "position_qty": 0})
    
    _arm_wait_after_close(prev_side)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_after_entry_enhanced(df, ind, info):
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    mode = STATE.get("mode", "trend")
    
    if mode == "trend":
        trend_strength = compute_trend_strength(df, ind)
        manage_trend_ride_intelligently(df, ind, info, trend_strength)
    else:
        manage_scalp_trade(df, ind, info)

def manage_scalp_trade(df, ind, info):
    px = info["price"]
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    
    pnl_pct = (px - entry) / entry * 100 * (1 if side == "long" else -1)
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct

    if STATE.get("scalp_multi_tp", False):
        for i, (tp_level, tp_weight) in enumerate(zip(STATE["scalp_tp_levels"], STATE["scalp_tp_weights"])):
            if not STATE["scalp_tp_achieved"][i] and pnl_pct >= tp_level:
                close_qty = safe_qty(STATE["qty"] * tp_weight)
                if close_qty > 0:
                    close_side = "sell" if STATE["side"] == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            log_g(f"🎯 SCALP TP{i+1} HIT: {tp_level:.2f}% | closed {tp_weight*100}%")
                            STATE["profit_targets_achieved"] += 1
                        except Exception as e:
                            log_e(f"❌ SCALP TP{i+1} close failed: {e}")
                    STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                    STATE["scalp_tp_achieved"][i] = True

    manage_scalp_trailing_stop(px, side, ind)

def manage_scalp_trailing_stop(current_price, side, ind):
    if not STATE.get("trail_active", False):
        if STATE.get("pnl", 0) >= SCALP_BE_AFTER_PCT:
            STATE["trail_active"] = True
            STATE["breakeven_armed"] = True
            STATE["breakeven"] = STATE["entry"]
            log_i("SCALP: Breakeven armed & Trail activated")

    if STATE.get("trail_active"):
        atr = safe_get(ind, "atr", 0.0)
        trail_mult = SCALP_ATR_TRAIL_MULT
        
        if side == "long":
            new_trail = current_price - (atr * trail_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:
            new_trail = current_price + (atr * trail_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail

        if STATE.get("trail"):
            if (side == "long" and current_price <= STATE["trail"]) or (side == "short" and current_price >= STATE["trail"]):
                log_w(f"SCALP TRAIL STOP: {current_price} vs trail {STATE['trail']}")
                close_market_strict("scalp_trailing_stop")

manage_after_entry = manage_after_entry_enhanced

def smart_exit_guard(state, df, ind, flow, bm, now_price, pnl_pct, mode, side, entry_price, gz=None):
    atr = safe_get(ind, 'atr', 0.0)
    adx = safe_get(ind, 'adx', 0.0)
    rsi = safe_get(ind, 'rsi', 50.0)
    rsi_ma = safe_get(ind, 'rsi_ma', 50.0)
    
    if len(df) >= 3:
        adx_slope = adx - safe_get(ind, 'adx_prev', adx)
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

    if state.get('tp1_done') and (gz and gz.get('ok')):
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

# =================== ENHANCED TRADE LOOP ===================
def trade_loop_enhanced():
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # تحديث orderbook للـFlow Boost
            try:
                STATE["last_orderbook"] = ex.fetch_order_book(SYMBOL, limit=FLOW_STACK_DEPTH)
            except Exception as e:
                log_w(f"Orderbook update failed: {e}")
            
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            if STATE["open"]:
                manage_after_entry(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    **info
                })
            
            if not STATE["open"]:
                vol_ma20 = float(ind.get("vol_ma20", 0))
                flow_ctx = compute_flow_metrics(df)
                volume_profile = compute_volume_profile(df)
                momentum = compute_momentum_indicators(df)
                
                STATE["last_ind"] = ind
                STATE["last_council"] = council_votes_pro_enhanced(df)
                STATE["last_flow"] = flow_ctx
                STATE["last_spread_bps"] = spread_bps
                
                if execute_super_scalp(px, bal, df, ind, flow_ctx, volume_profile, momentum, spread_bps):
                    continue
            
            reason = None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            
            council_data = council_votes_pro_enhanced(df)
            gz = council_data.get("gz")
            sig = None

            if (gz and gz.get("ok") and safe_get(ind, "adx",0) >= GOLDEN_ENTRY_ADX):
                if gz["zone"]["type"]=="golden_bottom" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    sig = "buy"
                    log_i(f"🎯 GOLDEN ENTRY: BUY | score={gz['score']:.1f} | منطقة ذهبية قوية")
                elif gz["zone"]["type"]=="golden_top" and gz["score"]>=GOLDEN_ENTRY_SCORE:
                    sig = "sell" 
                    log_i(f"🎯 GOLDEN ENTRY: SELL | score={gz['score']:.1f} | منطقة ذهبية قوية")

            if sig is None:
                if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 8.0:
                    sig = "buy"
                elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 8.0:
                    sig = "sell"
            
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
                            log_i(f"🎯 COUNCIL DECISION: {sig.upper()} | "
                                  f"Score B/S: {council_data['score_b']:.1f}/{council_data['score_s']:.1f} | "
                                  f"Votes B/S: {council_data['b']}/{council_data['s']} | "
                                  f"Confidence: {council_data.get('confidence', 0):.1f}")
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
        print(f"   🧮 RSI={fmt(safe_get(ind, 'rsi'))}  +DI={fmt(safe_get(ind, 'plus_di'))}  -DI={fmt(safe_get(ind, 'minus_di'))}  ADX={fmt(safe_get(ind, 'adx'))}  ATR={fmt(safe_get(ind, 'atr'))}")
        print(f"   🎯 ENTRY: SUPER COUNCIL AI + GOLDEN ENTRY + SUPER SCALP |  spread_bps={fmt(spread_bps,2)}")
        print(f"   ⏱️ closes_in ≈ {left_s}s")
        print("\n🧭 POSITION")
        bal_line = f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
        print(colored(f"   {bal_line}", "yellow"))
        if STATE["open"]:
            lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
            print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}")
            print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%")
            print(f"   🎯 MODE={STATE.get('mode', 'trend')}  Multi-TP={STATE.get('scalp_multi_tp', False)}")
        else:
            print("   ⚪ FLAT")
            if wait_for_next_signal_side:
                print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
        if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
        print(colored("─"*100,"cyan"))

# =================== API / KEEPALIVE ===================
app = Flask(__name__)

@app.get("/mark/<color>")
def mark_position(color):
    color = color.lower()
    if color not in ["green", "red"]:
        return jsonify({"ok": False, "error": "Use /mark/green or /mark/red"}), 400
    
    print_position_snapshot(reason="MANUAL_MARK", color=color)
    return jsonify({"ok": True, "marked": color, "timestamp": datetime.utcnow().isoformat()})

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI ULTRA PRO AI Bot — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode} — Super Council AI + Intelligent Trend Riding"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "entry_mode": "SUPER_COUNCIL_AI_GOLDEN_SCALP", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "scalp_mode": SCALP_MODE,
        "super_council_ai": COUNCIL_AI_MODE,
        "intelligent_trend_riding": TREND_RIDING_AI
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "exchange": EXCHANGE_NAME, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "SUPER_COUNCIL_AI_GOLDEN_SCALP", "wait_for_next_signal": wait_for_next_signal_side,
        "scalp_mode": SCALP_MODE,
        "super_council_ai": COUNCIL_AI_MODE
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

# =================== EXECUTION VERIFICATION ===================
def verify_execution_environment():
    print(f"⚙️ EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 GOLDEN ENTRY: score={GOLDEN_ENTRY_SCORE} | ADX={GOLDEN_ENTRY_ADX}", flush=True)
    print(f"📈 CANDLES: Full patterns + Wick exhaustion + Golden reversal", flush=True)
    print(f"⚡ RF SETTINGS: period={RF_PERIOD} | mult={RF_MULT} (SUI Optimized)", flush=True)
    print(f"🚀 SUPER SCALP ENGINE: {'ACTIVE' if SCALP_MODE else 'INACTIVE'}", flush=True)
    print(f"🧠 SUPER COUNCIL AI: {'ACTIVE' if COUNCIL_AI_MODE else 'INACTIVE'}", flush=True)
    print(f"💎 INTELLIGENT TREND RIDING: {'ACTIVE' if TREND_RIDING_AI else 'INACTIVE'}", flush=True)
    print(f"📊 FOOTPRINT & DIAGONAL FLOW: ACTIVE", flush=True)
    
    if not EXECUTE_ORDERS:
        print("🟡 WARNING: EXECUTE_ORDERS=False - البوت في وضع التحليل فقط!", flush=True)
    if DRY_RUN:
        print("🟡 WARNING: DRY_RUN=True - البوت في وضع المحاكاة!", flush=True)

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

# =================== BOOT ===================
if __name__ == "__main__":
    log_banner("SUI ULTRA PRO AI BOT - SUPER COUNCIL AI + INTELLIGENT TREND RIDING")
    state = load_state() or {}
    state.setdefault("in_position", False)

    if RESUME_ON_RESTART:
        try:
            state = resume_open_position(ex, SYMBOL, state)
        except Exception as e:
            log_w(f"resume error: {e}\n{traceback.format_exc()}")

    verify_execution_environment()

    print(colored(f"🎯 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • SUPER_COUNCIL_AI=ENABLED", "yellow"))
    print(colored(f"🏆 GOLDEN ENTRY: score≥{GOLDEN_ENTRY_SCORE} | ADX≥{GOLDEN_ENTRY_ADX}", "yellow"))
    print(colored(f"🕯️ CANDLES: Full patterns + Wick exhaustion + Golden reversal", "yellow"))
    print(colored(f"📊 RF SETTINGS: period={RF_PERIOD} | mult={RF_MULT} (SUI Optimized)", "yellow"))
    print(colored(f"🚀 SUPER SCALP ENGINE: {'ACTIVE' if SCALP_MODE else 'INACTIVE'}", "yellow"))
    print(colored(f"🧠 SUPER COUNCIL AI: {'ACTIVE' if COUNCIL_AI_MODE else 'INACTIVE'}", "yellow"))
    print(colored(f"💎 INTELLIGENT TREND RIDING: {'ACTIVE' if TREND_RIDING_AI else 'INACTIVE'}", "yellow"))
    print(colored(f"📊 FOOTPRINT & DIAGONAL FLOW: ACTIVE", "yellow"))
    print(colored(f"💰 PROFIT COMPOUNDING: ACTIVE", "yellow"))
    print(colored(f"⚡ EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
