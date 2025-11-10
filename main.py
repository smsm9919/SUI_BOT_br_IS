# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO — Smart Council + SMC/FVG + OrderBook Flow + Iceberg + Early-Ignition + Scale-In + Scalp Fusion
Exchange: Bybit/BingX (USDT Perp via CCXT)
"""

import os, time, json, logging, traceback, statistics
from logging.handlers import RotatingFileHandler
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from collections import deque
from datetime import datetime

import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify

# =================== LOGGING ===================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=3),
              logging.StreamHandler()]
)
def log_i(m): print(f"ℹ️ {m}", flush=True)
def log_g(m): print(f"✅ {m}", flush=True)
def log_w(m): print(f"🟨 {m}", flush=True)
def log_e(m): print(f"❌ {m}", flush=True)

# =================== ENV / MODE ===================
EXCHANGE_NAME = os.getenv("EXCHANGE", "bybit").lower()  # bybit | bingx
if EXCHANGE_NAME == "bybit":
    API_KEY, API_SECRET = os.getenv("BYBIT_API_KEY",""), os.getenv("BYBIT_API_SECRET","")
else:
    API_KEY, API_SECRET = os.getenv("BINGX_API_KEY",""), os.getenv("BINGX_API_SECRET","")

MODE_LIVE = bool(API_KEY and API_SECRET)
SELF_URL = os.getenv("SELF_URL","") or os.getenv("RENDER_EXTERNAL_URL","")
PORT = int(os.getenv("PORT", 5000))

SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

EXECUTE_ORDERS = True
DRY_RUN = False
LOG_MEDIUM_PANEL = True

BOT_VERSION = f"SUI ULTRA PRO — {EXCHANGE_NAME.upper()}"

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True

# =================== SETTINGS ===================
# Profit Mgmt
SCALP_TP_TARGETS=[0.55,0.90,1.20]
SCALP_TP_FRACTIONS=[0.50,0.30,0.20]
TREND_TP_TARGETS=[0.60,1.50,2.50,4.00]
TREND_TP_FRACTIONS=[0.30,0.30,0.25,0.15]
BREAKEVEN_AFTER=0.35
TRAIL_ACTIVATE_PCT=1.0
ATR_TRAIL_MULT=1.8

# Council thresholds
REQUIRED_CONFIRMATIONS=4
TREND_STRENGTH_ADX=25
VOLUME_CONFIRMATION=1.2
GOLDEN_ENTRY_ADX=22.0

# Spread Gate (bps = 0.01%)
MAX_SPREAD_BPS=float(os.getenv("MAX_SPREAD_BPS", "8"))

# Indicators core
RSI_LEN=14; ADX_LEN=14; ATR_LEN=14
ICHIMOKU=(9,26,52); BB=(20,2)

# FVG / OB / FLOW / ICEBERG
FVG_LOOKBACK=60; FVG_MIN_SIZE=0.0015; FVG_NEAR_PCT=0.15
OB_LOOKBACK=80; OB_MIN_DISP=1.0; OB_NEAR_PCT=0.20
ORDERBOOK_LIMIT=50; IMBALANCE_THRESH=0.20; WALL_X_SIGMA=2.2
ICEBERG_WINDOW=6; ICEBERG_LEVELS=5; ICEBERG_REFILL_RATIO=0.60
ICEBERG_TRADE_NEAR_PCT=0.08; ICEBERG_MIN_TICKS=2

# Early ignition / Scale-in
PROBE_SIZE = 0.35   # 35% من حجم الصفقة
SCALE_IN_ADD = 0.30 # 30% تعزيز
SCALE_IN_MAX = 2    # أقصى عدد تعزيزات
PULLBACK_FOR_ADD = 0.30  # %0.30 تراجع مقبول للتعزيز

# ==== SCALP COUNCIL FUSION (inside-council) ====
SCALP_MIN_ADX        = 18        # بوابة زخم دنيا للسكالب
SCALP_CHOP_ADX       = 15        # اعتبار المنطقة "تشوب" لو ADX ضعيف + Squeeze
SCALP_NEED_SMC_NEAR  = True      # يفضّل FVG/OB قريب
SCALP_NEED_FLOW      = True      # يفضّل Flow (Iceberg/Wall/Imbalance)
SCALP_IMPULSE_ATR    = 0.90      # شمعة اندفاع: جسم ≥ 0.9×ATR
SCALP_LONG_WICK_R    = 0.60      # فتيلة طويلة: ≥ 60% من الرينج
SCALP_BODY_MAX_FOR_W = 0.35      # جسم صغير مع فتيلة ⇒ حصاد سيولة

# جدوى الربح (داخل المجلس)
SCALP_MIN_PROFIT_PCT = 0.45      # أقل هدف منطقي بعد التكاليف
SCALP_MIN_RR         = 1.30      # أقل عائد/مخاطرة
TAKER_FEE_PCT        = 0.10      # % تقدير رسوم تاكر (عدّل لمنصتك)
EXTRA_SLIPPAGE_PCT   = 0.05      # % احتياطي انزلاق

# =================== EXCHANGE ===================
def make_ex():
    cfg={"apiKey":API_KEY,"secret":API_SECRET,"enableRateLimit":True,"timeout":20000,"options":{"defaultType":"swap"}}
    return ccxt.bybit(cfg) if EXCHANGE_NAME=="bybit" else ccxt.bingx(cfg)
ex = make_ex()

MARKET={}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None
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

def exchange_specific_params(side, is_close=False):
    if EXCHANGE_NAME=="bybit":
        if POSITION_MODE=="hedge":
            return {"positionSide": "Long" if side=="buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE=="hedge":
            return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, lv, symbol):
    try:
        if EXCHANGE_NAME=="bybit": exchange.set_leverage(lv, symbol)
        else: exchange.set_leverage(lv, symbol, params={"side":"BOTH"})
        log_g(f"Leverage set: {lv}x")
    except Exception as e:
        log_w(f"set_leverage: {e}")

try:
    load_market_specs()
    exchange_set_leverage(ex, LEVERAGE, SYMBOL)
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== STATE ===================
STATE = {"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
         "trail":None,"breakeven":None,"tp_levels_hit":[],"profit_targets_achieved":0,
         "highest_profit_pct":0.0,"mode":None,"adds":0}

def save_state(st:dict):
    try:
        st["ts"]=int(time.time()); st["compound_pnl"]=round(float(compound_pnl),6)
        with open(STATE_PATH,"w",encoding="utf-8") as f: json.dump(st,f,ensure_ascii=False,indent=2)
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state()->dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH,"r",encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

st_boot = load_state() or {}
try:
    compound_pnl = float(st_boot.get("compound_pnl", 0.0))
    log_i(f"💰 compound_pnl restored: {compound_pnl:.4f} USDT")
except Exception:
    compound_pnl = 0.0

wait_for_next_signal_side=None

# =================== HELPERS ===================
def _round_amt(q):
    if q is None: return 0.0
    try:
        d=Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step=Decimal(str(LOT_STEP)); d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec=int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d=d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d<Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation,ValueError,TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q=_round_amt(q)
    if q<=0: log_w(f"qty invalid after normalize → {q}")
    return q

def price_now():
    try:
        t=ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def fetch_ohlcv(limit=600):
    rows=ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def fetch_multi_timeframe(tfs=("5m","15m","1h")):
    mtf={}
    for tf in tfs:
        try:
            rows=ex.fetch_ohlcv(SYMBOL, timeframe=tf, limit=100, params={"type":"swap"})
            mtf[tf]=pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
        except Exception as e: log_w(f"fetch {tf}: {e}")
    sigs={}; strengths={}
    for tf,df in mtf.items():
        if len(df)<50: continue
        c=df["close"].astype(float)
        sma20=c.rolling(20).mean().iloc[-1]; sma50=c.rolling(50).mean().iloc[-1]
        cur=c.iloc[-1]
        trend="bullish" if (cur> sma20> sma50) else ("bearish" if (cur< sma20< sma50) else "neutral")
        sigs[tf]=trend; strengths[tf]=abs(cur-sma50)/max(sma50,1e-9)*100
    bull=sum(1 for s in sigs.values() if s=="bullish"); bear=sum(1 for s in sigs.values() if s=="bearish")
    overall="bullish" if bull>=2 else ("bearish" if bear>=2 else "neutral")
    return {"frames":mtf,"signals":sigs,"strengths":strengths,"overall_signal":overall,"bull_count":bull,"bear_count":bear,"confidence":(max(bull,bear)/max(len(sigs),1))}

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw=max(0.0, cap/max(float(price or 0.0),1e-9))
    return safe_qty(raw)

def spread_bps():
    try:
        ob=ex.fetch_order_book(SYMBOL, limit=5)
        bid=ob["bids"][0][0]; ask=ob["asks"][0][0]
        return (ask-bid)/((ask+bid)/2)*10000.0
    except Exception: return 0.0

# =================== SCALP ENHANCEMENT FUNCTIONS ===================
def analyze_candle_strength(df):
    """تحليل قوة الشمعة الحالية للكشف عن الاندفاع أو حصاد الفتائل"""
    if len(df) < 20:
        return {"impulse": False, "harvest": False, "body_ratio": 0, "wick_ratio": 0, "direction": "bull", "strong": False}
    
    try:
        c, o, h, l = map(float, (df["close"].iloc[-1], df["open"].iloc[-1], df["high"].iloc[-1], df["low"].iloc[-1]))
        atr = compute_indicators(df).get("atr", 0) or 0.001
        rng = (h - l) if (h > l) else 1e-9
        body = abs(c - o)
        body_ratio = body / rng
        up_w = (h - max(c, o)) / rng
        dn_w = (min(c, o) - l) / rng
        wick_ratio = max(up_w, dn_w)

        impulse = (body >= atr * SCALP_IMPULSE_ATR) and (body_ratio > 0.55)
        harvest = (wick_ratio >= SCALP_LONG_WICK_R) and (body_ratio <= SCALP_BODY_MAX_FOR_W)
        direction = "bull" if c > o else "bear"
        
        return {
            "impulse": impulse,
            "harvest": harvest, 
            "body_ratio": round(body_ratio, 3),
            "wick_ratio": round(wick_ratio, 3),
            "direction": direction,
            "strong": (impulse or harvest)
        }
    except Exception as e:
        log_w(f"analyze_candle_strength error: {e}")
        return {"impulse": False, "harvest": False, "body_ratio": 0, "wick_ratio": 0, "direction": "bull", "strong": False}

def detect_liquidity_sweep(df, lookback=20):
    """كشف سيولة ممسوحة (Liquidity Sweep)"""
    try:
        if len(df) < lookback + 5: 
            return {"has": False}
            
        hi = float(df["high"].iloc[-1])
        lo = float(df["low"].iloc[-1])
        prev_hi = float(df["high"].tail(lookback+1).head(lookback).max())
        prev_lo = float(df["low"].tail(lookback+1).head(lookback).min())
        c, o = float(df["close"].iloc[-1]), float(df["open"].iloc[-1])
        
        touched_prev_high = hi >= prev_hi and c < prev_hi and (hi - max(c, o)) / max(hi - lo, 1e-9) >= 0.5
        touched_prev_low = lo <= prev_lo and c > prev_lo and (min(c, o) - lo) / max(hi - lo, 1e-9) >= 0.5
        
        if touched_prev_high: 
            return {"has": True, "type": "buy_sweep"}   # سحب سيولة فوق ⇒ ميل بيع
        if touched_prev_low:  
            return {"has": True, "type": "sell_sweep"}  # سحب سيولة تحت ⇒ ميل شراء
            
        return {"has": False}
    except Exception as e:
        log_w(f"detect_liquidity_sweep error: {e}")
        return {"has": False}

def expected_move_and_cost(council, df, price):
    """تقدير الحركة المتوقعة والتكاليف"""
    try:
        ind = council.get("indicators", {})
        atr = ind.get("basic", {}).get("atr", 0) or compute_indicators(df).get("atr", 0) or 0.001
        flow = ind.get("orderbook_flow", {}) or {}
        wall = ind.get("orderbook_wall", {}) or {}
        ice = ind.get("iceberg", {}) or {}

        flow_boost = 0.20 if flow.get("signal") in ("buy", "sell") else 0.0
        wall_boost = 0.25 if wall.get("has") and wall.get("strong") else 0.0
        ice_boost = min(0.35, (ice.get("strength", 0) / 3.0)) if ice.get("has") else 0.0

        exp_move_pct = ((atr / max(price, 1e-9)) * 100.0) * 0.45 + (flow_boost + wall_boost + ice_boost)
        cost_pct = (TAKER_FEE_PCT * 2.0) + (spread_bps() / 100.0) + EXTRA_SLIPPAGE_PCT
        
        return round(exp_move_pct, 3), round(cost_pct, 3)
    except Exception as e:
        log_w(f"expected_move_and_cost error: {e}")
        return 0.5, 0.3

def scalp_council_fusion(df, council, price):
    """مقيّم السكالب داخل المجلس - ينتج تقييم شامل للصفقة"""
    reasons = []
    ind = council.get("indicators", {})
    basic = ind.get("basic", {})
    adx = float(basic.get("adx", 0))
    bb = ind.get("bollinger", {})
    squeeze = bool(bb.get("squeeze", False))
    fvg = ind.get("fvg", {})
    obx = ind.get("order_block", {})
    flow = ind.get("orderbook_flow", {})
    ice = ind.get("iceberg", {})
    wall = ind.get("orderbook_wall", {})
    mtf = council.get("mtf_analysis", {}) or {}
    candles = ind.get("candles", {}) or analyze_candle_strength(df)
    sweep = ind.get("sweep", {}) or detect_liquidity_sweep(df)

    # === البوابات الأساسية ===
    if adx < SCALP_MIN_ADX: 
        reasons.append("weak_adx")
    if squeeze and adx <= SCALP_CHOP_ADX: 
        reasons.append("chop_squeeze")
    if SCALP_NEED_SMC_NEAR and not ((fvg.get("has") and fvg.get("near")) or (obx.get("has") and obx.get("near"))):
        reasons.append("no_smc_near")
    if SCALP_NEED_FLOW and not ((flow.get("signal") in ("buy", "sell")) or (ice.get("has")) or (wall.get("has"))):
        reasons.append("no_flow")

    # === اتجاه اللحظة من الأطر الزمنية ===
    scalp_bias = "neutral"
    if mtf.get("signals", {}):
        s5 = mtf["signals"].get("5m", "neutral")
        s15 = mtf["signals"].get("15m", "neutral")
        if s5 == "bullish" and s15 != "bearish": 
            scalp_bias = "buy"
        if s5 == "bearish" and s15 != "bullish": 
            scalp_bias = "sell"

    # === نظام النقاط للشموع والإشارات ===
    grade = 0.0
    
    # الشموع القوية
    if candles.get("impulse"):
        grade += 1.6
        reasons.append(f"impulse_{candles.get('direction')}")
    if candles.get("harvest"):
        grade += 1.2
        reasons.append(f"harvest_{candles.get('direction')}")
    
    # السيولة الممسوحة
    if sweep.get("has"):
        if sweep["type"] == "sell_sweep": 
            grade += 1.0
            reasons.append("sweep_buy")   # ميل شراء
        if sweep["type"] == "buy_sweep":  
            grade += 1.0
            reasons.append("sweep_sell")  # ميل بيع

    # إشارات SMC/Flow الإضافية
    if fvg.get("has"): 
        grade += 0.8
    if obx.get("has"): 
        grade += 1.0
    if flow.get("signal") in ("buy", "sell"): 
        grade += 0.6
    if ice.get("has"): 
        grade += 0.6 + min(0.6, ice.get("strength", 0) / 2.0)
    if wall.get("has") and wall.get("strong"): 
        grade += 0.6

    # مواءمة الأطر الزمنية
    if scalp_bias == "buy": 
        grade += 0.7
    if scalp_bias == "sell": 
        grade += 0.7

    # === تحليل الجدوى المالية ===
    exp, cost = expected_move_and_cost(council, df, price)
    est_sl_pct = (ind.get("basic", {}).get("atr", 0) or compute_indicators(df).get("atr", 0) or 0.001) / max(price, 1e-9) * 100.0 * 0.80
    est_tp_pct = max(SCALP_MIN_PROFIT_PCT + cost, exp * 0.70)
    rr = est_tp_pct / max(est_sl_pct, 0.01)
    
    if est_tp_pct < (SCALP_MIN_PROFIT_PCT + cost): 
        reasons.append("low_exp_move")
    if rr < SCALP_MIN_RR: 
        reasons.append("low_rr")

    # === القرار النهائي ===
    ok = (len(reasons) == 0) or (("impulse" in "_".join(reasons) or "harvest" in "_".join(reasons)) and rr >= SCALP_MIN_RR)

    return {
        "grade": round(grade, 2),
        "ok": bool(ok),
        "bias": scalp_bias,
        "reasons": reasons,
        "exp_move_pct": exp,
        "cost_pct": cost,
        "rr": round(rr, 2),
        "candles": candles,
        "sweep": sweep
    }

# =================== INDICATORS / MODULES ===================
def wilder_ema(s:pd.Series,n:int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df:pd.DataFrame):
    if len(df)<60:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"adx":0.0,"atr":0.0}
    c,h,l=df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr=pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr=wilder_ema(tr, ATR_LEN)
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs=wilder_ema(up, RSI_LEN)/wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi=100-(100/(1+rs))
    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)
    return {"rsi":float(rsi.iloc[-1]),"plus_di":float(plus_di.iloc[-1]),
            "minus_di":float(minus_di.iloc[-1]),"adx":float(adx.iloc[-1]),"atr":float(atr.iloc[-1])}

def rsi_ma_context(df:pd.DataFrame):
    RSI_MA=9; NEUT=(45,55); PERSIST=3
    if len(df)<30: return {"rsi":50,"rsi_ma":50,"cross":"none","trendZ":"none","in_chop":True}
    c=df["close"].astype(float); delta=c.diff(); gain=delta.clip(lower=0).ewm(span=14).mean()
    loss=(-delta.clip(upper=0)).abs().ewm(span=14).mean(); rs=gain/loss.replace(0,1e-12)
    rsi=100-(100/(1+rs)); rsi_ma=rsi.rolling(RSI_MA, min_periods=1).mean()
    cross="none"
    if (rsi.iloc[-2]<=rsi_ma.iloc[-2]) and (rsi.iloc[-1]>rsi_ma.iloc[-1]): cross="bull"
    elif (rsi.iloc[-2]>=rsi_ma.iloc[-2]) and (rsi.iloc[-1]<rsi_ma.iloc[-1]): cross="bear"
    persist_b=(rsi.tail(PERSIST)>rsi_ma.tail(PERSIST)).all()
    persist_s=(rsi.tail(PERSIST)<rsi_ma.tail(PERSIST)).all()
    cur=float(rsi.iloc[-1]); in_chop = NEUT[0]<=cur<=NEUT[1]
    return {"rsi":cur,"rsi_ma":float(rsi_ma.iloc[-1]),
            "cross":cross,"trendZ":"bull" if persist_b else ("bear" if persist_s else "none"),
            "in_chop":in_chop}

def super_trend(df, period=10, multiplier=3):
    try:
        h,l,c=df['high'].astype(float),df['low'].astype(float),df['close'].astype(float)
        tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr=tr.ewm(span=period).mean(); hl2=(h+l)/2
        ub=hl2+(multiplier*atr); lb=hl2-(multiplier*atr)
        st=[ub.iloc[0]]; trend=[-1]
        for i in range(1,len(df)):
            if (st[i-1]==ub.iloc[i-1] and c.iloc[i] <= ub.iloc[i]) or (st[i-1]==lb.iloc[i-1] and c.iloc[i] < lb.iloc[i]):
                st.append(ub.iloc[i]); trend.append(-1)
            else:
                st.append(lb.iloc[i]); trend.append(1)
        strength=abs(c.iloc[-1]-st[-1])/max(atr.iloc[-1],1e-9)
        sig="buy" if trend[-1]>0 else "sell"
        return {"trend":trend[-1],"strength":float(strength),"value":float(st[-1]),"signal":sig}
    except Exception: return {"trend":0,"strength":0,"value":0,"signal":"neutral"}

def ichimoku_cloud(df, tenkan=9,kijun=26,senkou=52):
    try:
        h=df['high'].astype(float); l=df['low'].astype(float); c=df['close'].astype(float)
        ten=(h.rolling(tenkan).max()+l.rolling(tenkan).min())/2
        kij=(h.rolling(kijun).max()+l.rolling(kijun).min())/2
        sA=((ten+kij)/2).shift(kij); sB=((h.rolling(senkou).max()+l.rolling(senkou).min())/2).shift(kij)
        price=c.iloc[-1]; above=price>max(sA.iloc[-1],sB.iloc[-1]); below=price<min(sA.iloc[-1],sB.iloc[-1])
        tk_bull=ten.iloc[-1]>kij.iloc[-1]; price_abv_kij=price>kij.iloc[-1]
        sig="buy" if (above and tk_bull and price_abv_kij) else ("sell" if (below and (not tk_bull) and (not price_abv_kij)) else "neutral")
        return {"signal":sig,"strength":abs(price-kij.iloc[-1])/max(price,1e-9)*100}
    except Exception: return {"signal":"neutral","strength":0}

def bollinger_bands_advanced(df, period=20, std=2):
    try:
        c=df['close'].astype(float); sma=c.rolling(period).mean(); sd=c.rolling(period).std()
        ub=sma+(sd*std); lb=sma-(sd*std); cur=c.iloc[-1]
        bw=((ub-lb)/sma.replace(0,1e-12)*100).iloc[-1]; pctB=(cur-lb.iloc[-1])/max(ub.iloc[-1]-lb.iloc[-1],1e-12)
        squeeze=bw<10
        sig="buy" if (cur<=lb.iloc[-1] and not squeeze) else ("sell" if (cur>=ub.iloc[-1] and not squeeze) else "neutral")
        return {"percent_b":float(pctB),"squeeze":bool(squeeze),"signal":sig,"strength":abs(pctB-0.5)*2}
    except Exception: return {"signal":"neutral","strength":0}

def volume_weighted_macd(df, fast=12, slow=26, signal=9):
    try:
        c=df['close'].astype(float); v=df['volume'].astype(float)
        vwp=(c*v).cumsum()/v.cumsum()
        ema_f=vwp.ewm(span=fast).mean(); ema_s=vwp.ewm(span=slow).mean()
        macd=ema_f-ema_s; macd_sig=macd.ewm(span=signal).mean(); hist=macd-macd_sig
        bull=(macd.iloc[-1]>macd_sig.iloc[-1]) and (hist.iloc[-1]>hist.iloc[-2])
        bear=(macd.iloc[-1]<macd_sig.iloc[-1]) and (hist.iloc[-1]<hist.iloc[-2])
        sig="buy" if bull else ("sell" if bear else "neutral")
        strength=abs(hist.iloc[-1])/max(c.rolling(50).std().iloc[-1],1e-12)
        return {"histogram":float(hist.iloc[-1]),"signal":sig,"strength":float(strength)}
    except Exception: return {"signal":"neutral","strength":0}

def stochastic_rsi_advanced(df, rsi_length=14, stoch_length=14, k=3, d=3):
    try:
        c=df['close'].astype(float); delta=c.diff(); gain=(delta.where(delta>0,0)).rolling(rsi_length).mean()
        loss=(-delta.where(delta<0,0)).rolling(rsi_length).mean(); rs=gain/ loss.replace(0,1e-12)
        rsi=100-(100/(1+rs)); rsi_min=rsi.rolling(stoch_length).min(); rsi_max=rsi.rolling(stoch_length).max()
        stoch=(rsi-rsi_min)/ (rsi_max-rsi_min).replace(0,1e-12); k_=stoch.rolling(k).mean()*100; d_=k_.rolling(d).mean()
        bull=(k_.iloc[-1]>d_.iloc[-1] and k_.iloc[-2]<=d_.iloc[-2] and k_.iloc[-1]<30)
        bear=(k_.iloc[-1]<d_.iloc[-1] and k_.iloc[-2]>=d_.iloc[-2] and k_.iloc[-1]>70)
        sig="buy" if bull else ("sell" if bear else "neutral")
        return {"k":float(k_.iloc[-1]),"d":float(d_.iloc[-1]),"signal":sig,"strength":abs(k_.iloc[-1]-50)/50}
    except Exception: return {"signal":"neutral","strength":0}

def market_structure_break_advanced(df, lookback=50):
    try:
        h=df['high'].astype(float); l=df['low'].astype(float); c=df['close'].astype(float); v=df['volume'].astype(float)
        prev_h=h.tail(lookback+10).head(lookback).max(); prev_l=l.tail(lookback+10).head(lookback).min()
        cur=c.iloc[-1]; vol=v.iloc[-1]; avg=v.tail(lookback).mean()
        break_high=cur>prev_h; break_low=cur<prev_l; vol_ok=vol>avg*VOLUME_CONFIRMATION
        if break_high and vol_ok: return {"signal":"strong_buy","strength":(cur-prev_h)/max(prev_h,1e-9)*100}
        if break_low and vol_ok:  return {"signal":"strong_sell","strength":(prev_l-cur)/max(prev_l,1e-9)*100}
        return {"signal":"neutral","strength":0}
    except Exception: return {"signal":"neutral","strength":0}

def smart_money_flow(df, period=20):
    try:
        h=df['high'].astype(float); l=df['low'].astype(float); c=df['close'].astype(float); v=df['volume'].astype(float)
        mfm=((c-l)-(h-c))/ (h-l).replace(0,1e-12); mfv=mfm*v
        pos=mfv.where(mfv>0,0).rolling(period).sum(); neg=abs(mfv.where(mfv<0,0)).rolling(period).sum()
        mfi=100-(100/(1+pos/ neg.replace(0,1e-12)))
        return {"mfi":float(mfi.iloc[-1]),
                "smart_money_bullish": mfi.iloc[-1]<30,
                "smart_money_bearish": mfi.iloc[-1]>70,
                "strength":abs(mfi.iloc[-1]-50)/50}
    except Exception: return {"mfi":50,"smart_money_bullish":False,"smart_money_bearish":False,"strength":0}

# FVG / OB / FLOW / ICEBERG
def detect_fvg(df, lookback=FVG_LOOKBACK, min_size=FVG_MIN_SIZE):
    try:
        if len(df)<10: return {"has":False}
        h=df["high"].astype(float).values; l=df["low"].astype(float).values; c=df["close"].astype(float).values
        last=len(df)-1; start=max(2, last-lookback); gaps=[]
        for i in range(start,last+1):
            if l[i]>h[i-2]:
                gap=(l[i]-h[i-2])/max(c[i],1e-9)
                if gap>=min_size: gaps.append(("bull", h[i-2], l[i], gap, i))
            if h[i]<l[i-2]:
                gap=(l[i-2]-h[i])/max(c[i],1e-9)
                if gap>=min_size: gaps.append(("bear", h[i], l[i-2], gap, i))
        if not gaps: return {"has":False}
        px=float(df["close"].iloc[-1])
        typ,top,bot,gap,_=min(gaps, key=lambda t:min(abs(px-t[1]),abs(px-t[2])))
        near=(abs(px-top)/px<=FVG_NEAR_PCT/100.0) or (abs(px-bot)/px<=FVG_NEAR_PCT/100.0)
        return {"has":True,"type":typ,"top":top,"bottom":bot,"gap":gap,"near":near}
    except Exception: return {"has":False}

def detect_order_blocks(df, lookback=OB_LOOKBACK, min_disp=OB_MIN_DISP):
    try:
        if len(df)<lookback+10: return {"has":False}
        o=df["open"].astype(float).values; h=df["high"].astype(float).values
        l=df["low"].astype(float).values; c=df["close"].astype(float).values; px=c[-1]
        closes=df["close"].astype(float); vol_std=max(closes.tail(20).std(),1e-9)
        bull=None; bear=None; start=len(df)-lookback-1
        for i in range(start,len(df)-2):
            if c[i]<o[i]:
                disp_up=(c[i+2]-h[i])/vol_std
                if disp_up>=min_disp: bull=(l[i],o[i])
            if c[i]>o[i]:
                disp_dn=(l[i]-c[i+2])/vol_std
                if disp_dn>=min_disp: bear=(o[i],h[i])
        res={"has":False}
        if bull:
            low,high=bull; near=(low<=px<=high) or (min(abs(px-low),abs(px-high))/px<=OB_NEAR_PCT/100.0)
            res={"has":True,"type":"bull","low":low,"high":high,"near":near}
        if bear:
            low,high=bear; near=(low<=px<=high) or (min(abs(px-low),abs(px-high))/px<=OB_NEAR_PCT/100.0)
            if (not res["has"]) or (min(abs(px-res["low"]),abs(px-res["high"]))>min(abs(px-low),abs(px-high))):
                res={"has":True,"type":"bear","low":low,"high":high,"near":near}
        return res
    except Exception: return {"has":False}

def fetch_orderbook_metrics():
    try:
        ob=ex.fetch_order_book(SYMBOL, limit=ORDERBOOK_LIMIT)
        bids=ob.get("bids",[]) or []; asks=ob.get("asks",[]) or []
        if not bids or not asks: return {"signal":"neutral","imbalance":0.0,"bid_wall":0.0,"ask_wall":0.0}
        bid_sum=sum(b[1] for b in bids); ask_sum=sum(a[1] for a in asks); imb=(bid_sum-ask_sum)/max(bid_sum+ask_sum,1e-9)
        b_sizes=[b[1] for b in bids]; a_sizes=[a[1] for a in asks]
        b_mu,b_sd=(statistics.mean(b_sizes), statistics.pstdev(b_sizes) or 1e-9)
        a_mu,a_sd=(statistics.mean(a_sizes), statistics.pstdev(a_sizes) or 1e-9)
        bid_wall=max(0.0,(max(b_sizes)-b_mu)/b_sd); ask_wall=max(0.0,(max(a_sizes)-a_mu)/a_sd)
        sig="neutral"
        if imb> IMBALANCE_THRESH or bid_wall>=WALL_X_SIGMA: sig="buy"
        if imb< -IMBALANCE_THRESH or ask_wall>=WALL_X_SIGMA: sig="sell"
        return {"signal":sig,"imbalance":float(imb),"bid_wall":float(bid_wall),"ask_wall":float(ask_wall)}
    except Exception: return {"signal":"neutral","imbalance":0.0,"bid_wall":0.0,"ask_wall":0.0}

# Iceberg buffers
OB_SNAPSHOTS = deque(maxlen=ICEBERG_WINDOW)
TRADES_BUF   = deque(maxlen=300)

def update_orderflow_buffers():
    try:
        ob = ex.fetch_order_book(SYMBOL, limit=max(10, ICEBERG_LEVELS))
        OB_SNAPSHOTS.append({
            "ts": time.time(),
            "bids": ob.get("bids", [])[:ICEBERG_LEVELS],
            "asks": ob.get("asks", [])[:ICEBERG_LEVELS],
        })
    except Exception as e:
        log_w(f"orderbook snap fail: {e}")
    try:
        trades = ex.fetch_trades(SYMBOL, limit=100)
        now = time.time()
        for t in trades or []:
            TRADES_BUF.append({
                "ts": t.get("timestamp", 0)/1000 if t.get("timestamp") else now,
                "price": float(t.get("price", 0) or 0),
                "amount": float(t.get("amount", 0) or 0),
                "side": t.get("side", None)
            })
        while TRADES_BUF and (now - TRADES_BUF[0]["ts"] > 300):
            TRADES_BUF.popleft()
    except Exception as e:
        log_w(f"fetch_trades fail: {e}")

def detect_iceberg():
    if len(OB_SNAPSHOTS) < max(3, ICEBERG_MIN_TICKS):
        return {"has": False}
    bids_series = list(zip(*[snap["bids"] for snap in OB_SNAPSHOTS if snap["bids"]]))
    asks_series = list(zip(*[snap["asks"] for snap in OB_SNAPSHOTS if snap["asks"]]))
    def level_stats(series, side):
        out=[]
        for lvl_snaps in series[:ICEBERG_LEVELS]:
            prices=[s[0] for s in lvl_snaps]; qtys=[s[1] for s in lvl_snaps]
            if not prices or not qtys: continue
            px_now=prices[-1]; q_now=qtys[-1]; q_max=max(qtys); q_min=min(qtys)
            near_th = px_now * ICEBERG_TRADE_NEAR_PCT / 100.0
            vol_near=sum(t["amount"] for t in TRADES_BUF if abs(t["price"]-px_now)<=near_th)
            refilled = (q_now >= q_max*ICEBERG_REFILL_RATIO) and (q_max > q_min*1.05)
            # ✅ التصحيح - إضافة القوس المفقود
            out.append({"px":px_now,"q_now":q_now,"q_max":q_max,"vol_near":vol_near,"refilled":refilled})
        return out
    def best(stats):
        cand=[ (s["vol_near"] / max(s["q_max"],1e-9), s) for s in stats if s["refilled"] ]
        return max(cand, default=(0.0, None), key=lambda x: x[0])
    z_bid,s_bid = best(level_stats(bids_series,"bid"))
    z_ask,s_ask = best(level_stats(asks_series,"ask"))
    strong_bid = s_bid is not None and z_bid >= 0.35 and len(OB_SNAPSHOTS) >= ICEBERG_MIN_TICKS
    strong_ask = s_ask is not None and z_ask >= 0.35 and len(OB_SNAPSHOTS) >= ICEBERG_MIN_TICKS
    if strong_bid and (not strong_ask or z_bid>=z_ask):
        return {"has":True,"side":"bid","price":s_bid["px"],"strength":float(min(z_bid*3,3.0))}
    if strong_ask:
        return {"has":True,"side":"ask","price":s_ask["px"],"strength":float(min(z_ask*3,3.0))}
    return {"has":False}

# =================== COUNCIL ===================
def ultra_intelligent_council(df, mtf_meta=None):
    try:
        st = super_trend(df); ichi = ichimoku_cloud(df); bb = bollinger_bands_advanced(df)
        macd = volume_weighted_macd(df); stoch = stochastic_rsi_advanced(df)
        structure = market_structure_break_advanced(df); money = smart_money_flow(df)
        ind_basic = compute_indicators(df); rsi_ctx = rsi_ma_context(df)
        fvg = detect_fvg(df); obx = detect_order_blocks(df); flow = fetch_orderbook_metrics()
        iceberg = detect_iceberg()

        # === إضافة تحليل السكالب ===
        candle_ctx = analyze_candle_strength(df)
        sweep_ctx = detect_liquidity_sweep(df)

        votes_b=votes_s=0; score_b=score_s=0.0; logs=[]; confirms=[]

        def add(sig, buy_pts, sell_pts, text):
            nonlocal votes_b, votes_s, score_b, score_s, logs, confirms
            if sig=="buy": votes_b+=buy_pts[0]; score_b+=buy_pts[1]; logs.append(text); confirms.append(text.split()[0])
            elif sig=="sell": votes_s+=sell_pts[0]; score_s+=sell_pts[1]; logs.append(text); confirms.append(text.split()[0])

        add(st["signal"], (2, st["strength"]*2), (2, st["strength"]*2), f"🚀 ST {st['signal']}({st['strength']:.2f})")
        add(ichi["signal"], (2, ichi["strength"]*1.5), (2, ichi["strength"]*1.5), f"☁️ Ichi {ichi['signal']}({ichi['strength']:.2f})")
        if bb["signal"]!="neutral" and not bb["squeeze"]:
            add(bb["signal"], (2, bb["strength"]*2), (2, bb["strength"]*2), f"📊 BB {bb['signal']}")
        add(macd["signal"], (2, macd["strength"]*1.5), (2, macd["strength"]*1.5), f"📈 MACD {macd['signal']}")
        add(stoch["signal"], (1, stoch["strength"]*1.2), (1, stoch["strength"]*1.2), f"🎯 Stoch {stoch['signal']}")
        if structure["signal"]=="strong_buy": votes_b+=3; score_b+=structure["strength"]*0.5; logs.append(f"🔄 Break↑ {structure['strength']:.2f}"); confirms.append("Structure")
        if structure["signal"]=="strong_sell": votes_s+=3; score_s+=structure["strength"]*0.5; logs.append(f"🔄 Break↓ {structure['strength']:.2f}"); confirms.append("Structure")
        if money["smart_money_bullish"]: votes_b+=2; score_b+=money["strength"]*1.8; logs.append("💰 SmartMoney buy"); confirms.append("SmartMoney")
        if money["smart_money_bearish"]: votes_s+=2; score_s+=money["strength"]*1.8; logs.append("💰 SmartMoney sell"); confirms.append("SmartMoney")

        if mtf_meta:
            if mtf_meta["overall_signal"]=="bullish": votes_b+=2; score_b+=mtf_meta["confidence"]*2; logs.append(f"⏰ MTF bull({mtf_meta['bull_count']})"); confirms.append("MultiTF")
            if mtf_meta["overall_signal"]=="bearish": votes_s+=2; score_s+=mtf_meta["confidence"]*2; logs.append(f"⏰ MTF bear({mtf_meta['bear_count']})"); confirms.append("MultiTF")

        if ind_basic["adx"]>TREND_STRENGTH_ADX:
            if votes_b>votes_s: score_b*=1.3; logs.append(f"🔥 Trend+ ADX={ind_basic['adx']:.1f}")
            elif votes_s>votes_b: score_s*=1.3; logs.append(f"🔥 Trend- ADX={ind_basic['adx']:.1f}")

        # FVG/OB/Flow/Iceberg
        if fvg.get("has"):
            if fvg["type"]=="bull": votes_b+=2; score_b+=1.0+(0.5 if fvg.get("near") else 0); logs.append("🧩 FVG bull"); confirms.append("FVG")
            if fvg["type"]=="bear": votes_s+=2; score_s+=1.0+(0.5 if fvg.get("near") else 0); logs.append("🧩 FVG bear"); confirms.append("FVG")
        if obx.get("has"):
            if obx["type"]=="bull": votes_b+=3; score_b+=1.8 if obx.get("near") else 1.0; logs.append("🏦 OB bull"); confirms.append("OrderBlock")
            if obx["type"]=="bear": votes_s+=3; score_s+=1.8 if obx.get("near") else 1.0; logs.append("🏦 OB bear"); confirms.append("OrderBlock")
        if flow.get("signal")=="buy": votes_b+=2; score_b+=1.2+abs(flow["imbalance"]); logs.append(f"📊 FLOW buy imb={flow['imbalance']:.2f}"); confirms.append("Flow")
        if flow.get("signal")=="sell": votes_s+=2; score_s+=1.2+abs(flow["imbalance"]); logs.append(f"📊 FLOW sell imb={flow['imbalance']:.2f}"); confirms.append("Flow")
        if iceberg.get("has"):
            if iceberg["side"]=="bid": votes_b+=2; score_b+=1.2+iceberg["strength"]; logs.append(f"🧊 ICEBERG bid({iceberg['strength']:.2f})"); confirms.append("Iceberg")
            else: votes_s+=2; score_s+=1.2+iceberg["strength"]; logs.append(f"🧊 ICEBERG ask({iceberg['strength']:.2f})"); confirms.append("Iceberg")

        # --- Candle votes (Impulse / Harvest) ---
        if candle_ctx["strong"]:
            if candle_ctx["impulse"]:
                if candle_ctx["direction"]=="bull":
                    votes_b += 3; score_b += 2.0; logs.append(f"💥 Impulse Bull | body={candle_ctx['body_ratio']:.2f}"); confirms.append("Impulse")
                else:
                    votes_s += 3; score_s += 2.0; logs.append(f"💥 Impulse Bear | body={candle_ctx['body_ratio']:.2f}"); confirms.append("Impulse")
                    
            if candle_ctx["harvest"]:
                if candle_ctx["direction"]=="bull":
                    votes_b += 2; score_b += 1.5; logs.append(f"🪄 Wick Harvest (Buy) | wick={candle_ctx['wick_ratio']:.2f}"); confirms.append("WickHarvest")
                else:
                    votes_s += 2; score_s += 1.5; logs.append(f"🪄 Wick Harvest (Sell) | wick={candle_ctx['wick_ratio']:.2f}"); confirms.append("WickHarvest")

        # --- Liquidity Sweep bonus ---
        if sweep_ctx.get("has"):
            if sweep_ctx["type"]=="sell_sweep":
                votes_b += 2; score_b += 1.2; logs.append("🧲 Sweep Below ⇒ BUY bias"); confirms.append("Sweep")
            if sweep_ctx["type"]=="buy_sweep":
                votes_s += 2; score_s += 1.2; logs.append("🧲 Sweep Above ⇒ SELL bias"); confirms.append("Sweep")

        # --- SCALP Fusion (inside-council) ---
        px = float(df["close"].iloc[-1])
        scalp_meta = scalp_council_fusion(df, {
            "indicators": {
                "basic": ind_basic,
                "bollinger": bb,
                "fvg": fvg,
                "order_block": obx,
                "orderbook_flow": flow,
                "iceberg": iceberg,
                "candles": candle_ctx,
                "sweep": sweep_ctx
            },
            "mtf_analysis": mtf_meta
        }, px)

        # وزن السكالب ينعكس على التصويت العام
        if scalp_meta["ok"]:
            boost = min(3.0, 1.0 + scalp_meta["grade"] / 3.0)
            if scalp_meta["bias"] == "buy" or (candle_ctx["direction"] == "bull" and candle_ctx["strong"]):
                votes_b += 1
                score_b += boost
                logs.append(f"⚡ SCALP OK ↑ grade={scalp_meta['grade']:.2f} RR={scalp_meta['rr']:.2f}")
            elif scalp_meta["bias"] == "sell" or (candle_ctx["direction"] == "bear" and candle_ctx["strong"]):
                votes_s += 1
                score_s += boost
                logs.append(f"⚡ SCALP OK ↓ grade={scalp_meta['grade']:.2f} RR={scalp_meta['rr']:.2f}")
        else:
            logs.append(f"🛑 SCALP REJECT: {','.join(scalp_meta['reasons'])} | exp={scalp_meta['exp_move_pct']:.2f}% cost={scalp_meta['cost_pct']:.2f}% RR={scalp_meta['rr']:.2f}")

        return {
            "votes_buy": votes_b, "votes_sell": votes_s,
            "score_buy": round(score_b, 2), "score_sell": round(score_s, 2),
            "logs": logs, "confirmation_signals": confirms,
            "indicators": {
                "super_trend": st, "ichimoku": ichi, "bollinger": bb, "macd": macd, "stoch_rsi": stoch,
                "market_structure": structure, "money_flow": money, "basic": ind_basic, "rsi_context": rsi_ctx,
                "fvg": fvg, "order_block": obx, "orderbook_flow": flow, "iceberg": iceberg,
                "candles": candle_ctx, "sweep": sweep_ctx
            },
            "mtf_analysis": mtf_meta,
            "scalp": scalp_meta
        }
    except Exception as e:
        log_w(f"council error: {e}")
        return {"votes_buy":0,"votes_sell":0,"score_buy":0,"score_sell":0,
                "logs":[],"confirmation_signals":[], "indicators":{},"mtf_analysis":None, "scalp": {}}

# =================== LOG MEDIUM PANEL ===================
def render_medium_log(c):
    try:
        ind=c.get("indicators",{}); basic=ind.get("basic",{})
        st=ind.get("super_trend",{}); ich=ind.get("ichimoku",{}); bb=ind.get("bollinger",{})
        macd=ind.get("macd",{}); stoch=ind.get("stoch_rsi",{}); mstr=ind.get("market_structure",{})
        money=ind.get("money_flow",{}); mtf=c.get("mtf_analysis",{}) or {}
        fvg=ind.get("fvg",{}); obx=ind.get("order_block",{}); flow=ind.get("orderbook_flow",{}); ice=ind.get("iceberg",{})

        bal=balance_usdt(); bal_fmt=f"{bal:.2f}" if (bal is not None) else "N/A"
        print(f"💼 BALANCE: {bal_fmt} USDT | 📦 COMPOUND PNL: {compound_pnl:+.4f} USDT", flush=True)

        if STATE.get("open"):
            entry=STATE.get("entry"); entry_fmt=f"{entry:.6f}" if isinstance(entry,(int,float)) else str(entry)
            print(f"🧭 MODE={STATE.get('mode','-').upper()} | POS={STATE.get('side','-').upper()} | PnL={STATE.get('pnl',0.0):.2f}% | TP_hits={int(STATE.get('profit_targets_achieved',0))} | entry={entry_fmt} | adds={STATE.get('adds',0)}", flush=True)
        else:
            print(f"⚪ NO OPEN POSITIONS | Waiting: {wait_for_next_signal_side}", flush=True)

        votes=f"{c.get('votes_buy',0)}/{c.get('votes_sell',0)}"
        scores=f"{c.get('score_buy',0):.1f}/{c.get('score_sell',0):.1f}"
        print("—"*70, flush=True)
        print(f"🧠 COUNCIL: votes={votes} | scores={scores} | confirms={len(c.get('confirmation_signals',[]))}", flush=True)
        print(f"📊 ADX/DI: ADX={basic.get('adx',0):.1f} | +DI={basic.get('plus_di',0):.1f}/-DI={basic.get('minus_di',0):.1f} | RSI={basic.get('rsi',50):.1f}", flush=True)
        print(f"🟢 ST={st.get('signal','-')}({st.get('strength',0):.2f}) | ☁️ Ichi={ich.get('signal','-')}({ich.get('strength',0):.2f}) | 🎯 MTF={mtf.get('overall_signal','-')}({mtf.get('bull_count',0)}/{mtf.get('bear_count',0)})", flush=True)
        print(f"📈 MACD={macd.get('signal','-')}(hist={macd.get('histogram',0):.4f}) | 🔁 Stoch={stoch.get('signal','-')}(K={stoch.get('k',0):.1f}/D={stoch.get('d',0):.1f})", flush=True)
        print(f"📎 BB={bb.get('signal','-')}(%B={bb.get('percent_b',0):.2f}, sq={bool(bb.get('squeeze',False))}) | 🧱 Struct={mstr.get('signal','-')}({mstr.get('strength',0):.2f}%)", flush=True)
        flow_txt=f"{flow.get('signal','-')}(imb={float(flow.get('imbalance',0)):.2f},Wb={float(flow.get('bid_wall',0)):.1f}/Wa={float(flow.get('ask_wall',0)):.1f})"
        fvg_txt = f"{fvg.get('type','-')}{'~' if fvg.get('near') else ''}" if fvg.get("has") else "-"
        obx_txt = f"{obx.get('type','-')}{'~' if obx.get('near') else ''}" if obx.get("has") else "-"
        print(f"🧠 SMC: FVG={fvg_txt} | OB={obx_txt} | FLOW={flow_txt}", flush=True)
        if ice and ice.get("has"):
            print(f"🧊 Iceberg: side={ice.get('side')} @ {float(ice.get('price',0)):.4f} | str={float(ice.get('strength',0)):.2f}", flush=True)

        # إضافة تحليل السكالب
        scalp = c.get("scalp", {})
        if scalp:
            status = "✅" if scalp.get("ok") else "❌"
            print(f"⚡ SCALP: {status} grade={scalp.get('grade',0):.2f} | bias={scalp.get('bias','-')} | "
                  f"exp={scalp.get('exp_move_pct',0):.2f}% | cost={scalp.get('cost_pct',0):.2f}% | RR={scalp.get('rr',0):.2f}", flush=True)
            
            if scalp.get("reasons"):
                reasons_text = ', '.join(scalp['reasons'])
                print(f"   📋 Reasons: {reasons_text}", flush=True)
                
            candle = scalp.get("candles", {})
            if candle and candle.get("strong"):
                impulse_icon = "💥" if candle.get("impulse") else ""
                harvest_icon = "🪄" if candle.get("harvest") else ""
                print(f"   🕯️ Candle: {candle.get('direction')} {impulse_icon}{harvest_icon} | "
                      f"body={candle.get('body_ratio',0):.2f} | wick={candle.get('wick_ratio',0):.2f}", flush=True)
        
        print("—" * 70, flush=True)
    except Exception as e:
        log_w(f"render_medium_log error: {e}")

# =================== ENTRY/EXIT & MGMT ===================
def setup_trade_management(mode):
    if mode=="scalp":
        return {"tp_targets":SCALP_TP_TARGETS,"tp_fractions":SCALP_TP_FRACTIONS,
                "be_activate_pct":BREAKEVEN_AFTER/100.0,"trail_activate_pct":TRAIL_ACTIVATE_PCT/100.0,
                "atr_trail_mult":ATR_TRAIL_MULT,"max_tp_levels":len(SCALP_TP_TARGETS)}
    return {"tp_targets":TREND_TP_TARGETS,"tp_fractions":TREND_TP_FRACTIONS,
            "be_activate_pct":BREAKEVEN_AFTER/100.0,"trail_activate_pct":TRAIL_ACTIVATE_PCT/100.0,
            "atr_trail_mult":ATR_TRAIL_MULT,"max_tp_levels":len(TREND_TP_TARGETS)}

def manage_take_profits(state,current_price,pnl_pct,mgmt,mode):
    if state["qty"]<=0: return
    tps=mgmt["tp_targets"]; frs=mgmt["tp_fractions"]
    hit=state.get("tp_levels_hit",[False]*len(tps))
    for i,(tp,frac) in enumerate(zip(tps,frs)):
        if (mode=="scalp" and state.get(f"tp{i+1}_done")) or (mode!="scalp" and hit[i]): 
            continue
        if pnl_pct >= tp:
            close_qty=safe_qty(state["qty"]*frac)
            if close_qty>0 and MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    side="sell" if state["side"]=="long" else "buy"
                    params=exchange_specific_params(side, True)
                    ex.create_order(SYMBOL,"market",side,close_qty,None,params)
                except Exception as e: log_e(f"TP{i+1} failed: {e}")
            state["qty"]=safe_qty(state["qty"]-close_qty)
            if mode=="scalp": state[f"tp{i+1}_done"]=True
            else: hit[i]=True
            state["profit_targets_achieved"]+=1
            log_g(f"TP{i+1} hit ({tp}%)")
    state["tp_levels_hit"]=hit

def smart_entry_system(council):
    confirms=len(council["confirmation_signals"])
    if confirms>=REQUIRED_CONFIRMATIONS:
        if council["score_buy"]>council["score_sell"]+2: return "buy"
        if council["score_sell"]>council["score_buy"]+2: return "sell"
    return None

def execute_trade_decision(side, price, qty, council):
    if not isinstance(qty,(int,float)) or qty<=0: log_e("qty invalid"); return False
    if price is None or price<=0: log_e("price invalid"); return False
    if side not in ["buy","sell"]: log_e(f"side invalid {side}"); return False
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side.upper()} {qty:.4f} @ {price:.6f}"); return True
    print(f"🎯 EXECUTE: {side.upper()} {qty:.4f} @ {price:.6f} | score {council['score_buy']:.1f}/{council['score_sell']:.1f} | conf {len(council['confirmation_signals'])}", flush=True)
    try:
        if MODE_LIVE:
            if SYMBOL not in ex.markets: ex.load_markets()
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params=exchange_specific_params(side, False)
            try:
                ex.create_order(SYMBOL,"market",side,qty,None,params)
                log_g("EXECUTED")
                return True
            except ccxt.InsufficientFunds:
                log_w("margin low → reduce 20% and retry")
                rq=safe_qty(qty*0.8)
                if rq>0:
                    ex.create_order(SYMBOL,"market",side,rq,None,params); log_g("EXECUTED reduced")
                    return True
                return False
        else:
            log_g("SIMULATED EXECUTION"); return True
    except Exception as e:
        log_e(f"EXECUTION FAILED: {e}"); return False

def open_market_ultra(side, qty, price, council):
    if qty<=0: log_e("skip open (qty<=0)"); return False
    strength=max(council["score_buy"], council["score_sell"])
    mode="trend" if strength>12 else "scalp"
    mgmt=setup_trade_management(mode)
    if execute_trade_decision(side, price, qty, council):
        tp_hit=[False]*len(mgmt["tp_targets"])
        STATE.update({"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,"pnl":0.0,
                      "bars":0,"trail":None,"breakeven":None,"tp_levels_hit":tp_hit,"highest_profit_pct":0.0,
                      "profit_targets_achieved":0,"mode":mode,"management":mgmt,"adds":0})
        save_state({"in_position":True,"side":STATE["side"].upper(),"entry_price":price,
                    "position_qty":qty,"leverage":LEVERAGE,"mode":mode,"management":mgmt,
                    "opened_at":int(time.time()),"tp_levels_hit":tp_hit})
        log_g(f"POSITION OPENED: {side.upper()} | mode={mode} | strength={strength:.1f} | qty={qty}")
        return True
    return False

def try_scale_in(df, council):
    if not STATE.get("open") or STATE.get("mode")!="trend" or STATE.get("adds",0)>=SCALE_IN_MAX: 
        return
    px = price_now()
    if not px: 
        return
    entry = STATE.get("entry")
    pullback_ok = abs((px - entry)/entry)*100 <= PULLBACK_FOR_ADD
    flow_sig = council.get("indicators",{}).get("orderbook_flow",{}).get("signal","neutral")
    has_smc = council.get("indicators",{}).get("fvg",{}).get("has") or council.get("indicators",{}).get("order_block",{}).get("has")
    if STATE["side"]=="long" and pullback_ok and flow_sig=="buy" and has_smc:
        add = safe_qty(STATE["qty"] * SCALE_IN_ADD)
        if add>0 and execute_trade_decision("buy", px, add, council):
            STATE["qty"] += add; STATE["adds"] += 1; log_g("Scale-In +30% (LONG)")
    if STATE["side"]=="short" and pullback_ok and flow_sig=="sell" and has_smc:
        add = safe_qty(STATE["qty"] * SCALE_IN_ADD)
        if add>0 and execute_trade_decision("sell", px, add, council):
            STATE["qty"] += add; STATE["adds"] += 1; log_g("Scale-In +30% (SHORT)")

def smart_exit_analysis(df, council, price, pnl_pct, side, entry):
    cur = ultra_intelligent_council(df)
    cur_sig = "buy" if cur["score_buy"]>cur["score_sell"] else "sell"
    if (side=="long" and cur_sig=="sell") or (side=="short" and cur_sig=="buy"):
        return {"action":"close","reason":"council_reversal"}
    if pnl_pct >= (8 if STATE.get("mode")=="trend" else 4):
        return {"action":"close","reason":"excellent_profit"}
    if council["indicators"]["basic"]["adx"]<15 and abs(pnl_pct)>1:
        return {"action":"close","reason":"momentum_loss"}
    return {"action":"hold","reason":"continue"}

def manage_after_entry_enhanced(df, council):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = price_now()
    if not px:
        return
    entry=STATE["entry"]; side=STATE["side"]; mode=STATE.get("mode","trend"); mgmt=STATE.get("management",{})
    pnl_pct = ((px-entry)/entry*(100) if side=="long" else (entry-px)/entry*(100))
    STATE["pnl"]=pnl_pct
    if pnl_pct>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=pnl_pct
    try_scale_in(df, council)
    manage_take_profits(STATE, px, pnl_pct/100.0, mgmt, mode)
    decision=smart_exit_analysis(df, council, px, pnl_pct, side, entry)
    if decision["action"]=="close":
        log_w(f"EXIT: {decision['reason']}"); close_market_strict(decision["reason"])

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    def _read_position():
        try:
            poss=ex.fetch_positions(params={"type":"swap"})
            for p in poss:
                sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
                if SYMBOL.split(":")[0] not in sym: continue
                qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
                if qty<=0: return 0.0, None, None
                entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
                side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
                side = "long" if "long" in side_raw or ("both" in side_raw and STATE.get("side")=="long") else "short"
                return qty, side, entry
        except Exception: return 0.0, None, None
        return 0.0, None, None

    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0:
        if STATE.get("open"): _reset_after_close(reason)
        return

    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last_err=None
    while attempts<5:
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                params=exchange_specific_params(side_to_close, True)
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(1.5)
            q2,_,_= _read_position()
            if q2<=0.0000001:
                log_g(f"STRICT CLOSE done ({reason})")
                _reset_after_close(reason); return
            else:
                qty_to_close=safe_qty(q2)
        except Exception as e:
            last_err=e; log_w(f"strict close retry {attempts+1}/5: {e}")
        attempts+=1
    log_e(f"STRICT CLOSE failed ({reason}) — last_error={last_err}")

def _reset_after_close(reason):
    global compound_pnl, wait_for_next_signal_side
    wait_for_next_signal_side= "SELL" if STATE.get("side")=="long" else "BUY"
    STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"adds":0})
    save_state({"in_position":False,"position_qty":0,"compound_pnl":round(float(compound_pnl),6)})
    log_i(f"after-close: wait_for_next_signal_side={wait_for_next_signal_side}")

# Early ignition (probe)
def early_ignition_signal(council):
    ind    = council.get("indicators", {})
    basic  = ind.get("basic", {})
    fvg    = ind.get("fvg", {})
    obx    = ind.get("order_block", {})
    flow   = ind.get("orderbook_flow", {})
    struct = ind.get("market_structure", {})
    mtf    = council.get("mtf_analysis", {}) or {}
    adx = float(basic.get("adx", 0.0))
    flow_sig = flow.get("signal","neutral")
    mtf_ok = (mtf.get("overall_signal") in ("bullish","bearish")) and (max(mtf.get("bull_count",0), mtf.get("bear_count",0)) >= 2)
    bull_core = (fvg.get("has") and fvg.get("type")=="bull") and (obx.get("has") and obx.get("type")=="bull") and (flow_sig=="buy")
    bear_core = (fvg.get("has") and fvg.get("type")=="bear") and (obx.get("has") and obx.get("type")=="bear") and (flow_sig=="sell")
    strong_buy  = bull_core and (struct.get("signal") in ("strong_buy","neutral")) and mtf_ok and (adx >= 18)
    strong_sell = bear_core and (struct.get("signal") in ("strong_sell","neutral")) and mtf_ok and (adx >= 18)
    if strong_buy and (council["score_buy"] >= council["score_sell"] + 1.0): return "buy_probe"
    if strong_sell and (council["score_sell"] >= council["score_buy"] + 1.0): return "sell_probe"
    return None

# =================== ENHANCED POSITION LOGGING ===================
def log_position_opened(side, price, qty, mode, council):
    """لوغ مفصل عند فتح الصفقة الجديدة"""
    try:
        print("═" * 80, flush=True)
        
        # العنوان الرئيسي
        position_type = "🟩 LONG" if side == "buy" else "🟥 SHORT"
        print(f"{position_type} | MODE: {mode.upper()} | LEVERAGE: {LEVERAGE}x", flush=True)
        
        # معلومات الأساسية
        bal = balance_usdt()
        print(f"💰 ENTRY: {price:.6f} | QTY: {qty:.6f} | SYMBOL: {SYMBOL}", flush=True)
        print(f"💼 BALANCE: {bal:.2f} USDT | 📦 COMPOUND PNL: {compound_pnl:+.4f} USDT", flush=True)
        
        # تحليل المجلس
        if council:
            sc_buy = council.get('score_buy', 0.0)
            sc_sell = council.get('score_sell', 0.0)
            votes_b = council.get('votes_buy', 0)
            votes_s = council.get('votes_sell', 0)
            confirms = council.get('confirmation_signals', [])
            
            print(f"🧠 COUNCIL: BUY={sc_buy:.1f}({votes_b}v) | SELL={sc_sell:.1f}({votes_s}v) | CONFIRMS={len(confirms)}", flush=True)
            
            # عرض أهم إشارات التأكيد
            if confirms:
                top_confirms = confirms[:4]  # أول 4 إشارات فقط
                print(f"   ✅ Top signals: {', '.join(top_confirms)}", flush=True)
        
        # تحليل السكالب المتقدم
        scalp_meta = council.get('scalp', {}) if council else {}
        if scalp_meta:
            grade = scalp_meta.get('grade', 0)
            rr = scalp_meta.get('rr', 0)
            bias = scalp_meta.get('bias', 'neutral')
            exp_move = scalp_meta.get('exp_move_pct', 0)
            cost = scalp_meta.get('cost_pct', 0)
            
            print(f"⚡ SCALP ANALYSIS: Grade={grade:.2f} | RR={rr:.2f} | Bias={bias}", flush=True)
            print(f"   📊 Expected: {exp_move:.2f}% | Cost: {cost:.2f}% | Net: {exp_move-cost:.2f}%", flush=True)
            
            if scalp_meta.get('reasons'):
                reasons = scalp_meta['reasons'][:3]  # أول 3 أسباب فقط
                print(f"   🔍 Reasons: {', '.join(reasons)}", flush=True)
        
        # إعدادات إدارة الصفقة
        mgmt_config = setup_trade_management(mode)
        if mgmt_config:
            tp_targets = mgmt_config.get('tp_targets', [])
            tp_fractions = mgmt_config.get('tp_fractions', [])
            
            print(f"🎯 PROFIT TARGETS:", flush=True)
            for i, (target, fraction) in enumerate(zip(tp_targets, tp_fractions)):
                print(f"   TP{i+1}: {target}% ({fraction*100}% of position)", flush=True)
            
            # معلومات الحماية
            be_level = mgmt_config.get('be_activate_pct', 0) * 100
            trail_activate = mgmt_config.get('trail_activate_pct', 0) * 100
            atr_trail = mgmt_config.get('atr_trail_mult', 0)
            
            print(f"🛡️ PROTECTIONS: BE@ {be_level:.1f}% | Trail@ {trail_activate:.1f}% | ATR×{atr_trail}", flush=True)
        
        # معلومات السيولة والسبريد
        current_spread = spread_bps()
        spread_status = "✅ GOOD" if current_spread <= MAX_SPREAD_BPS else "⚠️ HIGH"
        print(f"📊 LIQUIDITY: Spread {current_spread:.1f}bps {spread_status} | Max: {MAX_SPREAD_BPS}bps", flush=True)
        
        # الطابع الزمني
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"⏰ OPENED: {ts} | Thread: {threading.get_ident()}", flush=True)
        
        print("═" * 80, flush=True)
        
    except Exception as e:
        log_w(f"Enhanced position log failed: {e}")

# =================== LOOP ===================
def ultra_trading_loop():
    global wait_for_next_signal_side
    while True:
        try:
            # spread gate
            spr=spread_bps()
            if spr>MAX_SPREAD_BPS:
                log_w(f"spread gate: {spr:.2f}bps > {MAX_SPREAD_BPS}bps")
                time.sleep(3); continue

            update_orderflow_buffers()

            df=fetch_ohlcv(limit=600)
            if len(df)<120: time.sleep(3); continue
            mtf=fetch_multi_timeframe(("5m","15m","1h"))
            mtf_meta={"overall_signal":mtf["overall_signal"],"bull_count":mtf["bull_count"],"bear_count":mtf["bear_count"],"confidence":mtf["confidence"],"signals":mtf["signals"]}
            council=ultra_intelligent_council(df, mtf_meta=mtf_meta)

            if LOG_MEDIUM_PANEL: render_medium_log(council)

            if not STATE["open"]:
                # Early ignition probe
                probe = early_ignition_signal(council)
                if probe:
                    px  = price_now()
                    if not px:
                        time.sleep(2)
                        continue
                    bal = balance_usdt()
                    qty = compute_size(bal, px) * PROBE_SIZE
                    open_market_ultra("buy" if probe=="buy_probe" else "sell", safe_qty(qty), px, council)
                    time.sleep(2)
                    continue

                # wait-for-next-signal guard
                if wait_for_next_signal_side:
                    sig = "BUY" if council["score_buy"]>council["score_sell"] else "SELL"
                    if sig != wait_for_next_signal_side:
                        time.sleep(2); continue
                    else:
                        wait_for_next_signal_side=None

                side = smart_entry_system(council)
                if side:
                    px=price_now()
                    if not px:
                        time.sleep(2)
                        continue
                    bal=balance_usdt(); qty=compute_size(bal, px)
                    open_market_ultra(side, qty, px, council)
            else:
                manage_after_entry_enhanced(df, council)

        except Exception as e:
            log_w(f"loop error: {e}")
            traceback.print_exc()
        time.sleep(2)

# =================== API / KEEPALIVE ===================
app=Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"ok":True,"ts":int(time.time()),"bot":BOT_VERSION,"open":STATE.get("open",False)})

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange":EXCHANGE_NAME,"symbol":SYMBOL,"interval":INTERVAL,
        "mode":"live" if MODE_LIVE else "paper","leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,
        "price":price_now(),"balance":balance_usdt(),
        "state":STATE,"compound_pnl":compound_pnl,"version":BOT_VERSION
    })

def keepalive_loop():
    if not SELF_URL:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    import requests
    sess=requests.Session()
    while True:
        try:
            r=sess.get((SELF_URL.rstrip("/")+"/health"), timeout=10)
            log_i(f"keepalive {r.status_code}")
        except Exception as e:
            log_w(f"keepalive fail: {e}")
        time.sleep(60)

if __name__=="__main__":
    import threading
    threading.Thread(target=ultra_trading_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
