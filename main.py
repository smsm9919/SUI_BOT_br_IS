# -*- coding: utf-8 -*-
"""
SUI Perp — Council PRO HUNTER (Golden Zones + Fancy Logs)
- Council: RF + ADX/DI + RSI + MACA + SMC(OB/FVG/BOS/ICT) + Footprint/Flow + Volume/ATR
- Golden Zones (Bottom/Top): Fib 0.618–0.786 + Sweep + RSI_MA Cross + Vol>MA20 + ADX Gate
- Sizing: 60% balance * leverage with exchange minQty/step
- Scalp Guard + Spread Guard + Dynamic TP (1/2/3) + ATR Trail
- Ops: / (OK), /health, /metrics, /logs + keepalive
"""

import os, time, math, logging, threading, traceback
from datetime import datetime
from collections import deque
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import ccxt
from flask import Flask, jsonify

# ===========================
# SETTINGS
# ===========================
SYMBOL             = os.getenv("SYMBOL", "SUI/USDT:USDT")
EXCHANGE_NAME      = os.getenv("EXCHANGE", "bybit").lower()           # bybit | bingx
API_KEY            = os.getenv("API_KEY", os.getenv("BYBIT_API_KEY", os.getenv("BINGX_API_KEY", "")))
API_SECRET         = os.getenv("API_SECRET", os.getenv("BYBIT_API_SECRET", os.getenv("BINGX_API_SECRET", "")))
POSITION_MODE      = os.getenv("POSITION_MODE", "oneway")
LEVERAGE           = int(os.getenv("LEVERAGE", "10"))
RISK_ALLOC         = float(os.getenv("RISK_ALLOC", "0.60"))            # 60%
TIMEFRAME          = os.getenv("TIMEFRAME", "15m")
LOOKBACK_BARS      = int(os.getenv("LOOKBACK_BARS", "400"))
LOOP_SLEEP_SEC     = float(os.getenv("LOOP_SLEEP_SEC", "6.0"))
DRY_RUN            = bool(int(os.getenv("DRY_RUN", "0")))

# RF + انتظار بعد الإغلاق
RF_PERIOD          = int(os.getenv("RF_PERIOD", "20"))
RF_MULT            = float(os.getenv("RF_MULT", "3.5"))
ENTRY_RF_ONLY      = bool(int(os.getenv("ENTRY_RF_ONLY", "0")))
WAIT_NEXT_SIGNAL   = bool(int(os.getenv("WAIT_NEXT_SIGNAL", "1")))

# Guards
MAX_SPREAD_BPS     = float(os.getenv("MAX_SPREAD_BPS", "8.0"))
FINAL_CHUNK_QTY    = float(os.getenv("FINAL_CHUNK_QTY", "2.0"))
MAX_TRADES_PER_HR  = int(os.getenv("MAX_TRADES_PER_HR", "8"))
COOLDOWN_AFTER_CLOSE_SEC = int(os.getenv("COOLDOWN_AFTER_CLOSE_SEC", "30"))

# Council thresholds
ULTIMATE_MIN_CONFIDENCE = float(os.getenv("ULTIMATE_MIN_CONFIDENCE", "7.0"))
SELL_SUPERIORITY        = float(os.getenv("SELL_SUPERIORITY", "1.0"))
BUY_SUPERIORITY         = float(os.getenv("BUY_SUPERIORITY",  "1.0"))

# Pro-Hunter
PROACTIVE_HUNTER          = bool(int(os.getenv("PROACTIVE_HUNTER", "1")))
HUNTER_STRONG_SCORE       = float(os.getenv("HUNTER_STRONG_SCORE", "6.5"))
HUNTER_TREND_GATE_ADX     = float(os.getenv("HUNTER_TREND_GATE_ADX", "20"))
HUNTER_MIN_DISPLACEMENT   = float(os.getenv("HUNTER_MIN_DISPLACEMENT", "0.004"))
HUNTER_VOL_BURST_MULT     = float(os.getenv("HUNTER_VOL_BURST_MULT", "1.6"))
HUNTER_FLOW_MIN_BIAS      = float(os.getenv("HUNTER_FLOW_MIN_BIAS", "0.8"))

# Golden Zones
GZ_ENABLED     = bool(int(os.getenv("GZ_ENABLED", "1")))
GZ_ADX_GATE    = float(os.getenv("GZ_ADX_GATE", "20"))
GZ_MIN_SCORE   = float(os.getenv("GZ_MIN_SCORE", "6.0"))
GZ_FIB_LOW     = float(os.getenv("GZ_FIB_LOW", "0.618"))
GZ_FIB_HIGH    = float(os.getenv("GZ_FIB_HIGH", "0.786"))
GZ_VOL_MULT    = float(os.getenv("GZ_VOL_MULT", "1.3"))
GZ_RSI_MA      = int(os.getenv("GZ_RSI_MA", "9"))

# SMC + MACA
ENABLE_SMC_OB   = bool(int(os.getenv("ENABLE_SMC_OB",   "1")))
ENABLE_SMC_FVG  = bool(int(os.getenv("ENABLE_SMC_FVG",  "1")))
ENABLE_SMC_BOS  = bool(int(os.getenv("ENABLE_SMC_BOS",  "1")))
ENABLE_SMC_ICT  = bool(int(os.getenv("ENABLE_SMC_ICT",  "1")))
SMC_LOOKBACK    = int(os.getenv("SMC_LOOKBACK", "60"))
SMC_MIN_DISP    = float(os.getenv("SMC_MIN_DISP", "0.004"))
FVG_MIN_GAP_PCT = float(os.getenv("FVG_MIN_GAP_PCT", "0.15"))

MACA_FAST      = int(os.getenv("MACA_FAST", "9"))
MACA_SLOW      = int(os.getenv("MACA_SLOW", "21"))
MACA_ANGLE_LEN = int(os.getenv("MACA_ANGLE_LEN", "5"))
MACA_MIN_ANGLE = float(os.getenv("MACA_MIN_ANGLE", "8.0"))

# Dynamic TP
TP1_PCT_BASE = float(os.getenv("TP1_PCT_BASE", "0.40"))
TP2_PCT_BASE = float(os.getenv("TP2_PCT_BASE", "0.90"))
TP3_PCT_BASE = float(os.getenv("TP3_PCT_BASE", "1.60"))
TP_LADDER_QTY = os.getenv("TP_LADDER_QTY", "40,35,25")

# Scalp Guard (تكلفة تقريبية)
FEES_TAKER_BPS          = float(os.getenv("FEES_TAKER_BPS", "7.0"))
SLIPPAGE_BPS_BASE       = float(os.getenv("SLIPPAGE_BPS_BASE", "3.0"))
SLIPPAGE_BPS_ATR_MULT   = float(os.getenv("SLIPPAGE_BPS_ATR_MULT", "0.25"))
SCALP_MIN_EDGE_BPS      = float(os.getenv("SCALP_MIN_EDGE_BPS", "12.0"))
SCALP_MIN_STRENGTH      = float(os.getenv("SCALP_MIN_STRENGTH", "6.5"))
SCALP_MAX_TP_PCT        = float(os.getenv("SCALP_MAX_TP_PCT", "1.0"))
SCALP_ALLOW_ONLY_IF_POS = bool(int(os.getenv("SCALP_ALLOW_ONLY_IF_POS", "1")))

# Ops
SELF_URL           = os.getenv("SELF_URL", os.getenv("RENDER_EXTERNAL_URL", ""))
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "50"))
PORT               = int(os.getenv("PORT", "5000"))

# ===========================
# LOGGING + FANCY BOXES
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("PRO-HUNTER")

ANSI = {
    "RESET":"\033[0m","BOLD":"\033[1m",
    "RED":"\033[31m","GREEN":"\033[32m","YELLOW":"\033[33m",
    "BLUE":"\033[34m","MAG":"\033[35m","CYAN":"\033[36m","GRAY":"\033[90m"
}
def _c(txt, color):
    if os.getenv("NO_COLOR") == "1": return txt
    return f"{ANSI.get(color,'')}{txt}{ANSI['RESET']}"

LOG_RING = deque(maxlen=300)
def push_event(kind:str, payload:dict):
    evt = {"ts": datetime.utcnow().isoformat()+"Z", "kind": kind, **payload}
    LOG_RING.append(evt); return evt

def box(title:str, lines:list, color="CYAN"):
    top = _c("┌"+"─"*58+"┐", color)
    mid = [_c(f"│ {title:<56} │", color)]
    mid += [_c(f"│ {ln:<56} │", color) for ln in lines]
    bot = _c("└"+"─"*58+"┘", color)
    return "\n".join([top,*mid,bot])

def log_trade_open(side:str, qty:float, price:float, reasons:str, min_conf:float):
    color = "GREEN" if side=="BUY" else "RED"
    lines = [
        f"Side     : {side}",
        f"Qty      : {qty}",
        f"Price    : {round(price,6)}",
        f"MinConf  : {round(min_conf,2)}",
        f"Reasons  : {reasons[:52]}",
    ]
    log.info(box(("🟢 EXECUTE" if side=="BUY" else "🔴 EXECUTE"), lines, color=color))
    push_event("open", {"side":side,"qty":qty,"price":price,"min_conf":min_conf,"reasons":reasons})

def log_tp_event(side_long:bool, pct:float, qty:float, remaining:float):
    lines = [f"TP Hit   : {pct}%", f"Closed   : {round(qty,6)}",
             f"Remain   : {round(remaining,6)}", f"Dir      : {'LONG' if side_long else 'SHORT'}"]
    log.info(box("🏁 TAKE-PROFIT", lines, color="YELLOW"))
    push_event("tp", {"pct":pct,"qty":qty,"remain":remaining,"dir":"LONG" if side_long else "SHORT"})

def log_guard(name:str, details:str):
    log.info(box(f"🛡 GUARD — {name}", [details], color="MAG"))
    push_event("guard", {"name":name,"details":details})

def log_golden(bottom:bool, top:bool, info:dict):
    if not (bottom or top): return
    which = "Golden BOTTOM" if bottom else "Golden TOP"
    lines = [
        f"Trend   : {info.get('trend')}",
        f"Fib     : {round(info.get('fib_low',0),6)} ~ {round(info.get('fib_high',0),6)}",
        f"Vol>MA  : {info.get('vol_ok')}",
        f"RSI▲    : {info.get('rsi_cross_up',False)} | RSI▼: {info.get('rsi_cross_dn',False)}",
    ]
    log.info(box(f"🌟 {which}", lines, color=("GREEN" if bottom else "RED")))
    push_event("golden", {"which":which, **info})

# ===========================
# EXCHANGE
# ===========================
def build_exchange():
    if EXCHANGE_NAME == "bybit":
        ex = ccxt.bybit({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
        ex.options["defaultType"] = "swap"
    elif EXCHANGE_NAME == "bingx":
        ex = ccxt.bingx({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
        ex.options["defaultType"] = "swap"
    else:
        ex = getattr(ccxt, EXCHANGE_NAME)({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
        ex.options["defaultType"] = "swap"
    return ex

exchange = build_exchange()

# ===========================
# INDICATORS & SMC & GOLDEN
# ===========================
def fetch_ohlcv(symbol, timeframe, limit=LOOKBACK_BARS):
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

def to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df

def compute_rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1/length, min_periods=length).mean()
    loss = down.ewm(alpha=1/length, min_periods=length).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100/(1+rs))

def compute_adx(df, length=14):
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm  = (h - h.shift(1)).clip(lower=0.0)
    minus_dm = (l.shift(1) - l).clip(lower=0.0)
    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(length).mean()
    plus_di  = 100 * (plus_dm.ewm(alpha=1/length).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1/length).mean() / (atr + 1e-9))
    dx = (100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)).fillna(0)
    adx = dx.ewm(alpha=1/length).mean()
    return adx, plus_di, minus_di, atr

def compute_range_filter(df, period=20, mult=3.5):
    src = df["close"]
    ma = src.rolling(period).mean()
    dev = (src - ma).abs().rolling(period).mean()
    upper = ma + mult * dev
    lower = ma - mult * dev
    sig = np.where(src > upper, 1, np.where(src < lower, -1, 0))
    return sig, upper, lower

def compute_maca(series_close, fast=9, slow=21, angle_len=5):
    ma_fast = pd.Series(series_close).rolling(fast).mean()
    ma_slow = pd.Series(series_close).rolling(slow).mean()
    cross_up = ma_fast.iloc[-2] < ma_slow.iloc[-2] and ma_fast.iloc[-1] > ma_slow.iloc[-1]
    cross_dn = ma_fast.iloc[-2] > ma_slow.iloc[-2] and ma_fast.iloc[-1] < ma_slow.iloc[-1]
    y2, y1 = ma_fast.iloc[-1], ma_fast.iloc[-1-angle_len]
    angle_deg = float(math.degrees(math.atan2((y2 - y1), angle_len)))
    return {"ma_fast": float(ma_fast.iloc[-1]), "ma_slow": float(ma_slow.iloc[-1]),
            "cross_up": bool(cross_up), "cross_dn": bool(cross_dn), "angle_deg": angle_deg}

def _swing_points(h, l, lb=5):
    highs, lows = [], []
    for i in range(lb, len(h)-lb):
        if h[i] == max(h[i-lb:i+lb+1]): highs.append(i)
        if l[i] == min(l[i-lb:i+lb+1]): lows.append(i)
    return highs, lows

def detect_bos(df, lb=5):
    h, l = df['high'].values, df['low'].values
    highs, lows = _swing_points(h, l, lb)
    bos_up = len(highs) > 1 and h[-1] > h[highs[-2]]
    bos_dn = len(lows)  > 1 and l[-1] < l[lows[-2]]
    return bos_up, bos_dn

def detect_order_blocks(df, lookback=60, min_disp=0.004):
    c, o, h, l = df['close'].values, df['open'].values, df['high'].values, df['low'].values
    rng = (h - l); body = (c - o)
    atr = max(1e-9, pd.Series(rng).tail(14).mean())
    bull_ob = bear_ob = False
    start = max(1, len(df)-lookback)
    for i in range(start, len(df)-2):
        is_up = body[i] > 0 and (body[i]/max(1e-9, rng[i])) > 0.6 and (rng[i]/atr) > 1.2
        is_dn = body[i] < 0 and (abs(body[i])/max(1e-9, rng[i])) > 0.6 and (rng[i]/atr) > 1.2
        if is_up and (c[i+1] >= c[i]) and (c[-1] >= c[i]*(1+min_disp)): bull_ob = True
        if is_dn and (c[i+1] <= c[i]) and (c[-1] <= c[i]*(1-min_disp)): bear_ob = True
    return bull_ob, bear_ob

def detect_fvg(df, min_gap_pct=0.15):
    if len(df) < 3: return False, False
    atr = (df['high']-df['low']).rolling(14).mean().iloc[-1]
    thr = atr * (min_gap_pct/100.0)
    h, l = df['high'].values, df['low'].values
    return bool(l[-1] > h[-3] + thr), bool(h[-1] < l[-3] - thr)

def detect_ict(df, min_disp=0.004):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    bull_sweep = l[-1] < min(l[-5:-1]) and c[-1] > c[-2]
    bear_sweep = h[-1] > max(h[-5:-1]) and c[-1] < c[-2]
    bull_disp = (c[-1] - c[-2]) / max(1e-9, c[-2]) >= min_disp
    bear_disp = (c[-2] - c[-1]) / max(1e-9, c[-2]) >= min_disp
    bull_retest = bull_sweep and bull_disp and (l[-1] <= c[-2])
    bear_retest = bear_sweep and bear_disp and (h[-1] >= c[-2])
    return bool(bull_retest), bool(bear_retest)

def rsi_ma(series_close, rsi_len=14, ma_len=9):
    r = compute_rsi(series_close, rsi_len)
    return r, r.rolling(ma_len).mean()

def last_impulse_leg(df, lookback=80):
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    a, b = max(2, len(df)-lookback), len(df)-1
    i_min = int(np.argmin(l[a:b])) + a
    i_max = int(np.argmax(h[a:b])) + a
    if i_min < i_max:
        return ("up", i_min, i_max, float(l[i_min]), float(h[i_max]))
    else:
        return ("down", i_max, i_min, float(l[i_min]), float(h[i_max]))

def detect_golden_zones(df, adx_val, vol_mult=1.3, fib_low=0.618, fib_high=0.786, rsi_ma_len=9):
    if len(df) < 50: return False, False, {}
    trend, _, _, low_p, high_p = last_impulse_leg(df, lookback=80)
    c = df["close"].values
    vol_ma20 = df["volume"].rolling(20).mean().iloc[-1]
    rsi, rma = rsi_ma(df["close"], 14, rsi_ma_len)

    if trend == "up":
        length = high_p - low_p
        fib_a = low_p + (length * fib_low)
        fib_b = low_p + (length * fib_high)
    else:
        length = low_p - high_p
        fib_a = high_p + (length * (1 - fib_high))
        fib_b = high_p + (length * (1 - fib_low))

    price = c[-1]
    vol_ok = (df["volume"].iloc[-1] > vol_ma20 * vol_mult) if vol_ma20 else False

    bull_sweep = (df["low"].iloc[-1] <= df["low"].rolling(6).min().iloc[-2]) and (df["close"].iloc[-1] > df["open"].iloc[-1])
    bear_sweep = (df["high"].iloc[-1] >= df["high"].rolling(6).max().iloc[-2]) and (df["close"].iloc[-1] < df["open"].iloc[-1])

    rsi_up   = rsi.iloc[-2] < rma.iloc[-2] and rsi.iloc[-1] > rma.iloc[-1] and rsi.iloc[-1] < 70
    rsi_down = rsi.iloc[-2] > rma.iloc[-2] and rsi.iloc[-1] < rma.iloc[-1] and rsi.iloc[-1] > 30

    in_bull_zone = min(fib_a, fib_b) <= price <= max(fib_a, fib_b) and trend == "down"
    in_bear_zone = min(fib_a, fib_b) <= price <= max(fib_a, fib_b) and trend == "up"

    golden_bottom = in_bull_zone and bull_sweep and rsi_up and vol_ok and adx_val >= GZ_ADX_GATE
    golden_top    = in_bear_zone and bear_sweep and rsi_down and vol_ok and adx_val >= GZ_ADX_GATE

    info = {"trend": trend, "fib_low": float(min(fib_a,fib_b)), "fib_high": float(max(fib_a,fib_b)),
            "vol_ok": bool(vol_ok), "rsi_cross_up": bool(rsi_up), "rsi_cross_dn": bool(rsi_down)}
    return bool(golden_bottom), bool(golden_top), info

# ===========================
# FLOW / FOOTPRINT
# ===========================
def get_orderbook_snapshot(symbol) -> Dict:
    try:
        ob = exchange.fetch_order_book(symbol, limit=5)
        best_bid = ob["bids"][0][0] if ob["bids"] else 0
        best_ask = ob["asks"][0][0] if ob["asks"] else 0
        return {"best_bid": best_bid, "best_ask": best_ask}
    except Exception:
        return {"best_bid": 0, "best_ask": 0}

def compute_flow_metrics(df) -> Dict:
    close = df["close"].values
    vol   = df["volume"].values
    delta = (np.sign(np.diff(close, prepend=close[0])) * vol)
    cvd   = float(np.cumsum(delta)[-1])
    bias  = float(np.tanh(cvd / (np.std(delta) + 1e-9)))
    return {"delta": float(delta[-1]), "cvd": cvd, "fp_bias": bias}

# ===========================
# COUNCIL
# ===========================
def compute_indicators(df: pd.DataFrame) -> Dict:
    adx, di_plus, di_minus, atr = compute_adx(df, 14)
    rsi = compute_rsi(df["close"], 14)
    rf_sig, _, _ = compute_range_filter(df, RF_PERIOD, RF_MULT)
    maca = compute_maca(df["close"].values, MACA_FAST, MACA_SLOW, MACA_ANGLE_LEN)
    bos_up, bos_dn = detect_bos(df, lb=5) if ENABLE_SMC_BOS else (False, False)
    ob_bull, ob_bear = detect_order_blocks(df, SMC_LOOKBACK, SMC_MIN_DISP) if ENABLE_SMC_OB else (False, False)
    fvg_bull, fvg_bear = detect_fvg(df, FVG_MIN_GAP_PCT) if ENABLE_SMC_FVG else (False, False)
    ict_bull, ict_bear = detect_ict(df, SMC_MIN_DISP) if ENABLE_SMC_ICT else (False, False)

    return {
        "rsi": float(rsi.iloc[-1]),
        "adx": float(adx.iloc[-1]),
        "di_plus": float(di_plus.iloc[-1]),
        "di_minus": float(di_minus.iloc[-1]),
        "atr": float(atr.iloc[-1]),
        "close": float(df["close"].iloc[-1]),
        "atr_pct": float((atr.iloc[-1] / max(1e-9, df["close"].iloc[-1]))),
        "rf_sig": int(rf_sig[-1]),
        "maca": maca,
        "bos_up": bool(bos_up), "bos_dn": bool(bos_dn),
        "ob_bull": bool(ob_bull), "ob_bear": bool(ob_bear),
        "fvg_bull": bool(fvg_bull), "fvg_bear": bool(fvg_bear),
        "ict_bull": bool(ict_bull), "ict_bear": bool(ict_bear),
        "vol_ma20": float(df["volume"].rolling(20).mean().iloc[-1]),
        "volume": float(df["volume"].iloc[-1])
    }

def compute_min_confidence_with_footprint(base_min: float, fp_bias: float) -> float:
    return min(base_min, 6.0) if abs(fp_bias) >= 0.8 else base_min

def early_trend_detector(df, indicators, flow_metrics):
    c = df['close'].values
    adx = indicators.get("adx", 0.0)
    vol_ma20 = indicators.get("vol_ma20", 0.0)
    vol_last = indicators.get("volume", 0.0)
    maca = indicators.get("maca", {})
    fp_bias = flow_metrics.get("fp_bias", 0.0)
    bos_up  = indicators.get("bos_up", False)
    bos_dn  = indicators.get("bos_dn", False)
    burst = (vol_last > vol_ma20 * HUNTER_VOL_BURST_MULT) if vol_ma20 else False
    disp_up = (c[-1] - c[-2]) / max(1e-9, c[-2]) >= HUNTER_MIN_DISPLACEMENT
    disp_dn = (c[-2] - c[-1]) / max(1e-9, c[-2]) >= HUNTER_MIN_DISPLACEMENT
    maca_up = maca.get("cross_up") and maca.get("angle_deg",0) >= MACA_MIN_ANGLE
    maca_dn = maca.get("cross_dn") and abs(maca.get("angle_deg",0)) >= MACA_MIN_ANGLE
    b = 0.0; s = 0.0; why = []
    if adx >= HUNTER_TREND_GATE_ADX:
        if bos_up:      b += 1.2; why.append("BOS↑")
        if bos_dn:      s += 1.2; why.append("BOS↓")
        if burst and disp_up: b += 1.4; why.append("VOLx & DISP↑")
        if burst and disp_dn: s += 1.4; why.append("VOLx & DISP↓")
        if maca_up:     b += 1.2; why.append("MACA↑")
        if maca_dn:     s += 1.2; why.append("MACA↓")
        if fp_bias >= HUNTER_FLOW_MIN_BIAS:  b += 0.8; why.append("FP bias BUY")
        if fp_bias <= -HUNTER_FLOW_MIN_BIAS: s += 0.8; why.append("FP bias SELL")
    return round(b,2), round(s,2), ",".join(why)

def council_decision(df: pd.DataFrame, indicators: Dict, flow_metrics: Dict, state: Dict) -> Tuple[str, Dict, float]:
    council = {"votes_b":0.0, "votes_s":0.0, "score_b":0.0, "score_s":0.0, "debug":{}}

    rf_sig = indicators["rf_sig"]
    if rf_sig > 0: council["votes_b"] += 1; council["score_b"] += 0.8
    elif rf_sig < 0: council["votes_s"] += 1; council["score_s"] += 0.8

    adx = indicators["adx"]; di_p = indicators["di_plus"]; di_m = indicators["di_minus"]
    strong_trend = adx >= 28 and abs(di_p - di_m) >= 8
    if strong_trend:
        if di_p > di_m: council["votes_b"] += 3; council["score_b"] += 1.5
        else:           council["votes_s"] += 3; council["score_s"] += 1.5

    rsi = indicators["rsi"]; damp = 0.8 if 45 <= rsi <= 55 else 1.0

    if indicators["volume"] > indicators["vol_ma20"] * 1.4:
        if rf_sig > 0: council["score_b"] += 1.2
        elif rf_sig < 0: council["score_s"] += 1.2

    if ENABLE_SMC_BOS:
        if indicators["bos_up"]: council["votes_b"] += 2; council["score_b"] += 1.2
        if indicators["bos_dn"]: council["votes_s"] += 2; council["score_s"] += 1.2
    if ENABLE_SMC_OB:
        if indicators["ob_bull"]: council["votes_b"] += 2; council["score_b"] += 1.5
        if indicators["ob_bear"]: council["votes_s"] += 2; council["score_s"] += 1.5
    if ENABLE_SMC_FVG:
        if indicators["fvg_bull"]: council["votes_b"] += 1; council["score_b"] += 0.8
        if indicators["fvg_bear"]: council["votes_s"] += 1; council["score_s"] += 0.8
    if ENABLE_SMC_ICT:
        if indicators["ict_bull"]: council["votes_b"] += 2; council["score_b"] += 1.6
        if indicators["ict_bear"]: council["votes_s"] += 2; council["score_s"] += 1.6

    maca = indicators["maca"]
    if maca.get("cross_up") and maca.get("angle_deg",0) >= MACA_MIN_ANGLE:
        council["votes_b"] += 2; council["score_b"] += 1.0
    if maca.get("cross_dn") and abs(maca.get("angle_deg",0)) >= MACA_MIN_ANGLE:
        council["votes_s"] += 2; council["score_s"] += 1.0

    fp_bias = flow_metrics.get("fp_bias", 0.0)
    min_confidence = compute_min_confidence_with_footprint(ULTIMATE_MIN_CONFIDENCE, fp_bias)

    hunter_b, hunter_s, hunter_why = early_trend_detector(df, indicators, flow_metrics)
    council["score_b"] += hunter_b; council["score_s"] += hunter_s
    council["votes_b"] += (1 if hunter_b >= 1.0 else 0)
    council["votes_s"] += (1 if hunter_s >= 1.0 else 0)
    council["debug"]["hunter"] = {"b": hunter_b, "s": hunter_s, "why": hunter_why}

    golden_bottom = bool(state.get("golden_bottom", False))
    golden_top    = bool(state.get("golden_top", False))
    adx_val = adx

    decision, reasons = "WAIT", []
    b_ok = (council["score_b"]*damp) >= min_confidence and (council["score_b"] > council["score_s"] + BUY_SUPERIORITY)
    s_ok = (council["score_s"]*damp) >= min_confidence and (council["score_s"] > council["score_b"] + SELL_SUPERIORITY)
    if b_ok: decision="BUY"; reasons.append("std-buy")
    elif s_ok: decision="SELL"; reasons.append("std-sell")

    if decision=="WAIT" and GZ_ENABLED and adx_val >= GZ_ADX_GATE:
        if golden_bottom and (council["score_b"] >= GZ_MIN_SCORE):
            decision="BUY"; reasons.append("golden-force")
        elif golden_top and (council["score_s"] >= GZ_MIN_SCORE):
            decision="SELL"; reasons.append("golden-force")

    if decision=="WAIT" and PROACTIVE_HUNTER and adx_val >= HUNTER_TREND_GATE_ADX:
        if hunter_b >= HUNTER_STRONG_SCORE: decision="BUY"; reasons.append("hunter-early-trend")
        elif hunter_s >= HUNTER_STRONG_SCORE: decision="SELL"; reasons.append("hunter-early-trend")

    # Scalp Guard
    if decision in ("BUY","SELL"):
        target_pct = TP1_PCT_BASE
        is_scalp = (target_pct <= SCALP_MAX_TP_PCT)
        book = get_orderbook_snapshot(SYMBOL)
        spread_bps = ((book.get("best_ask",0) - book.get("best_bid",0)) / max(1e-9, (book.get("best_ask",0)+book.get("best_bid",0))/2))*10000 if book.get("best_ask",0) and book.get("best_bid",0) else 0.0
        slip_bps = SLIPPAGE_BPS_BASE + SLIPPAGE_BPS_ATR_MULT * (indicators["atr_pct"]*100.0)
        total_cost = FEES_TAKER_BPS + spread_bps + slip_bps
        edge_bps = (target_pct*100.0) - total_cost
        maca_angle = abs(maca.get("angle_deg", 0.0))
        strength = float((council["score_b"] if decision=="BUY" else council["score_s"]) + (adx_val/10.0) + abs(fp_bias) + (maca_angle/15.0))
        council["debug"]["scalp_guard"] = {"is_scalp": is_scalp, "edge_bps": round(edge_bps,2),
                                           "total_cost_bps": round(total_cost,2), "strength": round(strength,2)}
        if is_scalp:
            blocked = False
            if SCALP_ALLOW_ONLY_IF_POS and edge_bps < SCALP_MIN_EDGE_BPS:
                blocked = True; reasons.append("scalp-ev-negative")
            if strength < SCALP_MIN_STRENGTH:
                blocked = True; reasons.append("scalp-weak")
            if blocked: decision="WAIT"

    council["debug"]["decision_reasons"] = reasons
    return decision, council, min_confidence

# ===========================
# EXECUTION / POSITION
# ===========================
state = {
    "position": None,
    "last_close_side": None,
    "wait_for_next_signal_side": None,
    "cooldown_until": 0,
    "compound_pnl": 0.0,
    "golden_bottom": False,
    "golden_top": False,
    "trades_last_hour": deque(maxlen=60)
}

def fetch_position():
    try:
        positions = exchange.fetch_positions([SYMBOL])
        for p in positions:
            amt = float(p.get("contracts") or p.get("positionAmt") or 0.0)
            if abs(amt) > 0:
                entry = float(p.get("entryPrice") or 0.0)
                side  = "LONG" if amt > 0 else "SHORT"
                return {"side": side, "qty": abs(amt), "entry": entry}
    except Exception:
        pass
    return None

def place_order(side: str, qty: float):
    if DRY_RUN:
        log.info(f"🧪 DRY RUN ORDER {side} qty={qty}")
        return {"status":"ok","price":None}
    typ = "buy" if side=="BUY" else "sell"
    return exchange.create_order(SYMBOL, "market", typ, qty)

# ===========================
# TRADE MANAGEMENT
# ===========================
def dynamic_tp_plan(side: str, indicators: Dict, council: Dict, flow: Dict):
    side_score = (council.get("score_b",0) if side=="BUY" else council.get("score_s",0))
    adx_val    = indicators.get("adx", 0.0)
    fp_bias    = abs(flow.get("fp_bias", 0.0))
    maca_angle = abs(indicators.get("maca",{}).get("angle_deg", 0.0))
    strength   = float(side_score + (adx_val/10.0) + fp_bias + (maca_angle/15.0))
    if strength < 6.5:
        plan = [(TP1_PCT_BASE, float(TP_LADDER_QTY.split(",")[0]))]
    elif strength < 9.0:
        plan = [(TP1_PCT_BASE, float(TP_LADDER_QTY.split(",")[0])),
                (TP2_PCT_BASE, float(TP_LADDER_QTY.split(",")[1]))]
    else:
        q = [float(x) for x in TP_LADDER_QTY.split(",")]
        plan = [(TP1_PCT_BASE, q[0]), (TP2_PCT_BASE, q[1]), (TP3_PCT_BASE, q[2])]
    return plan, strength

def manage_open_trade(df, indicators, council, flow):
    pos = state["position"]
    if not pos: return
    entry = pos["entry"]; side_long = (pos["side"]=="LONG")
    close = float(df["close"].iloc[-1])
    atr   = indicators.get("atr", 0.0)

    tp_plan, strength = dynamic_tp_plan("BUY" if side_long else "SELL", indicators, council, flow)

    remaining = pos["qty"]
    for pct, q_pct in tp_plan:
        target = entry * (1 + pct/100.0) if side_long else entry * (1 - pct/100.0)
        hit = (close >= target) if side_long else (close <= target)
        if hit and remaining > FINAL_CHUNK_QTY:
            qty_close = max(FINAL_CHUNK_QTY, remaining * (q_pct/100.0))
            if not DRY_RUN:
                side = "SELL" if side_long else "BUY"
                place_order(side, qty_close)
            remaining -= qty_close
            log_tp_event(side_long, pct, qty_close, remaining)

    trail_mult = 1.6
    adverse_move = (close <= entry - atr*trail_mult) if side_long else (close >= entry + atr*trail_mult)
    if adverse_move and remaining > FINAL_CHUNK_QTY:
        if not DRY_RUN:
            side = "SELL" if side_long else "BUY"
            place_order(side, remaining - FINAL_CHUNK_QTY)
        remaining = FINAL_CHUNK_QTY
        log_tp_event(side_long, "ATR-TRAIL", remaining, remaining)

# ===========================
# MAIN LOOP
# ===========================
def trade_loop():
    markets = exchange.load_markets()
    while True:
        try:
            if time.time() < state["cooldown_until"]:
                time.sleep(LOOP_SLEEP_SEC); continue

            ohlcv = fetch_ohlcv(SYMBOL, TIMEFRAME, LOOKBACK_BARS)
            df = to_df(ohlcv)
            indicators = compute_indicators(df)
            flow = compute_flow_metrics(df)
            flow["book"] = get_orderbook_snapshot(SYMBOL)

            # spread guard
            bid, ask = flow["book"].get("best_bid",0), flow["book"].get("best_ask",0)
            spread_bps = ((ask - bid)/max(1e-9, (ask+bid)/2))*10000 if ask and bid else 0.0
            if spread_bps > MAX_SPREAD_BPS:
                log_guard("Spread", f"{spread_bps:.2f} bps > {MAX_SPREAD_BPS}")
                time.sleep(LOOP_SLEEP_SEC); continue

            state["position"] = fetch_position()

            # Golden Zones detection
            if GZ_ENABLED:
                golden_b, golden_t, gz_info = detect_golden_zones(df, indicators["adx"], GZ_VOL_MULT, GZ_FIB_LOW, GZ_FIB_HIGH, GZ_RSI_MA)
                state["golden_bottom"] = golden_b
                state["golden_top"]    = golden_t
                if golden_b or golden_t: log_golden(golden_b, golden_t, gz_info)

            # RF wait-after-close
            if state["wait_for_next_signal_side"]:
                need = state["wait_for_next_signal_side"]
                if (need=="BUY" and indicators["rf_sig"]<=0) or (need=="SELL" and indicators["rf_sig"]>=0):
                    log_guard("RF-Wait", f"Waiting RF flip for {need}")
                    time.sleep(LOOP_SLEEP_SEC); continue
                else:
                    state["wait_for_next_signal_side"] = None

            decision, council, min_conf = council_decision(df, indicators, flow, state)

            if decision in ("BUY","SELL") and ENTRY_RF_ONLY:
                if (decision=="BUY" and indicators["rf_sig"]<=0) or (decision=="SELL" and indicators["rf_sig"]>=0):
                    council["debug"]["decision_reasons"] = council["debug"].get("decision_reasons",[])+["entry-rf-only-block"]
                    decision = "WAIT"

            # إدارة مركز مفتوح
            if state["position"]:
                manage_open_trade(df, indicators, council, flow)
                time.sleep(LOOP_SLEEP_SEC); continue

            # ===== ENTRY =====
            if decision in ("BUY","SELL"):
                reasons = ",".join(council["debug"].get("decision_reasons", []))

                # sizing with minQty/step
                balance = exchange.fetch_balance()
                usdt = 0.0
                if "USDT" in balance:
                    usdt = float(balance["USDT"].get("free", balance["USDT"].get("total", 0.0)))
                elif "free" in balance and "USDT" in balance["free"]:
                    usdt = float(balance["free"]["USDT"])
                else:
                    usdt = float(balance.get("total", {}).get("USDT", 50.0))

                price = float(df["close"].iloc[-1])
                notional = max(5.0, usdt * RISK_ALLOC * LEVERAGE)

                m = markets.get(SYMBOL, {})
                limits = (m.get("limits", {}) or {}).get("amount", {}) if m else {}
                prec = (m.get("precision", {}) or {}).get("amount", None) if m else None
                amt_min = float(limits.get("min") or 10.0)    # Bybit SUI≈10
                step    = float(prec if prec not in (None, 0) else 0.001)

                def _round_step(x, step): return round(x / step) * step
                raw_qty = notional / price
                qty = _round_step(max(amt_min, raw_qty), step)

                if not DRY_RUN: place_order(decision, qty)
                state["position"] = {"side":"LONG" if decision=="BUY" else "SHORT", "qty": qty, "entry": price}
                log_trade_open(decision, qty, price, reasons, min_conf)

                if WAIT_NEXT_SIGNAL:
                    state["last_close_side"] = decision
                    state["wait_for_next_signal_side"] = "SELL" if decision=="BUY" else "BUY"

                state["cooldown_until"] = time.time() + 2

            time.sleep(LOOP_SLEEP_SEC)

        except Exception as e:
            log.error("Loop error: "+str(e))
            log.debug(traceback.format_exc())
            time.sleep(LOOP_SLEEP_SEC)

# ===========================
# FLASK
# ===========================
app = Flask(__name__)

@app.route("/")
def root():
    return jsonify({"ok": True, "msg": "bot alive", "time": datetime.utcnow().isoformat()+"Z"})

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "time": datetime.utcnow().isoformat()+"Z",
        "symbol": SYMBOL,
        "exchange": EXCHANGE_NAME,
        "position": state["position"],
        "wait_for_next_signal_side": state["wait_for_next_signal_side"],
        "golden_bottom": state["golden_bottom"],
        "golden_top": state["golden_top"]
    })

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "exchange": EXCHANGE_NAME,
        "rf": {"period": RF_PERIOD, "mult": RF_MULT, "entry_rf_only": ENTRY_RF_ONLY},
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY,
                   "max_trades_per_hr": MAX_TRADES_PER_HR, "cooldown_after_close": COOLDOWN_AFTER_CLOSE_SEC},
        "council": {"min_conf": ULTIMATE_MIN_CONFIDENCE, "sell_superiority": SELL_SUPERIORITY,
                    "buy_superiority": BUY_SUPERIORITY},
        "hunter": {"enabled": PROACTIVE_HUNTER, "score_gate": HUNTER_STRONG_SCORE,
                   "adx_gate": HUNTER_TREND_GATE_ADX},
        "golden": {"enabled": GZ_ENABLED, "adx_gate": GZ_ADX_GATE, "min_score": GZ_MIN_SCORE,
                   "fib_low": GZ_FIB_LOW, "fib_high": GZ_FIB_HIGH, "vol_mult": GZ_VOL_MULT, "rsi_ma": GZ_RSI_MA},
        "smc": {"ob": ENABLE_SMC_OB, "fvg": ENABLE_SMC_FVG, "bos": ENABLE_SMC_BOS, "ict": ENABLE_SMC_ICT,
                "lookback": SMC_LOOKBACK, "min_disp": SMC_MIN_DISP},
        "maca": {"fast": MACA_FAST, "slow": MACA_SLOW, "angle_min": MACA_MIN_ANGLE},
        "tp_dynamic": {"tp1": TP1_PCT_BASE, "tp2": TP2_PCT_BASE, "tp3": TP3_PCT_BASE, "ladder": TP_LADDER_QTY}
    })

@app.route("/logs")
def logs_view():
    return jsonify(list(LOG_RING))

def keepalive_loop():
    if not SELF_URL: return
    import urllib.request
    while True:
        try:
            urllib.request.urlopen(SELF_URL+"/health", timeout=10).read()
        except Exception:
            pass
        time.sleep(KEEPALIVE_INTERVAL)

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    t = threading.Thread(target=trade_loop, daemon=True); t.start()
    if SELF_URL:
        threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
