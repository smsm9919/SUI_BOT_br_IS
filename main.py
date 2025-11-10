# file: sui_bot_pro_hunter.py
# -*- coding: utf-8 -*-
"""
BYBIT/BINGX — SUI Perp Council PRO HUNTER
- Council: RF + ADX/DI + RSI + MACA + SMC(OB/FVG/BOS/ICT) + Footprint/Flow + Volume/ATR
- Pro-Hunter (early trend) + Golden Force Entry (قاع/قمة ذهبية مؤكدة)
- Scalp Guard: لا سكالب إلا لو EV بعد الخصوم موجب وكافي + قوة صفقة مرتفعة
- Dynamic TP (1/2/3) + Breakeven + ATR Trailing + Ratchet + Golden Reversal
- Execution Guards: Spread / Wait-next RF / One-Position
- Ops: /health /metrics + keepalive
"""

import os, time, math, json, logging, threading, traceback
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt
from flask import Flask, jsonify

# ===========================
#         SETTINGS
# ===========================
SYMBOL             = os.getenv("SYMBOL", "SUI/USDT:USDT")
EXCHANGE_NAME      = os.getenv("EXCHANGE", "bybit").lower()        # bybit | bingx
API_KEY            = os.getenv("API_KEY", os.getenv("BYBIT_API_KEY", os.getenv("BINGX_API_KEY", "")))
API_SECRET         = os.getenv("API_SECRET", os.getenv("BYBIT_API_SECRET", os.getenv("BINGX_API_SECRET", "")))
POSITION_MODE      = os.getenv("POSITION_MODE", "oneway")
LEVERAGE           = int(os.getenv("LEVERAGE", "10"))
TIMEFRAME          = os.getenv("TIMEFRAME", "15m")
LOOKBACK_BARS      = int(os.getenv("LOOKBACK_BARS", "400"))
LOOP_SLEEP_SEC     = float(os.getenv("LOOP_SLEEP_SEC", "6.0"))
DRY_RUN            = bool(int(os.getenv("DRY_RUN", "0")))

# RF + دخول/انتظار
RF_PERIOD          = int(os.getenv("RF_PERIOD", "20"))
RF_MULT            = float(os.getenv("RF_MULT", "3.5"))
ENTRY_RF_ONLY      = bool(int(os.getenv("ENTRY_RF_ONLY", "0")))    # يلزم انقلاب RF للدخول
WAIT_NEXT_SIGNAL   = bool(int(os.getenv("WAIT_NEXT_SIGNAL", "1"))) # انتظار الشمعة التالية بعد إغلاق بعكس RF

# Guards
MAX_SPREAD_BPS     = float(os.getenv("MAX_SPREAD_BPS", "8.0"))
FINAL_CHUNK_QTY    = float(os.getenv("FINAL_CHUNK_QTY", "2.0"))
MAX_TRADES_PER_HR  = int(os.getenv("MAX_TRADES_PER_HR", "8"))
COOLDOWN_AFTER_CLOSE_SEC = int(os.getenv("COOLDOWN_AFTER_CLOSE_SEC", "30"))

# Council thresholds
ULTIMATE_MIN_CONFIDENCE = float(os.getenv("ULTIMATE_MIN_CONFIDENCE", "7.0"))  # يُخفض إلى 6 مع Footprint قوي
SELL_SUPERIORITY        = float(os.getenv("SELL_SUPERIORITY", "1.0"))
BUY_SUPERIORITY         = float(os.getenv("BUY_SUPERIORITY",  "1.0"))

# Pro Hunter (التقاط بداية الترند)
PROACTIVE_HUNTER          = bool(int(os.getenv("PROACTIVE_HUNTER", "1")))
HUNTER_STRONG_SCORE       = float(os.getenv("HUNTER_STRONG_SCORE", "6.5"))
HUNTER_TREND_GATE_ADX     = float(os.getenv("HUNTER_TREND_GATE_ADX", "20"))
HUNTER_MIN_DISPLACEMENT   = float(os.getenv("HUNTER_MIN_DISPLACEMENT", "0.004"))  # 0.4%
HUNTER_VOL_BURST_MULT     = float(os.getenv("HUNTER_VOL_BURST_MULT", "1.6"))
HUNTER_FLOW_MIN_BIAS      = float(os.getenv("HUNTER_FLOW_MIN_BIAS", "0.8"))

# Golden force entry (اربطها بكاشفك الذهبي لاحقًا أو اترك False)
GOLDEN_FORCE_ENTRY        = bool(int(os.getenv("GOLDEN_FORCE_ENTRY", "1")))
GOLDEN_SCORE_GATE         = float(os.getenv("GOLDEN_SCORE_GATE", "6.0"))
GOLDEN_ADX_GATE           = float(os.getenv("GOLDEN_ADX_GATE", "20"))

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

# Scalp Guard (لا سكالب إلا لو مجدي بعد الخصوم + قوي)
FEES_TAKER_BPS          = float(os.getenv("FEES_TAKER_BPS", "7.0"))
FEES_MAKER_BPS          = float(os.getenv("FEES_MAKER_BPS", "2.0"))
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
#       LOGGING
# ===========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("PRO-HUNTER")

# ===========================
#       EXCHANGE
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
#     DATA / INDICATORS
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
    rsi = 100 - (100/(1+rs))
    return rsi

def compute_adx(df, length=14):
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm  = (h - h.shift(1)).clip(lower=0.0)
    minus_dm = (l.shift(1) - l).clip(lower=0.0)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
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

# ---- SMC helpers ----
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
        is_impulse_up = body[i] > 0 and (body[i]/max(1e-9, rng[i])) > 0.6 and (rng[i]/atr) > 1.2
        is_impulse_dn = body[i] < 0 and (abs(body[i])/max(1e-9, rng[i])) > 0.6 and (rng[i]/atr) > 1.2
        if is_impulse_up and (c[i+1] >= c[i]) and (c[-1] >= c[i]*(1+min_disp)): bull_ob = True
        if is_impulse_dn and (c[i+1] <= c[i]) and (c[-1] <= c[i]*(1-min_disp)): bear_ob = True
    return bull_ob, bear_ob

def detect_fvg(df, min_gap_pct=0.15):
    if len(df) < 3: return False, False
    atr = (df['high']-df['low']).rolling(14).mean().iloc[-1]
    thr = atr * (min_gap_pct/100.0)
    h, l = df['high'].values, df['low'].values
    bull_fvg = l[-1] > h[-3] + thr
    bear_fvg = h[-1] < l[-3] - thr
    return bull_fvg, bear_fvg

def detect_ict(df, min_disp=0.004):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    bull_sweep = l[-1] < min(l[-5:-1]) and c[-1] > c[-2]
    bear_sweep = h[-1] > max(h[-5:-1]) and c[-1] < c[-2]
    bull_disp = (c[-1] - c[-2]) / max(1e-9, c[-2]) >= min_disp
    bear_disp = (c[-2] - c[-1]) / max(1e-9, c[-2]) >= min_disp
    bull_retest = bull_sweep and bull_disp and (l[-1] <= c[-2])
    bear_retest = bear_sweep and bear_disp and (h[-1] >= c[-2])
    return bull_retest, bear_retest

# ===========================
#     FLOW / FOOTPRINT
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
    # مُبسّط: CVD/Delta تقريبية + footprint bias
    close = df["close"].values
    vol   = df["volume"].values
    delta = (np.sign(np.diff(close, prepend=close[0])) * vol)
    cvd   = float(np.cumsum(delta)[-1])
    bias  = float(np.tanh(cvd / (np.std(delta) + 1e-9)))  # [-1..1]
    return {"delta": float(delta[-1]), "cvd": cvd, "fp_bias": bias}

# ===========================
#         COUNCIL
# ===========================
def compute_indicators(df: pd.DataFrame) -> Dict:
    adx, di_plus, di_minus, atr = compute_adx(df, 14)
    rsi = compute_rsi(df["close"], 14)
    rf_sig, rf_up, rf_dn = compute_range_filter(df, RF_PERIOD, RF_MULT)
    maca = compute_maca(df["close"].values, MACA_FAST, MACA_SLOW, MACA_ANGLE_LEN)
    bos_up, bos_dn = detect_bos(df, lb=5) if ENABLE_SMC_BOS else (False, False)
    ob_bull, ob_bear = detect_order_blocks(df, SMC_LOOKBACK, SMC_MIN_DISP) if ENABLE_SMC_OB else (False, False)
    fvg_bull, fvg_bear = detect_fvg(df, FVG_MIN_GAP_PCT) if ENABLE_SMC_FVG else (False, False)
    ict_bull, ict_bear = detect_ict(df, SMC_MIN_DISP) if ENABLE_SMC_ICT else (False, False)

    indicators = {
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
    return indicators

def compute_min_confidence_with_footprint(base_min: float, fp_bias: float) -> float:
    # Footprint قوي يخفض العتبة
    if abs(fp_bias) >= 0.8:
        return min(base_min, 6.0)
    return base_min

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

    # RF اتجاه مبدئي
    rf_sig = indicators["rf_sig"]
    if rf_sig > 0:
        council["votes_b"] += 1; council["score_b"] += 0.8
    elif rf_sig < 0:
        council["votes_s"] += 1; council["score_s"] += 0.8

    # ADX/Trend
    adx = indicators["adx"]; di_p = indicators["di_plus"]; di_m = indicators["di_minus"]
    strong_trend = adx >= 28 and abs(di_p - di_m) >= 8
    if strong_trend:
        if di_p > di_m: council["votes_b"] += 3; council["score_b"] += 1.5
        else:           council["votes_s"] += 3; council["score_s"] += 1.5

    # RSI حياد يخفّض الثقة
    rsi = indicators["rsi"]
    neutral = 45 <= rsi <= 55
    damp = 0.8 if neutral else 1.0

    # ATR/Volume
    if indicators["volume"] > indicators["vol_ma20"] * 1.4:
        if rf_sig > 0: council["score_b"] += 1.2
        elif rf_sig < 0: council["score_s"] += 1.2

    # SMC (OB/FVG/BOS/ICT)
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
        ict_b, ict_s = indicators["ict_bull"], indicators["ict_bear"]
        if ict_b: council["votes_b"] += 2; council["score_b"] += 1.6
        if ict_s: council["votes_s"] += 2; council["score_s"] += 1.6

    # MACA
    maca = indicators["maca"]
    if maca.get("cross_up") and maca.get("angle_deg",0) >= MACA_MIN_ANGLE:
        council["votes_b"] += 2; council["score_b"] += 1.0
    if maca.get("cross_dn") and abs(maca.get("angle_deg",0)) >= MACA_MIN_ANGLE:
        council["votes_s"] += 2; council["score_s"] += 1.0

    # Footprint
    fp_bias = flow_metrics.get("fp_bias", 0.0)  # [-1..1]
    min_confidence = compute_min_confidence_with_footprint(ULTIMATE_MIN_CONFIDENCE, fp_bias)

    # Early-trend Hunter
    hunter_b, hunter_s, hunter_why = early_trend_detector(df, indicators, flow_metrics)
    council["score_b"] += hunter_b; council["score_s"] += hunter_s
    council["votes_b"] += (1 if hunter_b >= 1.0 else 0)
    council["votes_s"] += (1 if hunter_s >= 1.0 else 0)
    council["debug"]["hunter"] = {"b": hunter_b, "s": hunter_s, "why": hunter_why}

    # Golden hints (اربطها لاحقًا بكاشف القاع/القمة الذهبية الحقيقي لديك)
    golden_bottom = bool(state.get("golden_bottom", False))
    golden_top    = bool(state.get("golden_top", False))
    adx_val = adx

    # قرار أساسي
    decision, reasons = "WAIT", []
    b_ok = (council["score_b"]*damp) >= min_confidence and (council["score_b"] > council["score_s"] + BUY_SUPERIORITY)
    s_ok = (council["score_s"]*damp) >= min_confidence and (council["score_s"] > council["score_b"] + SELL_SUPERIORITY)
    if b_ok: decision="BUY"; reasons.append("std-buy")
    elif s_ok: decision="SELL"; reasons.append("std-sell")

    # Golden force entry
    if decision=="WAIT" and GOLDEN_FORCE_ENTRY and adx_val >= GOLDEN_ADX_GATE:
        if golden_bottom and (council["score_b"] >= GOLDEN_SCORE_GATE):
            decision="BUY"; reasons.append("golden-force")
        elif golden_top and (council["score_s"] >= GOLDEN_SCORE_GATE):
            decision="SELL"; reasons.append("golden-force")

    # Proactive Hunter
    if decision=="WAIT" and PROACTIVE_HUNTER and adx_val >= HUNTER_TREND_GATE_ADX:
        if hunter_b >= HUNTER_STRONG_SCORE: decision="BUY"; reasons.append("hunter-early-trend")
        elif hunter_s >= HUNTER_STRONG_SCORE: decision="SELL"; reasons.append("hunter-early-trend")

    # ---- SCALP PROFIT GUARD: لا سكالب إلا لو مجدي وقوي ----
    if decision in ("BUY","SELL"):
        side = decision
        target_pct = TP1_PCT_BASE  # أقل هدف كمؤشر لتصنيف السكالب
        is_scalp = (target_pct <= SCALP_MAX_TP_PCT)

        # تكاليف: سبريد + انزلاق + عمولة taker
        book = get_orderbook_snapshot(SYMBOL)
        spread_bps = ((book.get("best_ask",0) - book.get("best_bid",0)) / max(1e-9, (book.get("best_ask",0)+book.get("best_bid",0))/2))*10000 if book.get("best_ask",0) and book.get("best_bid",0) else 0.0
        slip_bps = SLIPPAGE_BPS_BASE + SLIPPAGE_BPS_ATR_MULT * (indicators["atr_pct"]*100.0)
        total_cost = FEES_TAKER_BPS + spread_bps + slip_bps
        edge_bps = (target_pct*100.0) - total_cost

        # قوة الصفقة
        maca_angle = abs(maca.get("angle_deg", 0.0))
        strength = float((council["score_b"] if side=="BUY" else council["score_s"]) + (adx_val/10.0) + abs(fp_bias) + (maca_angle/15.0))

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
#   EXECUTION / POSITION
# ===========================
state = {
    "position": None,             # {"side":"LONG/SHORT","qty":..., "entry":...}
    "last_close_side": None,      # "BUY"/"SELL"
    "wait_for_next_signal_side": None,  # "BUY"/"SELL"
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
            amt = float(p.get("contracts") or p.get("contractSize") or p.get("positionAmt") or 0.0)
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

def place_tp_partial(symbol, side, entry_price, pct_target, qty_pct):
    # تنفيذ جزئي ماركت لحظة تحقق الهدف (يتم تشغيله من حلقة الإدارة)
    return {"target_pct": pct_target, "qty_pct": qty_pct}

def close_position(full: bool=True, qty: Optional[float]=None):
    pos = state["position"]
    if not pos: return
    side = "SELL" if pos["side"]=="LONG" else "BUY"
    q = pos["qty"] if full or not qty else min(qty, pos["qty"])
    return place_order(side, q)

# ===========================
#    TRADE MANAGEMENT
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

    # خطة TP مرنة
    tp_plan, strength = dynamic_tp_plan("BUY" if side_long else "SELL", indicators, council, flow)

    # تنفيذ TPs: تحقق الهدف → اغلق جزء
    remaining = pos["qty"]
    executed = []
    for pct, q_pct in tp_plan:
        target = entry * (1 + pct/100.0) if side_long else entry * (1 - pct/100.0)
        hit = (close >= target) if side_long else (close <= target)
        if hit and remaining > FINAL_CHUNK_QTY:
            qty_close = max(FINAL_CHUNK_QTY, remaining * (q_pct/100.0))
            if not DRY_RUN:
                place_order("SELL" if side_long else "BUY", qty_close)
            remaining -= qty_close
            executed.append((pct, qty_close))

    # Breakeven مبسّط: بعد +0.30% حرّس الربح
    be_trigger = 0.30/100.0
    if (side_long and close >= entry*(1+be_trigger)) or ((not side_long) and close <= entry*(1-be_trigger)):
        # يمكن إضافة منطق ستوب افتراضي/إغلاق ذكي عند انعكاس واضح
        pass

    # ATR Trail (ratchet) مبسّط: احمِ الربح عند انعكاس قوي مقابل اتجاهك
    trail_mult = 1.6
    adverse_move = (close <= entry - atr*trail_mult) if side_long else (close >= entry + atr*trail_mult)
    if adverse_move and remaining > FINAL_CHUNK_QTY:
        if not DRY_RUN:
            place_order("SELL" if side_long else "BUY", remaining - FINAL_CHUNK_QTY)
        remaining = FINAL_CHUNK_QTY
        executed.append(("ATR-TRAIL", remaining))

    log.info(f"🧭 Manage: strength={round(strength,2)} executed={executed} remaining≈{round(remaining,4)}")

# ===========================
#     MAIN TRADE LOOP
# ===========================
def trade_loop():
    while True:
        try:
            # Cooldown
            if time.time() < state["cooldown_until"]:
                time.sleep(LOOP_SLEEP_SEC); continue

            # بيانات
            ohlcv = fetch_ohlcv(SYMBOL, TIMEFRAME, LOOKBACK_BARS)
            df = to_df(ohlcv)
            indicators = compute_indicators(df)
            flow = compute_flow_metrics(df)
            flow["book"] = get_orderbook_snapshot(SYMBOL)

            # spread guard
            bid, ask = flow["book"].get("best_bid",0), flow["book"].get("best_ask",0)
            spread_bps = ((ask - bid)/max(1e-9, (ask+bid)/2))*10000 if ask and bid else 0.0
            if spread_bps > MAX_SPREAD_BPS:
                log.info(f"⛔ Spread guard: {spread_bps:.2f} bps > {MAX_SPREAD_BPS}")
                time.sleep(LOOP_SLEEP_SEC); continue

            # Position snapshot
            state["position"] = fetch_position()

            # حارس الانتظار بعد الإغلاق: إلزام انقلاب RF في الاتجاه المطلوب
            if state["wait_for_next_signal_side"]:
                need = state["wait_for_next_signal_side"]
                if (need=="BUY" and indicators["rf_sig"]<=0) or (need=="SELL" and indicators["rf_sig"]>=0):
                    log.info(f"⏳ Waiting RF flip for {need} ...")
                    time.sleep(LOOP_SLEEP_SEC); continue
                else:
                    state["wait_for_next_signal_side"] = None

            # قرار المجلس
            decision, council, min_conf = council_decision(df, indicators, flow, state)

            # ENTRY_RF_ONLY gate (اختياري)
            if decision in ("BUY","SELL") and ENTRY_RF_ONLY:
                if (decision=="BUY" and indicators["rf_sig"]<=0) or (decision=="SELL" and indicators["rf_sig"]>=0):
                    council["debug"]["decision_reasons"] = council["debug"].get("decision_reasons",[])+["entry-rf-only-block"]
                    decision = "WAIT"

            # إدارة الصفقة لو فيه مركز مفتوح
            if state["position"]:
                manage_open_trade(df, indicators, council, flow)
                time.sleep(LOOP_SLEEP_SEC); continue

            # دخول
            if decision in ("BUY","SELL"):
                reason = ",".join(council["debug"].get("decision_reasons", []))

                # حجم تقريبي (60% من الرصيد × ليفرج)
                balance = exchange.fetch_balance()
                # مرونة لاختلاف هيكل الرصيد بين البورصات
                usdt = 0.0
                if "USDT" in balance:
                    usdt = float(balance["USDT"].get("free", balance["USDT"].get("total", 0.0)))
                elif "free" in balance and "USDT" in balance["free"]:
                    usdt = float(balance["free"]["USDT"])
                else:
                    usdt = float(balance.get("total", {}).get("USDT", 50.0))
                price = float(df["close"].iloc[-1])
                notional = max(5.0, usdt * 0.60 * LEVERAGE)
                qty = max(0.1, round(notional/price, 3))

                if not DRY_RUN:
                    place_order(decision, qty)

                state["position"] = {"side":"LONG" if decision=="BUY" else "SHORT", "qty": qty, "entry": price}
                log.info(f"🎯 EXECUTE {decision} qty={qty} price≈{price} | min_conf={min_conf:.2f} | reasons={reason}")

                # طلب انتظار انقلاب RF بعد الإغلاق
                if WAIT_NEXT_SIGNAL:
                    state["last_close_side"] = decision
                    state["wait_for_next_signal_side"] = "SELL" if decision=="BUY" else "BUY"

                # تبريد بسيط بعد الدخول
                state["cooldown_until"] = time.time() + 2

            time.sleep(LOOP_SLEEP_SEC)

        except Exception as e:
            log.error("Loop error: "+str(e))
            log.debug(traceback.format_exc())
            time.sleep(LOOP_SLEEP_SEC)

# ===========================
#         FLASK API
# ===========================
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({
        "status":"ok",
        "time": datetime.utcnow().isoformat()+"Z",
        "symbol": SYMBOL,
        "exchange": EXCHANGE_NAME,
        "position": state["position"],
        "wait_for_next_signal_side": state["wait_for_next_signal_side"]
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
                   "adx_gate": HUNTER_TREND_GATE_ADX, "flow_bias": HUNTER_FLOW_MIN_BIAS},
        "golden": {"force": GOLDEN_FORCE_ENTRY, "score_gate": GOLDEN_SCORE_GATE, "adx_gate": GOLDEN_ADX_GATE},
        "smc": {"ob": ENABLE_SMC_OB, "fvg": ENABLE_SMC_FVG, "bos": ENABLE_SMC_BOS, "ict": ENABLE_SMC_ICT,
                "lookback": SMC_LOOKBACK, "min_disp": SMC_MIN_DISP},
        "maca": {"fast": MACA_FAST, "slow": MACA_SLOW, "angle_min": MACA_MIN_ANGLE},
        "tp_dynamic": {"tp1": TP1_PCT_BASE, "tp2": TP2_PCT_BASE, "tp3": TP3_PCT_BASE, "ladder": TP_LADDER_QTY},
        "scalp_guard": {"max_tp_pct": SCALP_MAX_TP_PCT, "min_edge_bps": SCALP_MIN_EDGE_BPS,
                        "min_strength": SCALP_MIN_STRENGTH}
    })

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
#           MAIN
# ===========================
if __name__ == "__main__":
    t = threading.Thread(target=trade_loop, daemon=True)
    t.start()

    if SELF_URL:
        threading.Thread(target=keepalive_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=PORT)
