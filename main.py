# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO TRADING BOT — COUNCIL INTELLIGENCE SYSTEM
• ULTRA INTELLIGENT COUNCIL DECISION MAKING
• MULTI-TIMEFRAME CONFIRMATION 
• ADVANCED MARKET STRUCTURE ANALYSIS
• SMART MONEY FLOW DETECTION
• PRECISE ENTRY & EXIT TIMING
• ADAPTIVE PROFIT MAXIMIZATION
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

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Enhanced Logging ====
LOG_LEGACY = False
LOG_ADDONS = True

BOT_VERSION = f"SUI ULTRA PRO TRADER v6.0 — {EXCHANGE_NAME.upper()}"
print("🚀 BOOTING:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# =================== ULTRA ENHANCED SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# ==== ULTRA INTELLIGENT COUNCIL PARAMETERS ====
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))
RF_MULT   = float(os.getenv("RF_MULT", 3.0))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# ==== MULTI-TIMEFRAME CONFIRMATION ====
MTF_ANALYSIS = True
MTF_TIMEFRAMES = ["5m", "15m", "1h"]
MTF_CONFIRMATION_REQUIRED = 2  # Minimum 2 timeframes must confirm

# ==== ADVANCED INDICATOR SETTINGS ====
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14
VWAP_LEN = 20
ICHIMOKU_SETTINGS = (9, 26, 52)
BOLLINGER_SETTINGS = (20, 2)
MACD_SETTINGS = (12, 26, 9)
STOCH_SETTINGS = (14, 3, 3)

# ==== SMART PROFIT MAXIMIZATION ====
SCALP_TP_TARGETS = [0.45, 0.80, 1.20]
SCALP_TP_FRACTIONS = [0.50, 0.30, 0.20]

TREND_TP_TARGETS = [0.60, 1.50, 2.50, 4.00]
TREND_TP_FRACTIONS = [0.30, 0.30, 0.25, 0.15]

BREAKEVEN_AFTER = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT = 1.8

# ==== AGGRESSIVE ENTRY SETTINGS ====
GOLDEN_ENTRY_SCORE = 7.0
GOLDEN_ENTRY_ADX = 22.0
GOLDEN_REVERSAL_SCORE = 7.5

# ==== MARKET STRUCTURE DETECTION ====
TREND_STRENGTH_THRESHOLD = 25
VOLUME_CONFIRMATION_THRESHOLD = 1.2
MOMENTUM_CONFIRMATION_PERIOD = 3

# =================== ULTRA ENHANCED INDICATORS ===================
def super_trend(df, period=10, multiplier=3):
    """مؤشر SuperTrend المتقدم"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
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

def ichimoku_cloud(df, tenkan=9, kijun=26, senkou=52):
    """مؤشر Ichimoku Cloud المتقدم"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        
        tenkan_sen = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2
        kijun_sen = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_span_b = ((h.rolling(senkou).max() + l.rolling(senkou).min()) / 2).shift(kijun)
        chikou_span = c.shift(-kijun)
        
        current_price = c.iloc[-1]
        above_cloud = current_price > max(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1])
        below_cloud = current_price < min(senkou_span_a.iloc[-1], senkou_span_b.iloc[-1])
        
        tenkan_kijun_bullish = tenkan_sen.iloc[-1] > kijun_sen.iloc[-1]
        price_above_kijun = current_price > kijun_sen.iloc[-1]
        
        cloud_green = senkou_span_a.iloc[-1] > senkou_span_b.iloc[-1]
        
        signal = "buy" if (above_cloud and tenkan_kijun_bullish and price_above_kijun) else \
                 "sell" if (below_cloud and not tenkan_kijun_bullish and not price_above_kijun) else "neutral"
        
        return {
            "tenkan": float(tenkan_sen.iloc[-1]),
            "kijun": float(kijun_sen.iloc[-1]),
            "senkou_a": float(senkou_span_a.iloc[-1]),
            "senkou_b": float(senkou_span_b.iloc[-1]),
            "chikou": float(chikou_span.iloc[-1]),
            "above_cloud": above_cloud,
            "below_cloud": below_cloud,
            "cloud_green": cloud_green,
            "signal": signal,
            "strength": abs(current_price - kijun_sen.iloc[-1]) / current_price * 100
        }
    except Exception as e:
        return {"signal": "neutral", "strength": 0}

def bollinger_bands_advanced(df, period=20, std=2):
    """مؤشر Bollinger Bands المتقدم"""
    try:
        c = df['close'].astype(float)
        sma = c.rolling(period).mean()
        std_dev = c.rolling(period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        current_price = c.iloc[-1]
        bandwidth = (upper_band - lower_band) / sma * 100
        current_bandwidth = bandwidth.iloc[-1]
        
        # %B indicator
        percent_b = (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
        
        # Squeeze detection
        squeeze = current_bandwidth < 10  # Low volatility
        
        signal = "buy" if (current_price <= lower_band.iloc[-1] and not squeeze) else \
                 "sell" if (current_price >= upper_band.iloc[-1] and not squeeze) else "neutral"
        
        return {
            "upper": float(upper_band.iloc[-1]),
            "middle": float(sma.iloc[-1]),
            "lower": float(lower_band.iloc[-1]),
            "bandwidth": float(current_bandwidth),
            "percent_b": float(percent_b),
            "squeeze": bool(squeeze),
            "signal": signal,
            "strength": abs(percent_b - 0.5) * 2  # 0 to 1 strength
        }
    except Exception as e:
        return {"signal": "neutral", "strength": 0}

def volume_weighted_macd(df, fast=12, slow=26, signal=9):
    """MACD مع ترجيح حجم التداول"""
    try:
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        # Volume weighted prices
        vwp = (c * v).cumsum() / v.cumsum()
        
        ema_fast = vwp.ewm(span=fast).mean()
        ema_slow = vwp.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        current_macd = macd.iloc[-1]
        current_signal = macd_signal.iloc[-1]
        current_histogram = macd_histogram.iloc[-1]
        
        bullish_cross = current_macd > current_signal and macd.iloc[-2] <= macd_signal.iloc[-2]
        bearish_cross = current_macd < current_signal and macd.iloc[-2] >= macd_signal.iloc[-2]
        
        histogram_rising = current_histogram > macd_histogram.iloc[-2]
        
        signal = "buy" if (bullish_cross and histogram_rising) else \
                 "sell" if (bearish_cross and not histogram_rising) else "neutral"
        
        strength = abs(current_histogram) / (c.rolling(50).std().iloc[-1] + 1e-12)
        
        return {
            "macd": float(current_macd),
            "signal": float(current_signal),
            "histogram": float(current_histogram),
            "bullish_cross": bool(bullish_cross),
            "bearish_cross": bool(bearish_cross),
            "histogram_rising": bool(histogram_rising),
            "signal": signal,
            "strength": float(strength)
        }
    except Exception as e:
        return {"signal": "neutral", "strength": 0}

def stochastic_rsi_advanced(df, rsi_length=14, stoch_length=14, k=3, d=3):
    """مؤشر Stochastic RSI المتقدم"""
    try:
        c = df['close'].astype(float)
        
        # RSI Calculation
        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic RSI
        rsi_min = rsi.rolling(stoch_length).min()
        rsi_max = rsi.rolling(stoch_length).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        
        k_percent = stoch_rsi.rolling(k).mean() * 100
        d_percent = k_percent.rolling(d).mean()
        
        current_k = k_percent.iloc[-1]
        current_d = d_percent.iloc[-1]
        
        bullish_cross = current_k > current_d and k_percent.iloc[-2] <= d_percent.iloc[-2]
        bearish_cross = current_k < current_d and k_percent.iloc[-2] >= d_percent.iloc[-2]
        
        oversold = current_k < 20
        overbought = current_k > 80
        
        signal = "buy" if (bullish_cross and oversold) else \
                 "sell" if (bearish_cross and overbought) else "neutral"
        
        strength = abs(current_k - 50) / 50  # 0 to 1 strength
        
        return {
            "k": float(current_k),
            "d": float(current_d),
            "oversold": bool(oversold),
            "overbought": bool(overbought),
            "bullish_cross": bool(bullish_cross),
            "bearish_cross": bool(bearish_cross),
            "signal": signal,
            "strength": float(strength)
        }
    except Exception as e:
        return {"signal": "neutral", "strength": 0}

def market_structure_break_advanced(df, lookback=50):
    """كسر هيكل السوق المتقدم"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        # Higher Highs/Lower Lows analysis
        recent_high = h.tail(lookback).max()
        recent_low = l.tail(lookback).min()
        
        prev_high = h.tail(lookback+10).head(lookback).max()
        prev_low = l.tail(lookback+10).head(lookback).min()
        
        current_price = c.iloc[-1]
        current_volume = v.iloc[-1]
        avg_volume = v.tail(lookback).mean()
        
        break_high = current_price > prev_high
        break_low = current_price < prev_low
        
        volume_confirmation = current_volume > avg_volume * VOLUME_CONFIRMATION_THRESHOLD
        
        if break_high and volume_confirmation:
            strength = (current_price - prev_high) / prev_high * 100
            signal = "strong_buy"
        elif break_low and volume_confirmation:
            strength = (prev_low - current_price) / prev_low * 100
            signal = "strong_sell"
        else:
            strength = 0
            signal = "neutral"
        
        # Trend structure
        highs = h.tail(lookback)
        lows = l.tail(lookback)
        
        higher_highs = all(highs.iloc[i] > highs.iloc[i-1] for i in range(-3, 0))
        lower_lows = all(lows.iloc[i] < lows.iloc[i-1] for i in range(-3, 0))
        
        return {
            "break_high": break_high,
            "break_low": break_low,
            "strength": strength,
            "signal": signal,
            "volume_confirmation": volume_confirmation,
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "resistance": prev_high,
            "support": prev_low
        }
    except Exception as e:
        return {"break_high": False, "break_low": False, "strength": 0, "signal": "neutral"}

def smart_money_flow(df, period=20):
    """تتبع تدفق الأموال الذكية"""
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)
        
        # Money Flow Multiplier
        mf_multiplier = ((c - l) - (h - c)) / (h - l).replace(0, 1e-12)
        mf_volume = mf_multiplier * v
        
        # Money Flow Index
        positive_flow = mf_volume.where(mf_volume > 0, 0).rolling(period).sum()
        negative_flow = abs(mf_volume.where(mf_volume < 0, 0)).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, 1e-12)))
        
        # Accumulation/Distribution Line
        adl = ((c - l) - (h - c)) / (h - l).replace(0, 1e-12) * v
        adl_cumulative = adl.cumsum()
        
        current_mfi = mfi.iloc[-1]
        mfi_signal = "buy" if current_mfi < 20 else "sell" if current_mfi > 80 else "neutral"
        
        adl_trend = "rising" if adl_cumulative.iloc[-1] > adl_cumulative.iloc[-2] else "falling"
        
        return {
            "mfi": float(current_mfi),
            "mfi_signal": mfi_signal,
            "adl_trend": adl_trend,
            "smart_money_bullish": adl_trend == "rising" and current_mfi < 30,
            "smart_money_bearish": adl_trend == "falling" and current_mfi > 70,
            "strength": abs(current_mfi - 50) / 50
        }
    except Exception as e:
        return {"mfi_signal": "neutral", "strength": 0}

def multi_timeframe_analysis(df_dict):
    """تحليل متعدد الأطر الزمنية"""
    signals = {}
    strengths = {}
    
    for tf, df in df_dict.items():
        if len(df) < 50:
            continue
            
        # Basic trend analysis for each timeframe
        current_price = float(df['close'].iloc[-1])
        sma_20 = df['close'].astype(float).rolling(20).mean().iloc[-1]
        sma_50 = df['close'].astype(float).rolling(50).mean().iloc[-1]
        
        trend = "bullish" if (current_price > sma_20 > sma_50) else \
                "bearish" if (current_price < sma_20 < sma_50) else "neutral"
        
        signals[tf] = trend
        strengths[tf] = abs(current_price - sma_50) / sma_50 * 100
    
    # Count confirmations
    bull_count = sum(1 for s in signals.values() if s == "bullish")
    bear_count = sum(1 for s in signals.values() if s == "bearish")
    
    overall_signal = "bullish" if bull_count >= MTF_CONFIRMATION_REQUIRED else \
                     "bearish" if bear_count >= MTF_CONFIRMATION_REQUIRED else "neutral"
    
    confidence = max(bull_count, bear_count) / len(signals) if signals else 0
    
    return {
        "signals": signals,
        "strengths": strengths,
        "overall_signal": overall_signal,
        "confidence": confidence,
        "bull_count": bull_count,
        "bear_count": bear_count
    }

# =================== ULTRA INTELLIGENT COUNCIL SYSTEM ===================
def ultra_intelligent_council(df, mtf_data=None):
    """مجلس التداول الذكي الخارق"""
    try:
        # جمع جميع المؤشرات المتقدمة
        super_trend_data = super_trend(df)
        ichimoku_data = ichimoku_cloud(df)
        bollinger_data = bollinger_bands_advanced(df)
        macd_data = volume_weighted_macd(df)
        stoch_data = stochastic_rsi_advanced(df)
        structure_data = market_structure_break_advanced(df)
        money_flow_data = smart_money_flow(df)
        
        # المؤشرات الأساسية
        ind = compute_indicators(df)
        rsi_ctx = rsi_ma_context(df)
        gz = golden_zone_check(df, ind)
        cd = compute_candles(df)
        
        votes_buy = 0
        votes_sell = 0
        score_buy = 0.0
        score_sell = 0.0
        council_logs = []
        confirmation_signals = []
        
        # === SUPER TREND VOTE ===
        if super_trend_data["signal"] == "buy":
            votes_buy += 2
            score_buy += super_trend_data["strength"] * 2
            council_logs.append(f"🚀 SuperTrend صاعد (قوة: {super_trend_data['strength']:.2f})")
            confirmation_signals.append("SuperTrend")
        
        if super_trend_data["signal"] == "sell":
            votes_sell += 2
            score_sell += super_trend_data["strength"] * 2
            council_logs.append(f"💥 SuperTrend هابط (قوة: {super_trend_data['strength']:.2f})")
            confirmation_signals.append("SuperTrend")
        
        # === ICHIMOKU CLOUD VOTE ===
        if ichimoku_data["signal"] == "buy":
            votes_buy += 2
            score_buy += ichimoku_data["strength"] * 1.5
            council_logs.append(f"☁️ Ichimoku إيجابي (قوة: {ichimoku_data['strength']:.2f}%)")
            confirmation_signals.append("Ichimoku")
        
        if ichimoku_data["signal"] == "sell":
            votes_sell += 2
            score_sell += ichimoku_data["strength"] * 1.5
            council_logs.append(f"☁️ Ichimoku سلبي (قوة: {ichimoku_data['strength']:.2f}%)")
            confirmation_signals.append("Ichimoku")
        
        # === BOLLINGER BANDS VOTE ===
        if bollinger_data["signal"] == "buy" and not bollinger_data["squeeze"]:
            votes_buy += 2
            score_buy += bollinger_data["strength"] * 2
            council_logs.append(f"📊 Bollinger شراء من المنطقة السفلية")
            confirmation_signals.append("Bollinger")
        
        if bollinger_data["signal"] == "sell" and not bollinger_data["squeeze"]:
            votes_sell += 2
            score_sell += bollinger_data["strength"] * 2
            council_logs.append(f"📊 Bollinger بيع من المنطقة العلوية")
            confirmation_signals.append("Bollinger")
        
        # === MACD VOTE ===
        if macd_data["signal"] == "buy":
            votes_buy += 2
            score_buy += macd_data["strength"] * 1.5
            council_logs.append(f"📈 MACD إيجابي (تقاطع صاعد)")
            confirmation_signals.append("MACD")
        
        if macd_data["signal"] == "sell":
            votes_sell += 2
            score_sell += macd_data["strength"] * 1.5
            council_logs.append(f"📉 MACD سلبي (تقاطع هابط)")
            confirmation_signals.append("MACD")
        
        # === STOCHASTIC RSI VOTE ===
        if stoch_data["signal"] == "buy":
            votes_buy += 1
            score_buy += stoch_data["strength"] * 1.2
            council_logs.append(f"🎯 Stoch RSI من منطقة التشبع بيع")
            confirmation_signals.append("StochRSI")
        
        if stoch_data["signal"] == "sell":
            votes_sell += 1
            score_sell += stoch_data["strength"] * 1.2
            council_logs.append(f"🎯 Stoch RSI من منطقة التشبع شراء")
            confirmation_signals.append("StochRSI")
        
        # === MARKET STRUCTURE VOTE ===
        if structure_data["signal"] == "strong_buy":
            votes_buy += 3
            score_buy += structure_data["strength"] * 0.5
            council_logs.append(f"🔄 كسر مقاومة قوي ({structure_data['strength']:.2f}%)")
            confirmation_signals.append("Structure")
        
        if structure_data["signal"] == "strong_sell":
            votes_sell += 3
            score_sell += structure_data["strength"] * 0.5
            council_logs.append(f"🔄 كسر دعم قوي ({structure_data['strength']:.2f}%)")
            confirmation_signals.append("Structure")
        
        # === SMART MONEY FLOW VOTE ===
        if money_flow_data["smart_money_bullish"]:
            votes_buy += 2
            score_buy += money_flow_data["strength"] * 1.8
            council_logs.append("💰 الأموال الذكية تشتري")
            confirmation_signals.append("SmartMoney")
        
        if money_flow_data["smart_money_bearish"]:
            votes_sell += 2
            score_sell += money_flow_data["strength"] * 1.8
            council_logs.append("💰 الأموال الذكية تبيع")
            confirmation_signals.append("SmartMoney")
        
        # === MULTI-TIMEFRAME CONFIRMATION ===
        if mtf_data and mtf_data["overall_signal"] == "bullish":
            votes_buy += 2
            score_buy += mtf_data["confidence"] * 2
            council_logs.append(f"⏰ تأكيد متعدد الأطر ({mtf_data['bull_count']}/{len(mtf_data['signals'])})")
            confirmation_signals.append("MultiTF")
        
        if mtf_data and mtf_data["overall_signal"] == "bearish":
            votes_sell += 2
            score_sell += mtf_data["confidence"] * 2
            council_logs.append(f"⏰ تأكيد متعدد الأطر ({mtf_data['bear_count']}/{len(mtf_data['signals'])})")
            confirmation_signals.append("MultiTF")
        
        # === TREND STRENGTH BOOST ===
        if ind["adx"] > TREND_STRENGTH_THRESHOLD:
            if votes_buy > votes_sell:
                score_buy *= 1.3
                council_logs.append(f"🔥 تعزيز قوة الترند الصاعد (ADX: {ind['adx']:.1f})")
            elif votes_sell > votes_buy:
                score_sell *= 1.3
                council_logs.append(f"🔥 تعزيز قوة الترند الهابط (ADX: {ind['adx']:.1f})")
        
        # === VOLUME CONFIRMATION BOOST ===
        current_volume = float(df['volume'].iloc[-1])
        avg_volume = float(df['volume'].tail(20).mean())
        if current_volume > avg_volume * VOLUME_CONFIRMATION_THRESHOLD:
            if votes_buy > votes_sell:
                score_buy *= 1.2
                council_logs.append("📈 حجم تداول عالي يدعم الشراء")
            elif votes_sell > votes_buy:
                score_sell *= 1.2
                council_logs.append("📉 حجم تداول عالي يدعم البيع")
        
        # === GOLDEN ZONE ULTRA BOOST ===
        if gz and gz.get("ok") and gz.get("score", 0) >= GOLDEN_ENTRY_SCORE:
            if gz['zone']['type'] == 'golden_bottom':
                votes_buy += 3
                score_buy += 2.5
                council_logs.append(f"🏆 قاع ذهبي فائق (قوة: {gz['score']:.1f})")
                confirmation_signals.append("GoldenZone")
            elif gz['zone']['type'] == 'golden_top':
                votes_sell += 3
                score_sell += 2.5
                council_logs.append(f"🏆 قمة ذهبية فائقة (قوة: {gz['score']:.1f})")
                confirmation_signals.append("GoldenZone")
        
        return {
            "votes_buy": votes_buy,
            "votes_sell": votes_sell,
            "score_buy": round(score_buy, 2),
            "score_sell": round(score_sell, 2),
            "logs": council_logs,
            "confirmation_signals": confirmation_signals,
            "indicators": {
                "super_trend": super_trend_data,
                "ichimoku": ichimoku_data,
                "bollinger": bollinger_data,
                "macd": macd_data,
                "stoch_rsi": stoch_data,
                "market_structure": structure_data,
                "money_flow": money_flow_data,
                "basic": ind,
                "rsi_context": rsi_ctx,
                "golden_zone": gz,
                "candles": cd
            },
            "mtf_analysis": mtf_data
        }
        
    except Exception as e:
        print(f"❌ مجلس الذكاء الخارق خطأ: {e}")
        return {
            "votes_buy": 0,
            "votes_sell": 0, 
            "score_buy": 0,
            "score_sell": 0,
            "logs": [],
            "confirmation_signals": [],
            "indicators": {},
            "mtf_analysis": None
        }

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
        log_w(f"state save failed: {e}")
    return {}

# =================== EXCHANGE SETUP ===================
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

# =================== CORE FUNCTIONS ===================
def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    
    def wilder_ema(s: pd.Series, n: int): 
        return s.ewm(alpha=1/n, adjust=False).mean()
    
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

def rsi_ma_context(df):
    RSI_MA_LEN = 9
    RSI_NEUTRAL_BAND = (45, 55)
    RSI_TREND_PERSIST = 3
    
    if len(df) < max(RSI_MA_LEN, 14):
        return {"rsi": 50, "rsi_ma": 50, "cross": "none", "trendZ": "none", "in_chop": True}
    
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

def golden_zone_check(df, ind=None, side_hint=None):
    FIB_LOW, FIB_HIGH = 0.618, 0.786
    MIN_WICK_PCT = 0.35
    VOL_MA_LEN = 20
    RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
    MIN_DISP = 0.8
    GZ_MIN_SCORE = 6.0
    GZ_REQ_ADX = 20
    
    if len(df) < 60:
        return {"ok": False, "score": 0.0, "zone": None, "reasons": ["short_df"]}
    
    try:
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        o = df['open'].astype(float)
        v = df['volume'].astype(float)
        
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
        return {"ok": False, "score": 0.0, "zone": None, "reasons": [f"error: {str(e)}"]}

def compute_candles(df):
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

# =================== EXECUTION SYSTEM ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "tp_levels_hit": [],
    "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
}
compound_pnl = 0.0
wait_for_next_signal_side = None

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

def price_now():
    try:
        t = ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def fetch_ohlcv(limit=600):
    rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def fetch_multi_timeframe():
    """جلب بيانات متعددة الأطر الزمنية"""
    mtf_data = {}
    for tf in MTF_TIMEFRAMES:
        try:
            rows = ex.fetch_ohlcv(SYMBOL, timeframe=tf, limit=100, params={"type":"swap"})
            mtf_data[tf] = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
        except Exception as e:
            log_w(f"Failed to fetch {tf} data: {e}")
    return mtf_data

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def setup_trade_management(mode):
    if mode == "scalp":
        return {
            "tp_targets": SCALP_TP_TARGETS,
            "tp_fractions": SCALP_TP_FRACTIONS,
            "be_activate_pct": BREAKEVEN_AFTER / 100.0,
            "trail_activate_pct": TRAIL_ACTIVATE_PCT / 100.0,
            "atr_trail_mult": ATR_TRAIL_MULT,
            "close_aggression": "high",
            "max_tp_levels": len(SCALP_TP_TARGETS)
        }
    else:
        return {
            "tp_targets": TREND_TP_TARGETS,
            "tp_fractions": TREND_TP_FRACTIONS,
            "be_activate_pct": BREAKEVEN_AFTER / 100.0,
            "trail_activate_pct": TRAIL_ACTIVATE_PCT / 100.0,
            "atr_trail_mult": ATR_TRAIL_MULT,
            "close_aggression": "medium",
            "max_tp_levels": len(TREND_TP_TARGETS)
        }

def manage_take_profits(state, current_price, pnl_pct, management_config, mode):
    if state["qty"] <= 0:
        return
    
    tp_targets = management_config["tp_targets"]
    tp_fractions = management_config["tp_fractions"]
    
    if mode == "scalp":
        for i, (tp_pct, frac) in enumerate(zip(tp_targets, tp_fractions)):
            if not state.get(f"tp{i+1}_done") and pnl_pct >= tp_pct/100.0:
                close_qty = safe_qty(state["qty"] * frac)
                if close_qty > 0:
                    close_side = "sell" if state["side"] == "long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            log_g(f"✅ SCALP TP{i+1} HIT: closed {frac*100}% at {tp_pct}%")
                        except Exception as e:
                            log_e(f"❌ Scalp TP{i+1} close failed: {e}")
                    state["qty"] = safe_qty(state["qty"] - close_qty)
                    state[f"tp{i+1}_done"] = True
                    state["profit_targets_achieved"] += 1
                    log_i(f"💰 SCALP SUCCESS: Taken {tp_pct}% profit!")
    
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

def smart_entry_system(council_data):
    """نظام دخول ذكي فائق الدقة"""
    indicators = council_data["indicators"]
    confirmation_signals = council_data["confirmation_signals"]
    
    required_confirmations = 4  # يحتاج 4 إشارات تأكيد على الأقل
    
    if len(confirmation_signals) >= required_confirmations:
        if council_data["score_buy"] > council_data["score_sell"] + 2:
            return "buy"
        elif council_data["score_sell"] > council_data["score_buy"] + 2:
            return "sell"
    
    return None

def execute_trade_decision(side, price, qty, council_data):
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}")
        return True
    
    if qty <= 0:
        log_e("❌ كمية غير صالحة للتنفيذ")
        return False

    print(f"🎯 EXECUTE ULTRA: {side.upper()} {qty:.4f} @ {price:.6f} | "
          f"score={council_data['score_buy']:.1f}/{council_data['score_sell']:.1f} | "
          f"confirmations={len(council_data['confirmation_signals'])}", flush=True)

    try:
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ EXECUTED ULTRA: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    except Exception as e:
        log_e(f"❌ EXECUTION FAILED: {e}")
        return False

def open_market_ultra(side, qty, price, council_data):
    if qty <= 0: 
        log_e("skip open (qty<=0)")
        return False
    
    # تحديد نمط التداول بناء على قوة الإشارة
    signal_strength = max(council_data["score_buy"], council_data["score_sell"])
    mode = "trend" if signal_strength > 12 else "scalp"
    
    management_config = setup_trade_management(mode)
    
    success = execute_trade_decision(side, price, qty, council_data)
    
    if success:
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
            "tp_levels_hit": tp_levels_hit,
            "highest_profit_pct": 0.0, 
            "profit_targets_achieved": 0,
            "mode": mode,
            "management": management_config,
            "council_snapshot": council_data
        })
        
        save_state({
            "in_position": True,
            "side": "LONG" if side.upper().startswith("B") else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "management": management_config,
            "council_snapshot": council_data,
            "opened_at": int(time.time()),
            "tp_levels_hit": tp_levels_hit
        })
        
        log_g(f"✅ ULTRA POSITION OPENED: {side.upper()} | mode={mode} | strength={signal_strength:.1f}")
        return True
    
    return False

def manage_after_entry_enhanced(df, council_data):
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = price_now()
    if not px:
        return

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

    # تحليل الخروج الذكي
    exit_analysis = smart_exit_analysis(df, council_data, px, pnl_pct, side, entry)
    
    if exit_analysis["action"] == "close":
        log_w(f"🚨 ULTRA EXIT: {exit_analysis['reason']}")
        close_market_strict(f"ultra_exit_{exit_analysis['reason']}")
        return
    elif exit_analysis["action"] == "partial":
        partial_qty = safe_qty(qty * exit_analysis.get("qty_pct", 0.3))
        if partial_qty > 0 and not STATE.get("partial_taken"):
            close_side = "sell" if side == "long" else "buy"
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                try:
                    params = exchange_specific_params(close_side, is_close=True)
                    ex.create_order(SYMBOL, "market", close_side, partial_qty, None, params)
                    log_g(f"✅ ULTRA PARTIAL CLOSE: {partial_qty:.4f}")
                    STATE["partial_taken"] = True
                    STATE["qty"] = safe_qty(qty - partial_qty)
                except Exception as e:
                    log_e(f"❌ Partial close failed: {e}")

def smart_exit_analysis(df, council_data, current_price, pnl_pct, side, entry):
    indicators = council_data["indicators"]
    
    # تحليل تناقض الإشارات
    current_council = ultra_intelligent_council(df)
    current_signal = "buy" if current_council["score_buy"] > current_council["score_sell"] else "sell"
    
    # إذا انعكس اتجاه المجلس
    if (side == "long" and current_signal == "sell") or (side == "short" and current_signal == "buy"):
        return {"action": "close", "reason": "council_reversal"}
    
    # خروج عند تحقيق ربح ممتاز
    if pnl_pct >= (8 if STATE.get("mode") == "trend" else 4):
        return {"action": "close", "reason": "excellent_profit"}
    
    # خروج عند ضعف الزخم
    if indicators["basic"]["adx"] < 15 and abs(pnl_pct) > 1:
        return {"action": "close", "reason": "momentum_loss"}
    
    return {"action": "hold", "reason": "continue_trend"}

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    
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

    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    
    while attempts < 6:
        try:
            if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                params = exchange_specific_params(side_to_close, is_close=True)
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            
            time.sleep(2.0)
            
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                log_i(f"STRICT CLOSE {side} reason={reason} pnl={pnl:.4f} total={compound_pnl:.4f}")
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            log_w(f"strict close retry {attempts}/6 — residual={left_qty:.4f}")
            time.sleep(2.0)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(2.0)
    log_e(f"STRICT CLOSE FAILED after 6 attempts — last error: {last_error}")

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "tp_levels_hit": [],
        "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
    })
    save_state({"in_position": False, "position_qty": 0})
    
    wait_for_next_signal_side = "sell" if prev_side=="long" else ("buy" if prev_side=="short" else None)
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

# =================== ULTRA TRADING LOOP ===================
def ultra_trading_loop():
    global wait_for_next_signal_side
    loop_i = 0
    
    while True:
        try:
            # جلب البيانات
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                time.sleep(5)
                continue
            
            # تحليل متعدد الأطر الزمنية
            mtf_data = None
            if MTF_ANALYSIS:
                mtf_df_dict = fetch_multi_timeframe()
                if mtf_df_dict:
                    mtf_data = multi_timeframe_analysis(mtf_df_dict)
            
            # مجلس الذكاء الخارق
            council_data = ultra_intelligent_council(df, mtf_data)
            
            # تحديث حالة المركز
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
                manage_after_entry_enhanced(df, council_data)
            
            # نظام الدخول الذكي
            if not STATE["open"]:
                entry_signal = smart_entry_system(council_data)
                
                if entry_signal and (wait_for_next_signal_side is None or entry_signal == wait_for_next_signal_side):
                    qty = compute_size(bal, px)
                    if qty > 0:
                        ok = open_market_ultra(entry_signal, qty, px, council_data)
                        if ok:
                            wait_for_next_signal_side = None
                            
                            # عرض تفاصيل قرار المجلس
                            log_i(f"🎯 ULTRA COUNCIL DECISION: {entry_signal.upper()}")
                            log_i(f"   📊 النقاط: {council_data['votes_buy']}/{council_data['votes_sell']}")
                            log_i(f"   💪 القوة: {council_data['score_buy']:.1f}/{council_data['score_sell']:.1f}")
                            log_i(f"   ✅ التأكيدات: {len(council_data['confirmation_signals'])}")
                            for log_msg in council_data.get("logs", []):
                                log_i(f"   - {log_msg}")
            
            # تحديث واجهة المستخدم
            if LOG_ADDONS:
                print(f"\n🧠 ULTRA COUNCIL SNAPSHOT:", flush=True)
                print(f"   📈 الإشارة: {'شراء' if council_data['score_buy'] > council_data['score_sell'] else 'بيع' if council_data['score_sell'] > council_data['score_buy'] else 'محايد'}", flush=True)
                print(f"   💪 القوة: {council_data['score_buy']:.1f} / {council_data['score_sell']:.1f}", flush=True)
                print(f"   ✅ التأكيدات: {len(council_data['confirmation_signals'])}", flush=True)
                print(f"   📊 المؤشرات: SuperTrend, Ichimoku, Bollinger, MACD, StochRSI, SmartMoney", flush=True)
                
                if STATE["open"]:
                    print(f"   🎯 المركز: {STATE['side']} | الربح: {STATE['pnl']:.2f}%", flush=True)
                else:
                    print(f"   ⚪ لا توجد صفقات مفتوحة", flush=True)
                
                if compound_pnl != 0:
                    print(f"   💰 إجمالي الأرباح: {compound_pnl:.4f} USDT", flush=True)
            
            loop_i += 1
            time.sleep(5)
            
        except Exception as e:
            log_e(f"ULTRA LOOP ERROR: {e}\n{traceback.format_exc()}")
            time.sleep(10)

# =================== API & HEALTH ===================
app = Flask(__name__)

@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ SUI ULTRA PRO TRADER — {EXCHANGE_NAME.upper()} — {SYMBOL} {INTERVAL} — {mode}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL, 
        "interval": INTERVAL, 
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, 
        "risk_alloc": RISK_ALLOC, 
        "price": price_now(),
        "state": STATE, 
        "compound_pnl": compound_pnl,
        "bot_version": BOT_VERSION
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, 
        "exchange": EXCHANGE_NAME, 
        "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], 
        "side": STATE["side"], 
        "qty": STATE["qty"],
        "compound_pnl": compound_pnl, 
        "timestamp": datetime.utcnow().isoformat(),
        "ultra_council": "ACTIVE"
    }), 200

def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        log_w("keepalive disabled (SELF_URL not set)")
        return
    
    import requests
    sess = requests.Session()
    sess.headers.update({"User-Agent": "ultra-pro-trader/keepalive"})
    log_i(f"KEEPALIVE every 50s → {url}")
    
    while True:
        try: 
            sess.get(url, timeout=8)
        except Exception: 
            pass
        time.sleep(50)

# =================== BOOT SYSTEM ===================
if __name__ == "__main__":
    log_banner("SUI ULTRA PRO TRADER - INTELLIGENT COUNCIL SYSTEM")
    
    print(colored(f"🚀 EXCHANGE: {EXCHANGE_NAME.upper()} • SYMBOL: {SYMBOL} • TIMEFRAME: {INTERVAL}", "yellow"))
    print(colored(f"⚡ RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • ULTRA_COUNCIL=ENABLED", "yellow"))
    print(colored(f"💰 PROFIT TARGETS: Scalp{SCALP_TP_TARGETS} | Trend{TREND_TP_TARGETS}", "yellow"))
    print(colored(f"🧠 ADVANCED INDICATORS: SuperTrend + Ichimoku + Bollinger + MACD + SmartMoney", "yellow"))
    print(colored(f"⏰ MULTI-TIMEFRAME: {MTF_TIMEFRAMES} ({MTF_CONFIRMATION_REQUIRED} confirmations required)", "yellow"))
    print(colored(f"🎯 EXECUTION: {'ACTIVE' if EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}", "yellow"))
    
    # تحميل الحالة السابقة
    state = load_state() or {}
    state.setdefault("in_position", False)
    
    if RESUME_ON_RESTART and state.get("in_position"):
        log_i("🔄 استئناف المركز المفتوح السابق")
        STATE.update({
            "open": True,
            "side": state.get("side", "").lower(),
            "entry": state.get("entry_price"),
            "qty": state.get("position_qty", 0),
            "mode": state.get("mode", "trend")
        })
    
    # بدء الأنظمة
    import threading
    threading.Thread(target=ultra_trading_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
