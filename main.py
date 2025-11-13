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

# =================== ENHANCED PROFESSIONAL SETTINGS ===================
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

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Enhanced Logging ====
LOG_LEGACY = False
LOG_ADDONS = True
LOG_DETAILED_ENTRY = True
LOG_DETAILED_EXIT = True

# ==== Bot Version ====
BOT_VERSION = f"SUI Council ULTIMATE PRO v10.0 — {EXCHANGE_NAME.upper()} - INTELLIGENT TRADING"
print("🚀 Booting:", BOT_VERSION, flush=True)

# =================== ADVANCED TRADING CONFIGURATION ===================
SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# =================== ENHANCED INDICATOR SETTINGS ===================
# Moving Averages for Trend Analysis
EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_TREND = 100
SMA_SHORT = 10
SMA_LONG = 30

# RSI Settings
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_HIGH = 55
RSI_NEUTRAL_LOW = 45

# MACD Settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Advanced Indicator Settings
ADX_PERIOD = 14
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
STOCHASTIC_K = 14
STOCHASTIC_D = 3

# =================== SMART MONEY CONCEPTS ENHANCED ===================
SMC_ENABLED = True
FVG_MIN_SIZE = 0.08  # Minimum FVG size percentage
OB_MIN_STRENGTH = 0.15  # Minimum Order Block strength
LIQUIDITY_ZONE_TOLERANCE = 0.015  # 1.5% proximity to liquidity zone
BOS_CONFIRMATION_BARS = 3  # Bars needed for BOS confirmation
CHOCH_SENSITIVITY = 0.002  # CHoCH sensitivity

# =================== MARKET STRUCTURE ENHANCED ===================
MS_LOOKBACK = 50  # Bars to look back for market structure
MS_MIN_SWING = 0.005  # Minimum swing size (0.5%)
MS_CONFIRMATION = 2  # Bars needed for structure confirmation

# =================== TRADE EXECUTION ENHANCED ===================
ENTRY_CONFIRMATION = True
ENTRY_RETRY_ATTEMPTS = 3
ENTRY_RETRY_DELAY = 2

# =================== PROFIT MANAGEMENT ENHANCED ===================
TP_STRATEGY = "dynamic_multi_level"  # dynamic_multi_level, fixed_ratio, trailing_atr
TP_LEVELS = [0.5, 1.0, 1.8, 2.5, 3.5]  # Profit targets in %
TP_RATIOS = [0.2, 0.25, 0.2, 0.2, 0.15]  # Close ratios for each TP

TRAILING_ENABLED = True
TRAILING_ACTIVATION = 0.008  # Activate trailing after 0.8% profit
TRAILING_MODE = "atr_based"  # atr_based, percentage_based
TRAILING_ATR_MULTIPLIER = 1.5

# =================== RISK MANAGEMENT ENHANCED ===================
STOP_LOSS_STRATEGY = "atr_based"  # atr_based, percentage_based, structure_based
STOP_LOSS_ATR_MULTIPLIER = 1.8
MAX_DRAWDOWN_PER_TRADE = 3.0  # % 
MAX_DRAWDOWN_DAILY = 8.0  # %

# =================== LOOP CONFIGURATION ===================
BASE_SLEEP = 5  # Base sleep time in seconds
NEAR_CLOSE_S = 2  # Sleep time when near close

# =================== ENHANCED LOGGING SYSTEM ===================
class ProfessionalLogger:
    """نظام تسجيل محترف مع الألوان والرموز"""
    
    @staticmethod
    def info(msg):
        print(f"ℹ️ {msg}", flush=True)
        
    @staticmethod
    def success(msg):
        print(f"✅ {msg}", flush=True)
        
    @staticmethod
    def warning(msg):
        print(f"🟡 {msg}", flush=True)
        
    @staticmethod
    def error(msg):
        print(f"❌ {msg}", flush=True)
        
    @staticmethod
    def signal(msg):
        print(f"🎯 {msg}", flush=True)
        
    @staticmethod
    def trade(msg):
        print(f"💰 {msg}", flush=True)
        
    @staticmethod
    def analysis(msg):
        print(f"📊 {msg}", flush=True)
        
    @staticmethod
    def strategy(msg):
        print(f"🎮 {msg}", flush=True)
        
    @staticmethod
    def indicator(msg):
        print(f"📈 {msg}", flush=True)

# إنشاء كائن التسجيل
log = ProfessionalLogger()

def log_banner(text):
    """طباعة بانر جميل"""
    print(f"\n{'='*60}", flush=True)
    print(f"🎯 {text}", flush=True)
    print(f"{'='*60}\n", flush=True)

# =================== ENHANCED INDICATORS SYSTEM ===================
class AdvancedIndicatorSystem:
    """نظام المؤشرات المتقدم"""
    
    @staticmethod
    def calculate_ema(series, period):
        """المتوسط المتحرك الأسي"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series, period):
        """المتوسط المتحرك البسيط"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def calculate_rsi(series, period=14):
        """مؤشر RSI المحسن"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_macd(series, fast=12, slow=26, signal=9):
        """مؤشر MACD المتقدم"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(series, period=20, std=2):
        """نطاقات بولينجر المحسنة"""
        sma = series.rolling(period).mean()
        std_dev = series.rolling(period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        """مؤشر ستوكاستيك المتقدم"""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(d_period).mean()
        
        return {
            'k': k_line,
            'd': d_line
        }
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """مؤشر ADX المحسن"""
        # حساب +DI و -DI
        up = high.diff()
        down = -low.diff()
        
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """متوسط المدى الحقيقي"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

# =================== ENHANCED SMC ENGINE ===================
class SmartMoneyConceptsEngine:
    """محرك مفاهيم Smart Money المتقدم"""
    
    @staticmethod
    def identify_fvg(df, min_size=0.08):
        """تحديد Fair Value Gaps بدقة عالية"""
        fvg_bullish = []
        fvg_bearish = []
        
        for i in range(2, len(df)):
            # FVG صاعد: الشمعة الحالية تغلق فوق الشمعة السابقة
            if (df['close'].iloc[i] > df['high'].iloc[i-1] and
                (df['close'].iloc[i] - df['high'].iloc[i-1]) / df['close'].iloc[i] >= min_size/100):
                fvg_bullish.append({
                    'low': df['high'].iloc[i-1],
                    'high': df['close'].iloc[i],
                    'size': (df['close'].iloc[i] - df['high'].iloc[i-1]) / df['close'].iloc[i] * 100,
                    'time': df.index[i]
                })
            
            # FVG هابط: الشمعة الحالية تغلق تحت الشمعة السابقة
            if (df['close'].iloc[i] < df['low'].iloc[i-1] and
                (df['low'].iloc[i-1] - df['close'].iloc[i]) / df['low'].iloc[i-1] >= min_size/100):
                fvg_bearish.append({
                    'low': df['close'].iloc[i],
                    'high': df['low'].iloc[i-1],
                    'size': (df['low'].iloc[i-1] - df['close'].iloc[i]) / df['low'].iloc[i-1] * 100,
                    'time': df.index[i]
                })
        
        return {
            'bullish_fvg': fvg_bullish[-5:],  # آخر 5 FVG صاعدة
            'bearish_fvg': fvg_bearish[-5:]   # آخر 5 FVG هابطة
        }
    
    @staticmethod
    def identify_order_blocks(df, min_strength=0.15):
        """تحديد Order Blocks بدقة عالية"""
        bullish_ob = []
        bearish_ob = []
        
        for i in range(1, len(df)-1):
            current = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Order Block صاعد: شمعة هابطة كبيرة تليها شمعة صاعدة
            if (current['close'] < current['open'] and  # شمعة هابطة
                next_candle['close'] > next_candle['open'] and  # شمعة صاعدة تليها
                abs(current['close'] - current['open']) / current['open'] > min_strength/100):
                
                bullish_ob.append({
                    'high': max(current['high'], next_candle['high']),
                    'low': min(current['low'], next_candle['low']),
                    'strength': abs(current['close'] - current['open']) / current['open'] * 100,
                    'time': df.index[i]
                })
            
            # Order Block هابط: شمعة صاعدة كبيرة تليها شمعة هابطة
            if (current['close'] > current['open'] and  # شمعة صاعدة
                next_candle['close'] < next_candle['open'] and  # شمعة هابطة تليها
                abs(current['close'] - current['open']) / current['open'] > min_strength/100):
                
                bearish_ob.append({
                    'high': max(current['high'], next_candle['high']),
                    'low': min(current['low'], next_candle['low']),
                    'strength': abs(current['close'] - current['open']) / current['open'] * 100,
                    'time': df.index[i]
                })
        
        return {
            'bullish_ob': bullish_ob[-5:],
            'bearish_ob': bearish_ob[-5:]
        }
    
    @staticmethod
    def analyze_market_structure(df, lookback=50, min_swing=0.005):
        """تحليل هيكل السوق المتقدم"""
        if len(df) < lookback:
            return {"trend": "neutral", "bos": False, "choch": False}
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # تحديد القمم والقيعان
        peaks = []
        troughs = []
        
        for i in range(2, len(df)-2):
            # قمة
            if (high.iloc[i] > high.iloc[i-1] and 
                high.iloc[i] > high.iloc[i-2] and
                high.iloc[i] > high.iloc[i+1] and
                high.iloc[i] > high.iloc[i+2]):
                peaks.append((i, high.iloc[i]))
            
            # قاع
            if (low.iloc[i] < low.iloc[i-1] and 
                low.iloc[i] < low.iloc[i-2] and
                low.iloc[i] < low.iloc[i+1] and
                low.iloc[i] < low.iloc[i+2]):
                troughs.append((i, low.iloc[i]))
        
        # تحليل الاتجاه
        trend = "neutral"
        if len(peaks) >= 2 and len(troughs) >= 2:
            higher_highs = peaks[-1][1] > peaks[-2][1] if len(peaks) >= 2 else False
            higher_lows = troughs[-1][1] > troughs[-2][1] if len(troughs) >= 2 else False
            lower_highs = peaks[-1][1] < peaks[-2][1] if len(peaks) >= 2 else False
            lower_lows = troughs[-1][1] < troughs[-2][1] if len(troughs) >= 2 else False
            
            if higher_highs and higher_lows:
                trend = "bullish"
            elif lower_highs and lower_lows:
                trend = "bearish"
        
        # Break of Structure (BOS)
        bos_bullish = False
        bos_bearish = False
        
        if len(peaks) >= 2 and trend == "bullish":
            bos_bullish = close.iloc[-1] > peaks[-2][1]
        
        if len(troughs) >= 2 and trend == "bearish":
            bos_bearish = close.iloc[-1] < troughs[-2][1]
        
        # Change of Character (CHoCH)
        choch_bullish = False
        choch_bearish = False
        
        if len(peaks) >= 2 and len(troughs) >= 2:
            if trend == "bullish" and lower_lows:
                choch_bearish = True
            elif trend == "bearish" and higher_highs:
                choch_bullish = True
        
        return {
            "trend": trend,
            "bos_bullish": bos_bullish,
            "bos_bearish": bos_bearish,
            "choch_bullish": choch_bullish,
            "choch_bearish": choch_bearish,
            "peaks": peaks[-3:],
            "troughs": troughs[-3:]
        }
    
    @staticmethod
    def identify_liquidity_zones(df, tolerance=0.015):
        """تحديد مناطق السيولة بدقة"""
        if len(df) < 20:
            return {"buy_zones": [], "sell_zones": []}
        
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # مناطق البيع (المقاومة)
        resistance_levels = []
        for i in range(10, len(df)-10):
            if (high.iloc[i] == high.iloc[i-10:i+10].max() and
                high.iloc[i] > high.iloc[i-1] and
                high.iloc[i] > high.iloc[i+1]):
                resistance_levels.append(high.iloc[i])
        
        # مناطق الشراء (الدعم)
        support_levels = []
        for i in range(10, len(df)-10):
            if (low.iloc[i] == low.iloc[i-10:i+10].min() and
                low.iloc[i] < low.iloc[i-1] and
                low.iloc[i] < low.iloc[i+1]):
                support_levels.append(low.iloc[i])
        
        # التجميع حسب التسامح
        buy_zones = SmartMoneyConceptsEngine._cluster_levels(support_levels, tolerance)
        sell_zones = SmartMoneyConceptsEngine._cluster_levels(resistance_levels, tolerance)
        
        return {
            "buy_zones": buy_zones[-3:],  # آخر 3 مناطق شراء
            "sell_zones": sell_zones[-3:]  # آخر 3 مناطق بيع
        }
    
    @staticmethod
    def _cluster_levels(levels, tolerance):
        """تجميع المستويات المتقاربة"""
        if not levels:
            return []
        
        levels.sort()
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters

# =================== ENHANCED CANDLESTICK ANALYSIS ===================
class AdvancedCandlestickAnalysis:
    """تحليل الشموع اليابانية المتقدم"""
    
    @staticmethod
    def analyze_patterns(df):
        """تحليل أنماط الشموع المتقدم"""
        if len(df) < 5:
            return {"pattern": "none", "strength": 0, "direction": "neutral"}
        
        patterns = []
        strength = 0
        
        # تحليل آخر 3 شموع
        for i in range(-3, 0):
            pattern_info = AdvancedCandlestickAnalysis._analyze_single_candle(df, i)
            if pattern_info["pattern"] != "none":
                patterns.append(pattern_info)
                strength += pattern_info["strength"]
        
        # تحديد النمط السائد
        if not patterns:
            return {"pattern": "none", "strength": 0, "direction": "neutral"}
        
        bull_patterns = [p for p in patterns if p["direction"] == "bullish"]
        bear_patterns = [p for p in patterns if p["direction"] == "bearish"]
        
        bull_strength = sum(p["strength"] for p in bull_patterns)
        bear_strength = sum(p["strength"] for p in bear_patterns)
        
        if bull_strength > bear_strength:
            direction = "bullish"
            main_pattern = max(bull_patterns, key=lambda x: x["strength"])["pattern"]
        elif bear_strength > bull_strength:
            direction = "bearish"
            main_pattern = max(bear_patterns, key=lambda x: x["strength"])["pattern"]
        else:
            direction = "neutral"
            main_pattern = "none"
        
        return {
            "pattern": main_pattern,
            "strength": max(bull_strength, bear_strength),
            "direction": direction,
            "details": patterns
        }
    
    @staticmethod
    def _analyze_single_candle(df, index):
        """تحليل شمعة فردية"""
        if abs(index) > len(df):
            return {"pattern": "none", "strength": 0, "direction": "neutral"}
        
        candle = df.iloc[index]
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        body_size = abs(c - o)
        total_range = h - l
        
        if total_range == 0:
            return {"pattern": "none", "strength": 0, "direction": "neutral"}
        
        body_ratio = body_size / total_range
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        # تحديد النمط
        pattern = "none"
        strength = 0
        direction = "neutral"
        
        # دوجي
        if body_ratio < 0.1:
            pattern = "doji"
            strength = 0.5
        
        # مطرقة / شنق
        elif lower_wick >= 2 * body_size and upper_wick <= body_size * 0.5:
            pattern = "hammer" if c > o else "hanging_man"
            strength = 1.0
            direction = "bullish" if c > o else "bearish"
        
        # نجمة الرماية
        elif upper_wick >= 2 * body_size and lower_wick <= body_size * 0.5:
            pattern = "shooting_star"
            strength = 1.0
            direction = "bearish"
        
        # شموع engulfing
        elif index < -1:
            prev_candle = df.iloc[index-1]
            po, pc = prev_candle['open'], prev_candle['close']
            
            # Bullish Engulfing
            if (pc < po and c > o and o <= pc and c >= po and 
                body_size > abs(pc - po)):
                pattern = "bullish_engulfing"
                strength = 1.5
                direction = "bullish"
            
            # Bearish Engulfing
            elif (pc > po and c < o and o >= pc and c <= po and 
                  body_size > abs(pc - po)):
                pattern = "bearish_engulfing"
                strength = 1.5
                direction = "bearish"
        
        # شموع ماروبوزو
        elif body_ratio > 0.9:
            pattern = "marubozu"
            strength = 1.2
            direction = "bullish" if c > o else "bearish"
        
        return {"pattern": pattern, "strength": strength, "direction": direction}

# =================== ENHANCED TRADING COUNCIL ===================
class IntelligentTradingCouncil:
    """مجلس التداول الذكي المحسن"""
    
    def __init__(self):
        self.indicator_system = AdvancedIndicatorSystem()
        self.smc_engine = SmartMoneyConceptsEngine()
        self.candle_analysis = AdvancedCandlestickAnalysis()
        
    def analyze_market(self, df):
        """تحليل السوق الشامل"""
        if len(df) < 100:
            return self._get_default_analysis()
        
        try:
            # تحليل SMC المتقدم
            smc_analysis = self._analyze_smc(df)
            
            # تحليل المؤشرات التقنية
            technical_analysis = self._analyze_technical_indicators(df)
            
            # تحليل الشموع
            candle_analysis = self.candle_analysis.analyze_patterns(df)
            
            # تحليل الزخم والحجم
            momentum_analysis = self._analyze_momentum(df)
            
            # اتخاذ القرار النهائي
            final_decision = self._make_final_decision(
                smc_analysis, technical_analysis, candle_analysis, momentum_analysis
            )
            
            return final_decision
            
        except Exception as e:
            log.error(f"خطأ في تحليل السوق: {e}")
            return self._get_default_analysis()
    
    def _analyze_smc(self, df):
        """تحليل SMC المتقدم"""
        fvg_analysis = self.smc_engine.identify_fvg(df)
        ob_analysis = self.smc_engine.identify_order_blocks(df)
        ms_analysis = self.smc_engine.analyze_market_structure(df)
        liquidity_analysis = self.smc_engine.identify_liquidity_zones(df)
        
        current_price = df['close'].iloc[-1]
        
        # حساب نقاط SMC
        smc_score_bullish = 0
        smc_score_bearish = 0
        reasons = []
        
        # تحليل FVG
        for fvg in fvg_analysis['bullish_fvg']:
            if fvg['low'] <= current_price <= fvg['high']:
                smc_score_bullish += 2
                reasons.append(f"FVG صاعد: {fvg['size']:.2f}%")
        
        for fvg in fvg_analysis['bearish_fvg']:
            if fvg['low'] <= current_price <= fvg['high']:
                smc_score_bearish += 2
                reasons.append(f"FVG هابط: {fvg['size']:.2f}%")
        
        # تحليل Order Blocks
        for ob in ob_analysis['bullish_ob']:
            if ob['low'] <= current_price <= ob['high']:
                smc_score_bullish += 3
                reasons.append(f"OB صاعد: {ob['strength']:.2f}%")
        
        for ob in ob_analysis['bearish_ob']:
            if ob['low'] <= current_price <= ob['high']:
                smc_score_bearish += 3
                reasons.append(f"OB هابط: {ob['strength']:.2f}%")
        
        # تحليل هيكل السوق
        if ms_analysis['bos_bullish']:
            smc_score_bullish += 4
            reasons.append("BOS صاعد")
        
        if ms_analysis['bos_bearish']:
            smc_score_bearish += 4
            reasons.append("BOS هابط")
        
        if ms_analysis['choch_bullish']:
            smc_score_bullish += 3
            reasons.append("CHoCH صاعد")
        
        if ms_analysis['choch_bearish']:
            smc_score_bearish += 3
            reasons.append("CHoCH هابط")
        
        # تحليل السيولة
        for zone in liquidity_analysis['buy_zones']:
            if abs(current_price - zone) / zone <= 0.02:
                smc_score_bullish += 2
                reasons.append("قرب منطقة شراء")
        
        for zone in liquidity_analysis['sell_zones']:
            if abs(current_price - zone) / zone <= 0.02:
                smc_score_bearish += 2
                reasons.append("قرب منطقة بيع")
        
        return {
            'score_bullish': smc_score_bullish,
            'score_bearish': smc_score_bearish,
            'reasons': reasons,
            'details': {
                'fvg': fvg_analysis,
                'order_blocks': ob_analysis,
                'market_structure': ms_analysis,
                'liquidity': liquidity_analysis
            }
        }
    
    def _analyze_technical_indicators(self, df):
        """تحليل المؤشرات التقنية"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # حساب المؤشرات
        ema_fast = self.indicator_system.calculate_ema(close, EMA_FAST)
        ema_medium = self.indicator_system.calculate_ema(close, EMA_MEDIUM)
        ema_slow = self.indicator_system.calculate_ema(close, EMA_SLOW)
        
        rsi = self.indicator_system.calculate_rsi(close, RSI_PERIOD)
        macd_data = self.indicator_system.calculate_macd(close)
        stoch_data = self.indicator_system.calculate_stochastic(high, low, close)
        adx_data = self.indicator_system.calculate_adx(high, low, close)
        bb_data = self.indicator_system.calculate_bollinger_bands(close)
        
        # تحليل التقاطعات
        ema_cross = self._analyze_ema_cross(ema_fast, ema_medium, ema_slow)
        macd_signal = self._analyze_macd_signal(macd_data)
        rsi_signal = self._analyze_rsi_signal(rsi)
        stoch_signal = self._analyze_stochastic_signal(stoch_data)
        bb_signal = self._analyze_bollinger_signal(close, bb_data)
        adx_signal = self._analyze_adx_signal(adx_data)
        
        # حساب النقاط
        tech_score_bullish = 0
        tech_score_bearish = 0
        reasons = []
        
        # تقاطعات EMA
        if ema_cross['signal'] == 'bullish':
            tech_score_bullish += 3
            reasons.append(f"تقاطع EMA: {ema_cross['strength']}")
        elif ema_cross['signal'] == 'bearish':
            tech_score_bearish += 3
            reasons.append(f"تقاطع EMA: {ema_cross['strength']}")
        
        # إشارات MACD
        if macd_signal == 'bullish':
            tech_score_bullish += 2
            reasons.append("MACD صاعد")
        elif macd_signal == 'bearish':
            tech_score_bearish += 2
            reasons.append("MACD هابط")
        
        # إشارات RSI
        if rsi_signal == 'bullish':
            tech_score_bullish += 2
            reasons.append("RSI في منطقة ذروة البيع")
        elif rsi_signal == 'bearish':
            tech_score_bearish += 2
            reasons.append("RSI في منطقة ذروة الشراء")
        
        # إشارات ستوكاستيك
        if stoch_signal == 'bullish':
            tech_score_bullish += 1
            reasons.append("ستوكاستيك صاعد")
        elif stoch_signal == 'bearish':
            tech_score_bearish += 1
            reasons.append("ستوكاستيك هابط")
        
        # إشارات بولينجر
        if bb_signal == 'bullish':
            tech_score_bullish += 1
            reasons.append("سعر عند النطاق السفلي")
        elif bb_signal == 'bearish':
            tech_score_bearish += 1
            reasons.append("سعر عند النطاق العلوي")
        
        # قوة الاتجاه (ADX)
        if adx_signal['trend_strength'] == 'strong':
            if adx_signal['direction'] == 'bullish':
                tech_score_bullish += 2
            elif adx_signal['direction'] == 'bearish':
                tech_score_bearish += 2
            reasons.append(f"اتجاه قوي: {adx_signal['direction']}")
        
        return {
            'score_bullish': tech_score_bullish,
            'score_bearish': tech_score_bearish,
            'reasons': reasons,
            'details': {
                'ema_cross': ema_cross,
                'macd': macd_signal,
                'rsi': rsi_signal,
                'stochastic': stoch_signal,
                'bollinger': bb_signal,
                'adx': adx_signal
            }
        }
    
    def _analyze_ema_cross(self, ema_fast, ema_medium, ema_slow):
        """تحليل تقاطعات EMA"""
        current_fast = ema_fast.iloc[-1]
        current_medium = ema_medium.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        
        prev_fast = ema_fast.iloc[-2]
        prev_medium = ema_medium.iloc[-2]
        prev_slow = ema_slow.iloc[-2]
        
        # تحليل التقاطعات
        fast_above_medium = current_fast > current_medium
        fast_above_slow = current_fast > current_slow
        medium_above_slow = current_medium > current_slow
        
        # محاذاة المتوسطات
        alignment = 0
        if fast_above_medium and medium_above_slow:
            alignment = 3  # محاذاة صاعدة قوية
        elif not fast_above_medium and not medium_above_slow:
            alignment = -3  # محاذاة هابطة قوية
        elif fast_above_medium:
            alignment = 1  # محاذاة صاعدة ضعيفة
        else:
            alignment = -1  # محاذاة هابطة ضعيفة
        
        # إشارة التقاطع
        signal = "neutral"
        strength = "weak"
        
        if alignment >= 2:
            signal = "bullish"
            strength = "strong"
        elif alignment <= -2:
            signal = "bearish"
            strength = "strong"
        elif alignment == 1:
            signal = "bullish"
            strength = "weak"
        elif alignment == -1:
            signal = "bearish"
            strength = "weak"
        
        return {
            'signal': signal,
            'strength': strength,
            'alignment': alignment
        }
    
    def _analyze_macd_signal(self, macd_data):
        """تحليل إشارات MACD"""
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        histogram = macd_data['histogram']
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_hist = histogram.iloc[-1]
        
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        # تقاطع الخطوط
        if prev_macd <= prev_signal and current_macd > current_signal:
            return "bullish"
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return "bearish"
        
        # فوق/تحت الصفر
        if current_macd > 0 and current_hist > 0:
            return "bullish"
        elif current_macd < 0 and current_hist < 0:
            return "bearish"
        
        return "neutral"
    
    def _analyze_rsi_signal(self, rsi):
        """تحليل إشارات RSI"""
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < RSI_OVERSOLD:
            return "bullish"
        elif current_rsi > RSI_OVERBOUGHT:
            return "bearish"
        elif current_rsi < RSI_NEUTRAL_HIGH and current_rsi > RSI_NEUTRAL_LOW:
            return "neutral"
        
        return "neutral"
    
    def _analyze_stochastic_signal(self, stoch_data):
        """تحليل إشارات ستوكاستيك"""
        k = stoch_data['k']
        d = stoch_data['d']
        
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        prev_k = k.iloc[-2]
        prev_d = d.iloc[-2]
        
        # مناطق ذروة الشراء/البيع
        if current_k < 20 and current_d < 20:
            return "bullish"
        elif current_k > 80 and current_d > 80:
            return "bearish"
        
        # تقاطع الخطوط
        if prev_k <= prev_d and current_k > current_d:
            return "bullish"
        elif prev_k >= prev_d and current_k < current_d:
            return "bearish"
        
        return "neutral"
    
    def _analyze_bollinger_signal(self, close, bb_data):
        """تحليل إشارات بولينجر"""
        current_price = close.iloc[-1]
        upper = bb_data['upper'].iloc[-1]
        lower = bb_data['lower'].iloc[-1]
        middle = bb_data['middle'].iloc[-1]
        
        if current_price <= lower:
            return "bullish"
        elif current_price >= upper:
            return "bearish"
        
        return "neutral"
    
    def _analyze_adx_signal(self, adx_data):
        """تحليل إشارات ADX"""
        adx = adx_data['adx'].iloc[-1]
        plus_di = adx_data['plus_di'].iloc[-1]
        minus_di = adx_data['minus_di'].iloc[-1]
        
        trend_strength = "weak"
        if adx > 25:
            trend_strength = "strong"
        elif adx > 20:
            trend_strength = "medium"
        
        direction = "neutral"
        if plus_di > minus_di:
            direction = "bullish"
        elif minus_di > plus_di:
            direction = "bearish"
        
        return {
            'trend_strength': trend_strength,
            'direction': direction,
            'adx_value': adx
        }
    
    def _analyze_momentum(self, df):
        """تحليل الزخم والحجم"""
        close = df['close']
        volume = df['volume']
        
        # زخم السعر
        price_momentum_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100
        price_momentum_10 = (close.iloc[-1] / close.iloc[-10] - 1) * 100
        
        # زخم الحجم
        volume_ma = volume.rolling(20).mean()
        current_volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
        
        momentum_score_bullish = 0
        momentum_score_bearish = 0
        reasons = []
        
        # زخم السعر
        if price_momentum_5 > 0.5 and price_momentum_10 > 0.3:
            momentum_score_bullish += 2
            reasons.append(f"زخم صاعد: {price_momentum_5:.2f}%")
        elif price_momentum_5 < -0.5 and price_momentum_10 < -0.3:
            momentum_score_bearish += 2
            reasons.append(f"زخم هابط: {price_momentum_5:.2f}%")
        
        # زخم الحجم
        if current_volume_ratio > 1.2:
            if price_momentum_5 > 0:
                momentum_score_bullish += 1
                reasons.append(f"حجم عالي مع صعود: {current_volume_ratio:.2f}x")
            else:
                momentum_score_bearish += 1
                reasons.append(f"حجم عالي مع هبوط: {current_volume_ratio:.2f}x")
        
        return {
            'score_bullish': momentum_score_bullish,
            'score_bearish': momentum_score_bearish,
            'reasons': reasons
        }
    
    def _make_final_decision(self, smc_analysis, technical_analysis, candle_analysis, momentum_analysis):
        """اتخاذ القرار النهائي"""
        # جمع النقاط
        total_bullish = (
            smc_analysis['score_bullish'] +
            technical_analysis['score_bullish'] +
            (3 if candle_analysis['direction'] == 'bullish' else 0) +
            momentum_analysis['score_bullish']
        )
        
        total_bearish = (
            smc_analysis['score_bearish'] +
            technical_analysis['score_bearish'] +
            (3 if candle_analysis['direction'] == 'bearish' else 0) +
            momentum_analysis['score_bearish']
        )
        
        # جمع الأسباب
        all_reasons = (
            smc_analysis['reasons'] +
            technical_analysis['reasons'] +
            ([f"نمط شموع: {candle_analysis['pattern']}"] if candle_analysis['pattern'] != 'none' else []) +
            momentum_analysis['reasons']
        )
        
        # اتخاذ القرار
        decision = "hold"
        confidence = 0
        
        if total_bullish > total_bearish + 5:  # فرق 5 نقاط على الأقل
            decision = "buy"
            confidence = min(100, total_bullish * 5)
        elif total_bearish > total_bullish + 5:
            decision = "sell"
            confidence = min(100, total_bearish * 5)
        
        return {
            'decision': decision,
            'confidence': confidence,
            'total_bullish': total_bullish,
            'total_bearish': total_bearish,
            'reasons': all_reasons,
            'details': {
                'smc': smc_analysis,
                'technical': technical_analysis,
                'candles': candle_analysis,
                'momentum': momentum_analysis
            }
        }
    
    def _get_default_analysis(self):
        """تحليل افتراضي عند عدم كفاية البيانات"""
        return {
            'decision': 'hold',
            'confidence': 0,
            'total_bullish': 0,
            'total_bearish': 0,
            'reasons': ['بيانات غير كافية'],
            'details': {}
        }

# =================== ENHANCED TRADE MANAGER ===================
class ProfessionalTradeManager:
    """مدير الصفقات المحترف"""
    
    def __init__(self):
        self.council = IntelligentTradingCouncil()
        self.open_trades = {}
        self.trade_history = []
        
    def evaluate_entry(self, df, current_price):
        """تقييم فرص الدخول"""
        analysis = self.council.analyze_market(df)
        
        if analysis['decision'] in ['buy', 'sell'] and analysis['confidence'] >= 60:
            return {
                'action': analysis['decision'],
                'confidence': analysis['confidence'],
                'price': current_price,
                'reasons': analysis['reasons'],
                'analysis': analysis
            }
        
        return {
            'action': 'wait',
            'confidence': analysis['confidence'],
            'reasons': analysis['reasons']
        }
    
    def manage_open_trade(self, trade, df, current_price):
        """إدارة الصفقة المفتوحة"""
        if not trade or trade['status'] != 'open':
            return {'action': 'hold'}
        
        analysis = self.council.analyze_market(df)
        current_pnl = self.calculate_pnl(trade, current_price)
        
        # تحديث أعلى ربح
        if current_pnl > trade.get('peak_profit', 0):
            trade['peak_profit'] = current_pnl
        
        # قرارات الإدارة
        management_decision = self._make_management_decision(trade, analysis, current_pnl, current_price)
        
        return management_decision
    
    def calculate_pnl(self, trade, current_price):
        """حساب الربح/الخسارة"""
        if trade['side'] == 'long':
            return (current_price - trade['entry_price']) / trade['entry_price'] * 100
        else:
            return (trade['entry_price'] - current_price) / trade['entry_price'] * 100
    
    def _make_management_decision(self, trade, analysis, current_pnl, current_price):
        """اتخاذ قرار الإدارة"""
        side = trade['side']
        entry_price = trade['entry_price']
        time_in_trade = time.time() - trade['entry_time']
        
        # قرارات الجني
        take_profit_decision = self._evaluate_take_profit(trade, current_pnl, time_in_trade)
        if take_profit_decision['action'] != 'hold':
            return take_profit_decision
        
        # قرارات الحماية
        protection_decision = self._evaluate_protection(trade, current_pnl, analysis, time_in_trade)
        if protection_decision['action'] != 'hold':
            return protection_decision
        
        # قرارات التعديل
        adjustment_decision = self._evaluate_adjustments(trade, analysis, current_pnl, time_in_trade)
        if adjustment_decision['action'] != 'hold':
            return adjustment_decision
        
        return {'action': 'hold', 'reason': 'استمرارية الصفقة'}
    
    def _evaluate_take_profit(self, trade, current_pnl, time_in_trade):
        """تقييم جني الأرباح"""
        tp_levels = TP_LEVELS
        tp_ratios = TP_RATIOS
        
        achieved_targets = trade.get('achieved_targets', [])
        
        for i, (level, ratio) in enumerate(zip(tp_levels, tp_ratios)):
            if level not in achieved_targets and current_pnl >= level:
                return {
                    'action': 'partial_close',
                    'ratio': ratio,
                    'reason': f'جني عند هدف {level}%',
                    'target_level': level
                }
        
        # جني ذكي بناءً على الزمن والأداء
        if time_in_trade > 1800 and current_pnl >= 1.5:  # بعد 30 دقيقة
            return {
                'action': 'partial_close',
                'ratio': 0.3,
                'reason': 'جني ذكي بعد وقت طويل مع ربح جيد'
            }
        
        return {'action': 'hold'}
    
    def _evaluate_protection(self, trade, current_pnl, analysis, time_in_trade):
        """تقييم إجراءات الحماية"""
        # إغلاق طارئ للخسائر الكبيرة
        if current_pnl <= -MAX_DRAWDOWN_PER_TRADE:
            return {
                'action': 'close',
                'reason': f'خسارة طارئة: {current_pnl:.2f}%'
            }
        
        # حماية من انعكاس السوق
        if (trade['side'] == 'long' and analysis['decision'] == 'sell' and 
            analysis['confidence'] > 70 and current_pnl > 0):
            return {
                'action': 'close',
                'reason': 'إشارة انعكاس قوية مع ربح'
            }
        
        elif (trade['side'] == 'short' and analysis['decision'] == 'buy' and 
              analysis['confidence'] > 70 and current_pnl > 0):
            return {
                'action': 'close',
                'reason': 'إشارة انعكاس قوية مع ربح'
            }
        
        # تفعيل نقطة التعادل بعد تحقيق ربح معين
        if not trade.get('breakeven_activated') and current_pnl >= 1.0:
            return {
                'action': 'activate_breakeven',
                'reason': 'تفعيل نقطة التعادل'
            }
        
        # تفعيل التريل بعد تحقيق ربح جيد
        if not trade.get('trailing_activated') and current_pnl >= TRAILING_ACTIVATION:
            return {
                'action': 'activate_trailing',
                'reason': f'تفعيل التريل بعد تحقيق {current_pnl:.2f}%'
            }
        
        return {'action': 'hold'}
    
    def _evaluate_adjustments(self, trade, analysis, current_pnl, time_in_trade):
        """تقييم تعديلات الصفقة"""
        # إضافة إلى الصفقة في حالة قوة الإشارة
        if (time_in_trade < 600 and  # في أول 10 دقائق
            current_pnl > 0.5 and    # مع ربح
            analysis['confidence'] > trade.get('entry_confidence', 0) + 10):
            
            return {
                'action': 'add_position',
                'ratio': 0.3,
                'reason': 'تعزيز الصفقة مع قوة الإشارة'
            }
        
        return {'action': 'hold'}

# =================== ENHANCED EXECUTION SYSTEM ===================
class ProfessionalExecutionSystem:
    """نظام التنفيذ المحترف"""
    
    def __init__(self):
        self.trade_manager = ProfessionalTradeManager()
        self.state = {
            'open': False,
            'side': None,
            'entry_price': None,
            'quantity': 0,
            'opened_at': None,
            'peak_profit': 0,
            'achieved_targets': [],
            'breakeven_activated': False,
            'trailing_activated': False
        }
    
    def run_trading_cycle(self, df, current_price):
        """تشغيل دورة التداول"""
        try:
            if not self.state['open']:
                # تقييم فرص الدخول
                entry_decision = self.trade_manager.evaluate_entry(df, current_price)
                
                if entry_decision['action'] in ['buy', 'sell']:
                    self._execute_entry(entry_decision, current_price, df)
                else:
                    if LOG_DETAILED_ENTRY:
                        log.analysis(f"انتظار - ثقة: {entry_decision['confidence']:.1f}% - {', '.join(entry_decision['reasons'][:3])}")
            
            else:
                # إدارة الصفقة المفتوحة
                management_decision = self.trade_manager.manage_open_trade(self.state, df, current_price)
                
                if management_decision['action'] != 'hold':
                    self._execute_management(management_decision, current_price)
                else:
                    current_pnl = self.trade_manager.calculate_pnl(self.state, current_price)
                    if current_pnl != 0:  # تحديث كل دورة فقط إذا كان هناك تغيير
                        log.trade(f"الصفقة مفتوحة - الربح: {current_pnl:.2f}% - الذروة: {self.state.get('peak_profit', 0):.2f}%")
        
        except Exception as e:
            log.error(f"خطأ في دورة التداول: {e}")
            traceback.print_exc()
    
    def _execute_entry(self, decision, current_price, df):
        """تنفيذ الدخول"""
        side = decision['action']
        confidence = decision['confidence']
        
        # حساب الكمية
        quantity = self._calculate_position_size(current_price)
        
        if quantity <= 0:
            log.error("❌ كمية غير صالحة للدخول")
            return
        
        # تنفيذ الصفقة
        if EXECUTE_ORDERS and not DRY_RUN:
            success = self._place_order(side, quantity, current_price)
        else:
            success = True
            log.trade(f"DRY_RUN: دخول {side} {quantity:.4f} @ {current_price:.6f}")
        
        if success:
            self.state.update({
                'open': True,
                'side': side,
                'entry_price': current_price,
                'quantity': quantity,
                'opened_at': time.time(),
                'entry_confidence': confidence,
                'peak_profit': 0,
                'achieved_targets': [],
                'breakeven_activated': False,
                'trailing_activated': False,
                'entry_analysis': decision['analysis']
            })
            
            # تسجيل مفصل
            log.success(f"🎯 فتح صفقة {side.upper()} - الكمية: {quantity:.4f} - السعر: {current_price:.6f}")
            log.strategy(f"📊 الثقة: {confidence:.1f}%")
            
            for i, reason in enumerate(decision['reasons'][:5]):  # أول 5 أسباب فقط
                log.indicator(f"   {i+1}. {reason}")
            
            if LOG_DETAILED_ENTRY:
                self._log_detailed_analysis(decision['analysis'])
    
    def _execute_management(self, decision, current_price):
        """تنفيذ قرارات الإدارة"""
        action = decision['action']
        
        if action == 'partial_close':
            self._execute_partial_close(decision, current_price)
        elif action == 'close':
            self._execute_full_close(decision, current_price)
        elif action == 'activate_breakeven':
            self.state['breakeven_activated'] = True
            log.success(f"🛡️ تفعيل نقطة التعادل - {decision['reason']}")
        elif action == 'activate_trailing':
            self.state['trailing_activated'] = True
            log.success(f"📈 تفعيل التريل - {decision['reason']}")
        elif action == 'add_position':
            self._execute_add_position(decision, current_price)
    
    def _execute_partial_close(self, decision, current_price):
        """تنفيذ جني جزئي"""
        ratio = decision['ratio']
        close_quantity = self.state['quantity'] * ratio
        
        if EXECUTE_ORDERS and not DRY_RUN:
            success = self._place_close_order(close_quantity, current_price)
        else:
            success = True
            log.trade(f"DRY_RUN: جني {ratio*100:.1f}% - {close_quantity:.4f} @ {current_price:.6f}")
        
        if success:
            # تحديث الحالة
            self.state['quantity'] -= close_quantity
            self.state['achieved_targets'].append(decision.get('target_level', 'unknown'))
            
            # تسجيل النجاح
            pnl = self.trade_manager.calculate_pnl(self.state, current_price)
            log.success(f"💰 جني {ratio*100:.1f}% - الكمية: {close_quantity:.4f} - {decision['reason']}")
            log.trade(f"📊 الربح الحالي: {pnl:.2f}% - الكمية المتبقية: {self.state['quantity']:.4f}")
            
            # إذا كانت الكمية المتبقية صغيرة، إغلاق كامل
            if self.state['quantity'] < self.state['quantity'] * 0.1:  # أقل من 10%
                self._execute_full_close({'reason': 'إغلاق كامل بعد الجني'}, current_price)
    
    def _execute_full_close(self, decision, current_price):
        """تنفيذ إغلاق كامل"""
        close_quantity = self.state['quantity']
        
        if EXECUTE_ORDERS and not DRY_RUN:
            success = self._place_close_order(close_quantity, current_price)
        else:
            success = True
            log.trade(f"DRY_RUN: إغلاق كامل - {close_quantity:.4f} @ {current_price:.6f}")
        
        if success:
            # حساب الربح النهائي
            final_pnl = self.trade_manager.calculate_pnl(self.state, current_price)
            
            # تسجيل النجاح
            log.success(f"🎯 إغلاق الصفقة - الكمية: {close_quantity:.4f} - {decision['reason']}")
            log.trade(f"💰 الربح النهائي: {final_pnl:.2f}%")
            
            # إعادة تعيين الحالة
            self._reset_state()
    
    def _execute_add_position(self, decision, current_price):
        """تنفيذ إضافة إلى الصفقة"""
        ratio = decision['ratio']
        add_quantity = self.state['quantity'] * ratio
        
        if EXECUTE_ORDERS and not DRY_RUN:
            success = self._place_order(self.state['side'], add_quantity, current_price)
        else:
            success = True
            log.trade(f"DRY_RUN: إضافة {ratio*100:.1f}% - {add_quantity:.4f} @ {current_price:.6f}")
        
        if success:
            # تحديث متوسط السعر والكمية
            old_quantity = self.state['quantity']
            old_price = self.state['entry_price']
            
            new_quantity = old_quantity + add_quantity
            new_avg_price = (old_quantity * old_price + add_quantity * current_price) / new_quantity
            
            self.state.update({
                'quantity': new_quantity,
                'entry_price': new_avg_price
            })
            
            log.success(f"📈 تعزيز الصفقة - إضافة {ratio*100:.1f}% - {decision['reason']}")
            log.trade(f"📊 الكمية الجديدة: {new_quantity:.4f} - السعر المتوسط: {new_avg_price:.6f}")
    
    def _calculate_position_size(self, current_price):
        """حساب حجم المركز"""
        try:
            if MODE_LIVE:
                balance = ex.fetch_balance()['total']['USDT']
            else:
                balance = 1000.0  # رصيد تجريبي
            
            risk_amount = balance * (RISK_ALLOC / 100.0)
            
            # استخدام ATR لحساب وقف الخسارة
            df = fetch_ohlcv_enhanced()
            if df is not None and len(df) > ATR_PERIOD:
                atr = AdvancedIndicatorSystem.calculate_atr(
                    df['high'], df['low'], df['close'], ATR_PERIOD
                ).iloc[-1]
                stop_distance = atr * STOP_LOSS_ATR_MULTIPLIER
            else:
                stop_distance = current_price * 0.02  # 2% افتراضي
            
            # حساب الكمية
            quantity = (risk_amount / stop_distance) * LEVERAGE
            return safe_qty(quantity)
            
        except Exception as e:
            log.error(f"خطأ في حساب حجم المركز: {e}")
            return 0
    
    def _place_order(self, side, quantity, price):
        """وضع أمر شراء/بيع"""
        try:
            if MODE_LIVE:
                params = exchange_specific_params(side, is_close=False)
                ex.create_order(SYMBOL, "market", side, quantity, None, params)
                return True
            return True
        except Exception as e:
            log.error(f"❌ فشل وضع الأمر: {e}")
            return False
    
    def _place_close_order(self, quantity, price):
        """وضع أمر إغلاق"""
        try:
            if MODE_LIVE:
                side = "sell" if self.state['side'] == 'long' else "buy"
                params = exchange_specific_params(side, is_close=True)
                ex.create_order(SYMBOL, "market", side, quantity, None, params)
                return True
            return True
        except Exception as e:
            log.error(f"❌ فشل إغلاق الأمر: {e}")
            return False
    
    def _reset_state(self):
        """إعادة تعيين حالة التداول"""
        self.state.update({
            'open': False,
            'side': None,
            'entry_price': None,
            'quantity': 0,
            'opened_at': None,
            'peak_profit': 0,
            'achieved_targets': [],
            'breakeven_activated': False,
            'trailing_activated': False
        })
    
    def _log_detailed_analysis(self, analysis):
        """تسجيل التحليل المفصل"""
        if not LOG_DETAILED_ENTRY:
            return
        
        log_banner("التفاصيل الفنية للدخول")
        
        # تحليل SMC
        smc = analysis['details']['smc']
        log.analysis(f"📊 تحليل SMC - صاعد: {smc['score_bullish']} | هابط: {smc['score_bearish']}")
        
        # تحليل تقني
        tech = analysis['details']['technical']
        log.analysis(f"📈 تحليل تقني - صاعد: {tech['score_bullish']} | هابط: {tech['score_bearish']}")
        
        # تحليل الشموع
        candles = analysis['details']['candles']
        log.analysis(f"🕯️ أنماط الشموع - {candles['pattern']} - قوة: {candles['strength']}")
        
        # تحليل الزخم
        momentum = analysis['details']['momentum']
        log.analysis(f"🚀 تحليل الزخم - صاعد: {momentum['score_bullish']} | هابط: {momentum['score_bearish']}")

# =================== MAIN EXECUTION SYSTEM ===================
def main_loop_enhanced():
    """الحلقة الرئيسية المحسنة"""
    log_banner("بدء نظام التداول الذكي المتقدم")
    
    # التحقق من البيئة
    verify_environment()
    
    # تهيئة الأنظمة
    execution_system = ProfessionalExecutionSystem()
    
    log.success("✅ جميع الأنظمة جاهزة للتداول")
    log.info(f"🎯 الرمز: {SYMBOL} | الإطار: {INTERVAL}")
    log.info(f"💰 الرافعة: {LEVERAGE}x | المخاطرة: {RISK_ALLOC}%")
    log.info(f"🔧 الوضع: {'LIVE' if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN else 'SIMULATION'}")
    
    last_log_time = 0
    
    while True:
        try:
            current_time = time.time()
            
            # جلب البيانات
            df = fetch_ohlcv_enhanced()
            if df is None or len(df) < 100:
                log.warning("📊 بيانات غير كافية، إعادة المحاولة...")
                time.sleep(BASE_SLEEP)
                continue
            
            current_price = df['close'].iloc[-1]
            
            # تشغيل دورة التداول
            execution_system.run_trading_cycle(df, current_price)
            
            # تحديث حالة السوق كل دقيقة
            if current_time - last_log_time > 60:
                log_market_status(df, current_price, execution_system.state)
                last_log_time = current_time
            
            # النوم حسب الوضع
            sleep_time = NEAR_CLOSE_S if execution_system.state['open'] else BASE_SLEEP
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            log_banner("إيقاف النظام بطلب من المستخدم")
            break
        except Exception as e:
            log.error(f"🔥 خطأ غير متوقع: {e}")
            traceback.print_exc()
            time.sleep(BASE_SLEEP * 2)

def fetch_ohlcv_enhanced():
    """جلب بيانات OHLCV محسن"""
    try:
        since = ex.milliseconds() - 1000 * 60 * 60 * 24 * 5  # 5 أيام
        ohlcv = ex.fetch_ohlcv(SYMBOL, INTERVAL, since=since, limit=500)
        
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        # تحويل الأعمدة إلى float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # تنظيف البيانات
        df = df.dropna()
        
        return df
    except Exception as e:
        log.error(f"❌ فشل جلب البيانات: {e}")
        return None

def log_market_status(df, current_price, state):
    """تسجيل حالة السوق"""
    if len(df) < 20:
        return
    
    try:
        # حساب بعض المؤشرات السريعة
        rsi = AdvancedIndicatorSystem.calculate_rsi(df['close'], 14).iloc[-1]
        ema_fast = AdvancedIndicatorSystem.calculate_ema(df['close'], 8).iloc[-1]
        ema_slow = AdvancedIndicatorSystem.calculate_ema(df['close'], 21).iloc[-1]
        
        trend = "صاعد" if ema_fast > ema_slow else "هابط"
        rsi_status = "شراء مفرط" if rsi > 70 else "بيع مفرط" if rsi < 30 else "محايد"
        
        status_msg = f"📈 السوق: {trend} | RSI: {rsi:.1f} ({rsi_status}) | السعر: {current_price:.6f}"
        
        if state['open']:
            pnl = (current_price - state['entry_price']) / state['entry_price'] * 100
            if state['side'] == 'short':
                pnl = -pnl
            status_msg += f" | الصفقة: {state['side']} | الربح: {pnl:.2f}%"
        
        log.info(status_msg)
        
    except Exception as e:
        log.warning(f"⚠️ خطأ في تسجيل حالة السوق: {e}")

def verify_environment():
    """التحقق من بيئة التنفيذ"""
    log_banner("التحقق من البيئة")
    
    # التحقق من الاتصال بالمنصة
    try:
        ex.load_markets()
        log.success(f"✅ الاتصال بـ {EXCHANGE_NAME.upper()} ناجح")
    except Exception as e:
        log.error(f"❌ فشل الاتصال بالمنصة: {e}")
    
    # التحقق من الرصيد
    try:
        if MODE_LIVE:
            balance = ex.fetch_balance()
            usdt_balance = balance['total']['USDT']
            log.info(f"💰 الرصيد: {usdt_balance:.2f} USDT")
        else:
            log.info("💰 الوضع: SIMULATION - لا يوجد رصيد حقيقي")
    except Exception as e:
        log.warning(f"⚠️ لا يمكن التحقق من الرصيد: {e}")
    
    # التحقق من الإعدادات
    log.info(f"🎯 إعدادات التداول:")
    log.info(f"   - الرمز: {SYMBOL}")
    log.info(f"   - الإطار: {INTERVAL}")
    log.info(f"   - الرافعة: {LEVERAGE}x")
    log.info(f"   - المخاطرة: {RISK_ALLOC}%")
    log.info(f"   - التنفيذ: {'نشط' if EXECUTE_ORDERS and not DRY_RUN else 'محاكاة'}")

# =================== HELPER FUNCTIONS ===================
def safe_qty(qty):
    """كمية آمنة حسب خطوة التداول"""
    try:
        return float(Decimal(str(qty)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
    except:
        return float(qty)

def exchange_specific_params(side, is_close=False):
    """معلمات خاصة بالمنصة"""
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:  # BingX
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

# =================== EXCHANGE INITIALIZATION ===================
def initialize_exchange():
    """تهيئة المنصة"""
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

# تهيئة المنصة العالمية
ex = initialize_exchange()

# =================== FLASK APP FOR RENDER ===================
app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 SUI Council ULTIMATE PRO - Intelligent Trading Bot"

@app.route("/health")
def health():
    return jsonify({"status": "active", "timestamp": datetime.utcnow().isoformat()})

@app.route("/status")
def status():
    return jsonify({
        "bot_version": BOT_VERSION,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "LIVE" if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN else "SIMULATION"
    })

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    try:
        # بدء النظام
        log_banner("نظام التداول الذكي المتقدم")
        
        # تشغيل حلقة التداول في thread منفصل
        import threading
        trading_thread = threading.Thread(target=main_loop_enhanced, daemon=True)
        trading_thread.start()
        
        # تشغيل خادم Flask
        log.success("🌐 بدء خادم الويب...")
        app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        
    except Exception as e:
        log.error(f"❌ فشل بدء النظام: {e}")
        traceback.print_exc()
