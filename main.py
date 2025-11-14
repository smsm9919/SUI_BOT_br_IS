# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المحترف المتكامل مع مؤشرات TradingView/Bybit
• مجلس الإدارة الفائق الذكي مع SMC + عرض/طلب + تحليل محترف
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• كشف التلاعب والتذبذب والكسر الحقيقي/الوهمي
• إدارة صفقات ذكية متكيفة مع قوة الترند
• نظام Footprint + Diagonal Order-Flow المتقدم
• Multi-Exchange Support: BingX & Bybit
• TradingView/Bybit Precision Indicators
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from scipy import stats

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== TRADINGVIEW-STYLE TECHNICAL INDICATORS ===================
class TradingViewIndicators:
    """مكتبة مؤشرات بنمط TradingView/Bybit الدقيق"""
    
    @staticmethod
    def tv_rma(series: pd.Series, length: int) -> pd.Series:
        """Running Moving Average (كما في TradingView)"""
        series = series.astype(float)
        alpha = 1.0 / length
        rma = series.ewm(alpha=alpha, adjust=False).mean()
        return rma
    
    @staticmethod
    def tv_rsi(close: pd.Series, length: int = 14) -> pd.Series:
        """مؤشر RSI بنمط TradingView الدقيق"""
        close = close.astype(float)
        diff = close.diff()
        gain = diff.where(diff > 0, 0.0)
        loss = -diff.where(diff < 0, 0.0)
        avg_gain = TradingViewIndicators.tv_rma(gain, length)
        avg_loss = TradingViewIndicators.tv_rma(loss, length)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def tv_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """مؤشر ATR بنمط TradingView الدقيق"""
        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = TradingViewIndicators.tv_rma(tr, length)
        return atr
    
    @staticmethod
    def tv_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
        """مؤشر ADX بنمط TradingView الدقيق"""
        high = high.astype(float)
        low = low.astype(float)
        close = close.astype(float)

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0), up_move, 0.0
        )
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0), down_move, 0.0
        )

        tr = TradingViewIndicators.tv_atr(high, low, close, length) * length
        tr_rma = TradingViewIndicators.tv_rma(tr, length)
        plus_dm_rma = TradingViewIndicators.tv_rma(pd.Series(plus_dm, index=high.index), length)
        minus_dm_rma = TradingViewIndicators.tv_rma(pd.Series(minus_dm, index=high.index), length)

        plus_di = 100 * (plus_dm_rma / tr_rma.replace(0, np.nan))
        minus_di = 100 * (minus_dm_rma / tr_rma.replace(0, np.nan))

        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = TradingViewIndicators.tv_rma(dx, length)

        return adx, plus_di, minus_di
    
    @staticmethod
    def tv_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """مؤشر MACD بنمط TradingView الدقيق"""
        close = close.astype(float)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def tv_ema(data: pd.Series, period: int):
        """المتوسط المتحرك الأسي بنمط TradingView"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def tv_sma(data: pd.Series, period: int):
        """المتوسط المتحرك البسيط بنمط TradingView"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def tv_bollinger_bands(data, period=20, std_dev=2):
        """نطاقات بولينجر بنمط TradingView"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def tv_stoch(high, low, close, k_period=14, d_period=3):
        """مؤشر ستوكاستك بنمط TradingView"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def tv_obv(close, volume):
        """حجم الرصيد بنمط TradingView"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

# إنشاء كائن المؤشرات بنمط TradingView
tv = TradingViewIndicators()

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
BOT_VERSION = f"SUI ULTRA PRO AI v10.0 — {EXCHANGE_NAME.upper()} - TRADINGVIEW PRECISION"
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

# ===== SUPER COUNCIL ENHANCEMENTS =====
COUNCIL_AI_MODE = True
TREND_EARLY_DETECTION = True
MOMENTUM_ACCELERATION = True
VOLUME_CONFIRMATION = True
PRICE_ACTION_INTELLIGENCE = True

# أوزان التصويت الذكية المحسنة
WEIGHT_ADX = 1.8
WEIGHT_RSI = 1.4
WEIGHT_MACD = 1.6
WEIGHT_VOLUME = 1.3
WEIGHT_FLOW = 1.7
WEIGHT_GOLDEN = 2.0
WEIGHT_CANDLES = 1.4
WEIGHT_MOMENTUM = 1.6
WEIGHT_FOOTPRINT = 1.8
WEIGHT_DIAGONAL = 1.7
WEIGHT_EARLY_TREND = 2.0
WEIGHT_BREAKOUT = 2.2
WEIGHT_MARKET_STRUCTURE = 1.9
WEIGHT_VOLATILITY = 1.2
WEIGHT_SENTIMENT = 1.5

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

# =================== PROFESSIONAL MARKET ANALYSIS ===================
class ProfessionalMarketAnalyzer:
    def __init__(self):
        self.supply_zones = []
        self.demand_zones = []
        self.manipulation_signals = []
        self.breakout_levels = []
        
    def detect_supply_demand_zones(self, df):
        """كشف مناطق العرض والطلب المحترفة"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            volume = df['volume'].astype(float)
            
            supply_zones = []
            demand_zones = []
            
            # البحث عن مناطق العرض (مقاومة)
            for i in range(2, len(df)-5):
                # منطقة عرض: قمة عالية + حجم عالي + انعكاس هابط
                if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and
                    volume.iloc[i] > volume.iloc[i-1] * 1.5 and
                    all(close.iloc[i+j] < close.iloc[i] for j in range(1, 4))):
                    
                    supply_zones.append({
                        'price': high.iloc[i],
                        'strength': (high.iloc[i] - low.iloc[i]) / low.iloc[i] * 100,
                        'volume_ratio': volume.iloc[i] / volume.iloc[i-1],
                        'timestamp': df['time'].iloc[i]
                    })
            
            # البحث عن مناطق الطلب (دعم)
            for i in range(2, len(df)-5):
                # منطقة طلب: قاع منخفض + حجم عالي + انعكاس صاعد
                if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and
                    volume.iloc[i] > volume.iloc[i-1] * 1.5 and
                    all(close.iloc[i+j] > close.iloc[i] for j in range(1, 4))):
                    
                    demand_zones.append({
                        'price': low.iloc[i],
                        'strength': (high.iloc[i] - low.iloc[i]) / low.iloc[i] * 100,
                        'volume_ratio': volume.iloc[i] / volume.iloc[i-1],
                        'timestamp': df['time'].iloc[i]
                    })
            
            return {
                'supply_zones': sorted(supply_zones, key=lambda x: x['price'], reverse=True)[:5],
                'demand_zones': sorted(demand_zones, key=lambda x: x['price'])[:5]
            }
        except Exception as e:
            return {'supply_zones': [], 'demand_zones': []}

    def detect_manipulation_volatility(self, df):
        """كشف التلاعب والتذبذب غير الطبيعي"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            volume = df['volume'].astype(float)
            
            signals = []
            current_price = close.iloc[-1]
            
            # كشف التذبذب المفاجئ
            price_changes = close.pct_change().dropna()
            volatility = price_changes.rolling(10).std()
            current_volatility = volatility.iloc[-1] if len(volatility) > 0 else 0
            
            # كشف الشموع ذات الظلال الطويلة (تلاعب)
            for i in range(max(0, len(df)-10), len(df)):
                candle_range = high.iloc[i] - low.iloc[i]
                body_size = abs(close.iloc[i] - df['open'].iloc[i])
                upper_wick = high.iloc[i] - max(close.iloc[i], df['open'].iloc[i])
                lower_wick = min(close.iloc[i], df['open'].iloc[i]) - low.iloc[i]
                
                # شمعة مطرقة أو شهاب (تلاعب محتمل)
                if (upper_wick > body_size * 2 and lower_wick < body_size * 0.5) or \
                   (lower_wick > body_size * 2 and upper_wick < body_size * 0.5):
                    signals.append({
                        'type': 'manipulation_candle',
                        'index': i,
                        'strength': max(upper_wick, lower_wick) / body_size,
                        'direction': 'bearish' if upper_wick > lower_wick else 'bullish'
                    })
            
            # كشف الاختراقات الوهمية
            fake_breakouts = self.detect_fake_breakouts(df)
            
            return {
                'current_volatility': current_volatility,
                'manipulation_signals': signals[-3:],  # آخر 3 إشارات
                'fake_breakouts': fake_breakouts,
                'high_volatility_alert': current_volatility > 0.02
            }
        except Exception as e:
            return {'current_volatility': 0, 'manipulation_signals': [], 'fake_breakouts': [], 'high_volatility_alert': False}

    def detect_fake_breakouts(self, df):
        """كشف الاختراقات الوهمية مقابل الحقيقية"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            fake_breakouts = []
            
            for i in range(3, len(df)-2):
                # اختراق وهمي: كسر مستوى ثم عودة سريعة
                if (high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and
                    all(close.iloc[i+j] < high.iloc[i-1] for j in range(1, 3))):
                    fake_breakouts.append({
                        'type': 'fake_breakout_high',
                        'level': high.iloc[i-1],
                        'timestamp': df['time'].iloc[i]
                    })
                
                if (low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and
                    all(close.iloc[i+j] > low.iloc[i-1] for j in range(1, 3))):
                    fake_breakouts.append({
                        'type': 'fake_breakout_low',
                        'level': low.iloc[i-1],
                        'timestamp': df['time'].iloc[i]
                    })
            
            return fake_breakouts[-5:]  # آخر 5 اختراقات وهمية
        except Exception as e:
            return []

    def analyze_price_testing(self, df):
        """تحليل اختبار السعر والاختراقات"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            current_price = close.iloc[-1]
            recent_high = high.tail(20).max()
            recent_low = low.tail(20).min()
            
            # تحليل اختبار المستويات
            price_tests = {
                'testing_resistance': current_price >= recent_high * 0.995,
                'testing_support': current_price <= recent_low * 1.005,
                'breakout_confirmed': current_price > recent_high,
                'breakdown_confirmed': current_price < recent_low,
                'recent_high': recent_high,
                'recent_low': recent_low
            }
            
            # تحليل قوة الاختراق
            if price_tests['breakout_confirmed']:
                volume_confirmation = df['volume'].tail(3).mean() > df['volume'].tail(10).mean() * 1.2
                price_tests['breakout_strength'] = 'strong' if volume_confirmation else 'weak'
            elif price_tests['breakdown_confirmed']:
                volume_confirmation = df['volume'].tail(3).mean() > df['volume'].tail(10).mean() * 1.2
                price_tests['breakdown_strength'] = 'strong' if volume_confirmation else 'weak'
            
            return price_tests
        except Exception as e:
            return {}

# إنشاء محلل السوق المحترف
pro_market_analyzer = ProfessionalMarketAnalyzer()

# =================== SUPER SMC ENGINE ===================
class SuperSMCEngine:
    def __init__(self):
        self.order_blocks = []
        self.fvgs = []
        self.liquidity_zones = []
        self.market_structure = []
        
    def detect_order_blocks(self, df):
        """كشف Order Blocks (OB) - مناطق الأوامر"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            open_ = df['open'].astype(float)
            
            ob_blocks = []
            for i in range(2, len(df)-1):
                # Bullish OB: شمعة صاعدة كبيرة تليها شمعة هابطة
                if (close[i] > open_[i] and (high[i] - low[i]) > (high[i-1] - low[i-1]) * 0.8 and
                    close[i+1] < open_[i+1]):
                    ob_blocks.append({
                        'type': 'bullish_ob',
                        'price_level': low[i],
                        'timestamp': df['time'].iloc[i],
                        'strength': (high[i] - low[i]) / low[i] * 100
                    })
                # Bearish OB: شمعة هابطة كبيرة تليها شمعة صاعدة
                elif (close[i] < open_[i] and (high[i] - low[i]) > (high[i-1] - low[i-1]) * 0.8 and
                      close[i+1] > open_[i+1]):
                    ob_blocks.append({
                        'type': 'bearish_ob',
                        'price_level': high[i],
                        'timestamp': df['time'].iloc[i],
                        'strength': (high[i] - low[i]) / low[i] * 100
                    })
            return ob_blocks
        except Exception as e:
            return []

    def detect_fvgs(self, df):
        """كشف Fair Value Gaps (FVG) - فجوات القيمة العادلة"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            fvgs = []
            for i in range(1, len(df)-1):
                # FVG صاعد: قاع الشمعة الحالية > قمة الشمعة السابقة
                if low[i] > high[i-1]:
                    fvgs.append({
                        'type': 'bullish_fvg',
                        'low': high[i-1],
                        'high': low[i],
                        'timestamp': df['time'].iloc[i]
                    })
                # FVG هابط: قمة الشمعة الحالية < قاع الشمعة السابقة
                elif high[i] < low[i-1]:
                    fvgs.append({
                        'type': 'bearish_fvg',
                        'low': high[i],
                        'high': low[i-1],
                        'timestamp': df['time'].iloc[i]
                    })
            return fvgs
        except Exception as e:
            return []

    def detect_bos_choch(self, df):
        """كشف Break of Structure (BOS) & Change of Character (CHoCH)"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # كسر الهيكل الصاعد: قمة جديدة أعلى من القمة السابقة
            bos_bullish = high.iloc[-1] > high.iloc[-3:-1].max() if len(df) > 3 else False
            # كسر الهيكل الهابط: قاع جديد أقل من القاع السابق
            bos_bearish = low.iloc[-1] < low.iloc[-3:-1].min() if len(df) > 3 else False
            
            # تغيير الطابع: انعكاس في الهيكل
            choch_bullish = (low.iloc[-1] > low.iloc[-2] and low.iloc[-2] < low.iloc[-3]) if len(df) > 3 else False
            choch_bearish = (high.iloc[-1] < high.iloc[-2] and high.iloc[-2] > high.iloc[-3]) if len(df) > 3 else False
            
            return {
                'bos_bullish': bos_bullish,
                'bos_bearish': bos_bearish,
                'choch_bullish': choch_bullish,
                'choch_bearish': choch_bearish
            }
        except Exception as e:
            return {}

    def analyze_liquidity(self, df, orderbook):
        """تحليل السيولة - السحب والضخ"""
        try:
            current_price = df['close'].iloc[-1]
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # تحديد مستويات السيولة القريبة
            recent_high = high.tail(20).max()
            recent_low = low.tail(20).min()
            
            liquidity_zones = {
                'above_liquidity': recent_high * 1.002,  # 0.2% فوق القمة
                'below_liquidity': recent_low * 0.998,   # 0.2% تحت القاع
                'liquidity_sweep_bullish': current_price > recent_high,
                'liquidity_sweep_bearish': current_price < recent_low
            }
            
            # دمج مع تحليل كتاب الطلبات
            if orderbook.get('ok'):
                buy_pressure = sum([qty for _, qty in orderbook.get('buy_walls', [])])
                sell_pressure = sum([qty for _, qty in orderbook.get('sell_walls', [])])
                liquidity_zones['orderbook_imbalance'] = buy_pressure / sell_pressure if sell_pressure > 0 else 1.0
            
            return liquidity_zones
        except Exception as e:
            return {}

# إنشاء محرك SMC الخارق
smc_engine = SuperSMCEngine()

# =================== PROFESSIONAL TRADE MANAGER ===================
class ProfessionalTradeManager:
    def __init__(self):
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0
        }
        
    def record_trade(self, side, entry, exit_price, quantity, profit, duration, reason=""):
        """تسجيل الصفقة مع أسباب النجاح/الفشل"""
        trade = {
            'timestamp': datetime.now(),
            'side': side,
            'entry': entry,
            'exit': exit_price,
            'quantity': quantity,
            'profit': profit,
            'duration': duration,
            'profit_pct': (profit / (entry * quantity)) * 100 if entry * quantity > 0 else 0,
            'reason': reason,
            'success': profit > 0
        }
        
        self.trade_history.append(trade)
        
        # تحديث المقاييس
        self.performance_metrics['total_trades'] += 1
        if profit > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['total_profit'] += profit
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['total_profit'] += profit
            
        # تحديث المتوسطات
        wins = [t['profit'] for t in self.trade_history if t['profit'] > 0]
        losses = [t['profit'] for t in self.trade_history if t['profit'] < 0]
        
        if wins:
            self.performance_metrics['average_win'] = sum(wins) / len(wins)
        if losses:
            self.performance_metrics['average_loss'] = abs(sum(losses) / len(losses))
    
    def get_win_rate(self):
        """الحصول على نسبة النجاح"""
        if self.performance_metrics['total_trades'] == 0:
            return 0.0
        return (self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']) * 100
    
    def analyze_trade_performance(self):
        """تحليل أداء التداول"""
        if not self.trade_history:
            return {"analysis": "لا توجد صفقات سابقة", "suggestions": []}
        
        recent_trades = self.trade_history[-10:]  # آخر 10 صفقات
        
        analysis = {
            "win_rate": self.get_win_rate(),
            "total_profit": self.performance_metrics['total_profit'],
            "avg_profit_per_trade": self.performance_metrics['total_profit'] / len(self.trade_history) if self.trade_history else 0,
            "suggestions": []
        }
        
        # تحليل الأخطاء الشائعة
        losing_trades = [t for t in recent_trades if not t['success']]
        if losing_trades:
            common_errors = []
            for trade in losing_trades:
                if "reversal" in trade.get('reason', '').lower():
                    common_errors.append("الدخول ضد الترند")
                elif "volatility" in trade.get('reason', '').lower():
                    common_errors.append("التداول في ظروف متقلبة")
                elif "spread" in trade.get('reason', '').lower():
                    common_errors.append("انتشار سعري مرتفع")
            
            if common_errors:
                analysis["common_errors"] = list(set(common_errors))
                analysis["suggestions"].append("تحسين توقيت الدخول وتجنب الأخطاء الشائعة")
        
        return analysis

# إنشاء مدير الصفقات المحترف
pro_trade_manager = ProfessionalTradeManager()

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): 
    print(f"ℹ️ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_g(msg): 
    print(f"✅ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_w(msg): 
    print(f"🟨 {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_e(msg): 
    print(f"❌ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_banner(text): 
    print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

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
            state = json.load(f)
        return state
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
    """إعداد التسجيل المهني"""
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
    
    log_i("🔄 Professional logging ready")

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
    """يرجع float من آخر عنصر"""
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

        if color is None:
            icon = "🟢" if side=="buy" else "🔴"
        else:
            icon = "🟢" if str(color).lower()=="green" else "🔴"

        log_i(f"{icon} {reason} — POSITION SNAPSHOT")
        log_i(f"SIDE: {side} | QTY: {_fmt(qty)} | ENTRY: {_fmt(px)} | LEV: {lev}× | MODE: {mode} | OPEN: {open_f}")
        log_i(f"TP1: {_pct(tp1)} | BE@: {_pct(be_a)} | TRAIL: act≥{_pct(trailA)}, ATR×{atrM} | SPREAD: {_fmt(spread,2)} bps")
        log_i(f"IND: {_ind_brief(ind)}")
        log_i(f"COUNCIL: {_council_brief(council)}")
        
        # إضافة تحليل الأداء
        performance = pro_trade_manager.analyze_trade_performance()
        log_i(f"PERFORMANCE: Win Rate: {performance.get('win_rate', 0):.1f}% | Total Profit: {performance.get('total_profit', 0):.2f}")
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
    if not MODE_LIVE: return 1000.0
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

# =================== UPDATED INDICATOR COMPUTATION ===================
def compute_indicators(df):
    """حساب المؤشرات بنمط TradingView/Bybit الدقيق"""
    try:
        if len(df) < 100:
            return {}
        
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # === RSI (TradingView style) ===
        rsi = tv.tv_rsi(close, RSI_LEN)
        rsi_ma = tv.tv_ema(rsi, RSI_MA_LEN)
        
        # === ADX + DI (TradingView style) ===
        adx, plus_di, minus_di = tv.tv_adx(high, low, close, ADX_LEN)
        di_spread = abs(plus_di - minus_di)
        
        # === ATR (TradingView style) ===
        atr = tv.tv_atr(high, low, close, ATR_LEN)
        
        # === MACD (TradingView style) ===
        macd, macd_signal, macd_hist = tv.tv_macd(close)
        
        return {
            'rsi': last_scalar(rsi),
            'rsi_ma': last_scalar(rsi_ma),
            'adx': last_scalar(adx),
            'plus_di': last_scalar(plus_di),
            'minus_di': last_scalar(minus_di),
            'di_spread': last_scalar(di_spread),
            'atr': last_scalar(atr),
            'macd': last_scalar(macd),
            'macd_signal': last_scalar(macd_signal),
            'macd_hist': last_scalar(macd_hist)
        }
    except Exception as e:
        log_w(f"TradingView indicators error: {e}")
        return {}

def compute_advanced_indicators(df):
    """حساب المؤشرات المتقدمة بنمط TradingView"""
    try:
        if len(df) < 100:
            return {}
            
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # المتوسطات المتحركة
        sma_20 = tv.tv_sma(close, 20)
        sma_50 = tv.tv_sma(close, 50)
        ema_20 = tv.tv_ema(close, 20)
        
        # بولينجر باندز
        bb_middle = tv.tv_sma(close, 20)
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # ستوكاستك
        stoch_k, stoch_d = tv.tv_stoch(high, low, close)
        
        # OBV
        obv = tv.tv_obv(close, volume)
        
        return {
            'sma_20': last_scalar(sma_20),
            'sma_50': last_scalar(sma_50),
            'ema_20': last_scalar(ema_20),
            'bollinger_upper': last_scalar(bb_upper),
            'bollinger_middle': last_scalar(bb_middle),
            'bollinger_lower': last_scalar(bb_lower),
            'stoch_k': last_scalar(stoch_k),
            'stoch_d': last_scalar(stoch_d),
            'obv': last_scalar(obv),
            'volume': last_scalar(volume)
        }
    except Exception as e:
        log_w(f"Advanced TradingView indicators error: {e}")
        return {}

def compute_candles(df):
    """تحليل الشموع"""
    try:
        open_ = df['open'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        score_buy = 0
        score_sell = 0
        pattern = ""
        
        # تحليل آخر 3 شموع
        for i in range(-3, 0):
            idx = i if i < 0 else len(df) + i
            if idx < 0: continue
            
            candle_body = abs(close.iloc[idx] - open_.iloc[idx])
            candle_range = high.iloc[idx] - low.iloc[idx]
            
            if candle_range == 0: continue
            
            body_ratio = candle_body / candle_range
            
            # شمعة صاعدة قوية
            if close.iloc[idx] > open_.iloc[idx] and body_ratio > 0.6:
                score_buy += 2
            # شمعة هابطة قوية
            elif close.iloc[idx] < open_.iloc[idx] and body_ratio > 0.6:
                score_sell += 2
            # دوجي (تردد)
            elif body_ratio < 0.1:
                # دوجي بعد ترند يمكن أن يكون انعكاس
                if idx > 0:
                    if close.iloc[idx-1] > open_.iloc[idx-1]:
                        score_sell += 1
                    else:
                        score_buy += 1
        
        return {
            'score_buy': score_buy,
            'score_sell': score_sell,
            'pattern': pattern
        }
    except Exception as e:
        return {'score_buy': 0, 'score_sell': 0, 'pattern': ''}

def golden_zone_check(df, indicators):
    """التحقق من المناطق الذهبية"""
    try:
        # تنفيذ مبسط للمناطق الذهبية
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        
        if (rsi < 30 and adx > 20) or (rsi > 70 and adx > 20):
            return {'ok': True, 'score': 7.0, 'zone': {'type': 'golden_bottom' if rsi < 30 else 'golden_top'}}
        return {'ok': False, 'score': 0}
    except Exception as e:
        return {'ok': False, 'score': 0}

# =================== ULTRA PROFESSIONAL COUNCIL AI ===================
def ultra_professional_council_ai(df):
    """
    مجلس الإدارة المحترف - يدمج كل استراتيجيات التداول المتقدمة مع مؤشرات TradingView الدقيقة
    """
    try:
        if len(df) < 100:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "trade_type": "scalp", "logs": []}
        
        # === تحليل SMC المتقدم ===
        order_blocks = smc_engine.detect_order_blocks(df)
        fvgs = smc_engine.detect_fvgs(df)
        bos_choch = smc_engine.detect_bos_choch(df)
        
        # === تحليل السوق المحترف ===
        supply_demand = pro_market_analyzer.detect_supply_demand_zones(df)
        manipulation_analysis = pro_market_analyzer.detect_manipulation_volatility(df)
        price_testing = pro_market_analyzer.analyze_price_testing(df)
        
        # === المؤشرات المتقدمة بنمط TradingView ===
        advanced_indicators = compute_advanced_indicators(df)
        basic_indicators = compute_indicators(df)
        indicators = {**basic_indicators, **advanced_indicators}
        
        # === تحليل إضافي ===
        candles = compute_candles(df)
        flow_metrics = compute_flow_metrics(df)
        orderbook = bookmap_snapshot(ex, SYMBOL)
        liquidity_analysis = smc_engine.analyze_liquidity(df, orderbook)
        
        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0
        logs = []
        confidence_factors = []
        
        current_price = float(df['close'].iloc[-1])
        
        # ===== 1. تحليل العرض والطلب =====
        # الطلب (دعم)
        for zone in supply_demand.get('demand_zones', []):
            if abs(zone['price'] - current_price) / current_price < 0.005:  # 0.5% قرب من المنطقة
                score_b += 3.0
                votes_b += 2
                logs.append(f"🛡️ Demand Zone - Strength: {zone['strength']:.1f}%")
                confidence_factors.append(1.8)
        
        # العرض (مقاومة)
        for zone in supply_demand.get('supply_zones', []):
            if abs(zone['price'] - current_price) / current_price < 0.005:  # 0.5% قرب من المنطقة
                score_s += 3.0
                votes_s += 2
                logs.append(f"🚧 Supply Zone - Strength: {zone['strength']:.1f}%")
                confidence_factors.append(1.8)
        
        # ===== 2. تحليل التلاعب والتذبذب =====
        if manipulation_analysis.get('high_volatility_alert'):
            # في التقلب العالي، نكون أكثر تحفظاً
            score_b *= 0.7
            score_s *= 0.7
            logs.append("⚡ High Volatility - Reduced Confidence")
        
        for signal in manipulation_analysis.get('manipulation_signals', []):
            if signal['direction'] == 'bullish':
                score_b += 1.5
                logs.append(f"🎭 Bullish Manipulation Detected")
            else:
                score_s += 1.5
                logs.append(f"🎭 Bearish Manipulation Detected")
        
        # ===== 3. تحليل اختبار السعر والاختراق =====
        if price_testing.get('testing_support'):
            score_b += 2.0
            votes_b += 1
            logs.append("📊 Price Testing Support - Potential Bounce")
        
        if price_testing.get('testing_resistance'):
            score_s += 2.0
            votes_s += 1
            logs.append("📊 Price Testing Resistance - Potential Rejection")
        
        if price_testing.get('breakout_confirmed'):
            if price_testing.get('breakout_strength') == 'strong':
                score_b += 3.0
                votes_b += 2
                logs.append("🚀 STRONG Breakout Confirmed")
                confidence_factors.append(2.0)
        
        if price_testing.get('breakdown_confirmed'):
            if price_testing.get('breakdown_strength') == 'strong':
                score_s += 3.0
                votes_s += 2
                logs.append("🚀 STRONG Breakdown Confirmed")
                confidence_factors.append(2.0)
        
        # ===== 4. تحليل SMC - ORDER BLOCKS =====
        current_obs = [ob for ob in order_blocks if abs(ob['price_level'] - current_price) / current_price < 0.01]
        for ob in current_obs:
            if ob['type'] == 'bullish_ob' and current_price >= ob['price_level']:
                score_b += 2.5
                votes_b += 2
                logs.append(f"🎯 Bullish OB Activated - Strength: {ob['strength']:.1f}%")
                confidence_factors.append(1.5)
            elif ob['type'] == 'bearish_ob' and current_price <= ob['price_level']:
                score_s += 2.5
                votes_s += 2
                logs.append(f"🎯 Bearish OB Activated - Strength: {ob['strength']:.1f}%")
                confidence_factors.append(1.5)
        
        # ===== 5. تحليل SMC - FVGs =====
        current_fvgs = [fvg for fvg in fvgs if fvg['low'] <= current_price <= fvg['high']]
        for fvg in current_fvgs:
            if fvg['type'] == 'bullish_fvg':
                score_b += 2.0
                votes_b += 1
                logs.append("📊 Bullish FVG Zone")
            elif fvg['type'] == 'bearish_fvg':
                score_s += 2.0
                votes_s += 1
                logs.append("📊 Bearish FVG Zone")
        
        # ===== 6. تحليل BOS/CHoCH =====
        if bos_choch.get('bos_bullish'):
            score_b += 3.0
            votes_b += 3
            logs.append("🚀 BULLISH Break of Structure")
            confidence_factors.append(2.0)
        if bos_choch.get('bos_bearish'):
            score_s += 3.0
            votes_s += 3
            logs.append("🚀 BEARISH Break of Structure")
            confidence_factors.append(2.0)
        if bos_choch.get('choch_bullish'):
            score_b += 2.5
            votes_b += 2
            logs.append("🔄 BULLISH Change of Character")
        if bos_choch.get('choch_bearish'):
            score_s += 2.5
            votes_s += 2
            logs.append("🔄 BEARISH Change of Character")
        
        # ===== 7. تحليل السيولة =====
        if liquidity_analysis.get('liquidity_sweep_bullish'):
            score_b += 2.0
            votes_b += 2
            logs.append("💧 Bullish Liquidity Sweep")
        if liquidity_analysis.get('liquidity_sweep_bearish'):
            score_s += 2.0
            votes_s += 2
            logs.append("💧 Bearish Liquidity Sweep")
        
        # ===== 8. تحليل كتاب الطلبات =====
        if orderbook.get('ok'):
            imbalance = orderbook.get('imbalance', 1.0)
            if imbalance > 2.0:
                score_b += 2.0
                votes_b += 2
                logs.append(f"📚 Strong Buy Pressure (Imb: {imbalance:.2f})")
            elif imbalance < 0.5:
                score_s += 2.0
                votes_s += 2
                logs.append(f"📚 Strong Sell Pressure (Imb: {imbalance:.2f})")
        
        # ===== 9. تحليل التدفق =====
        if flow_metrics.get('ok'):
            delta_z = flow_metrics.get('delta_z', 0)
            cvd_trend = flow_metrics.get('cvd_trend', '')
            
            if delta_z > 2.0 and cvd_trend == 'up':
                score_b += 2.5
                votes_b += 2
                logs.append(f"🌊 Strong Buy Flow (z: {delta_z:.2f})")
                confidence_factors.append(1.5)
            elif delta_z < -2.0 and cvd_trend == 'down':
                score_s += 2.5
                votes_s += 2
                logs.append(f"🌊 Strong Sell Flow (z: {delta_z:.2f})")
                confidence_factors.append(1.5)
        
        # ===== 10. المؤشرات التقليدية المحسنة (TradingView Style) =====
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        adx = indicators.get('adx', 0)
        atr = indicators.get('atr', 0.001)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        # RSI مع مستويات متقدمة (TradingView Style)
        if rsi < 25:
            score_b += 2.5
            votes_b += 2
            logs.append("📊 RSI Oversold (Strong Level)")
        elif rsi > 75:
            score_s += 2.5
            votes_s += 2
            logs.append("📊 RSI Overbought (Strong Level)")
        elif 30 < rsi < 40:
            score_b += 1.0
            logs.append("📊 RSI Near Oversold")
        elif 60 < rsi < 70:
            score_s += 1.0
            logs.append("📊 RSI Near Overbought")
        
        # MACD مع تأكيد متقدم (TradingView Style)
        if macd > macd_signal and indicators.get('macd_hist', 0) > 0:
            score_b += 1.5
            votes_b += 1
            logs.append("📈 MACD Bullish Cross")
        elif macd < macd_signal and indicators.get('macd_hist', 0) < 0:
            score_s += 1.5
            votes_s += 1
            logs.append("📉 MACD Bearish Cross")
        
        # ADX مع تحليل القوة المتقدم (TradingView Style)
        if adx > 25:
            if plus_di > minus_di:
                score_b += 2.0
                votes_b += 2
                logs.append(f"💪 Strong Uptrend - ADX: {adx:.1f}, +DI: {plus_di:.1f}")
                confidence_factors.append(1.5)
            else:
                score_s += 2.0
                votes_s += 2
                logs.append(f"💪 Strong Downtrend - ADX: {adx:.1f}, -DI: {minus_di:.1f}")
                confidence_factors.append(1.5)
        elif adx < 15:
            # سوق جانبي - تقليل الثقة في الصفقات الاتجاهية
            score_b *= 0.7
            score_s *= 0.7
            logs.append("🔄 Range Market - Reduced Directional Confidence")
        
        # ===== 11. تحديد نوع الصفقة =====
        if adx > 22 and abs(score_b - score_s) > 8 and (price_testing.get('breakout_confirmed') or price_testing.get('breakdown_confirmed')):
            trade_type = "trend"
            logs.append("🎯 TRADE TYPE: TREND (Strong directional move)")
        else:
            trade_type = "scalp"
            logs.append("🎯 TRADE TYPE: SCALP (Range or weak trend)")
        
        # ===== 12. تطبيق عوامل الثقة =====
        if confidence_factors:
            confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
            score_b *= confidence_multiplier
            score_s *= confidence_multiplier
        
        # ===== 13. حساب الثقة النهائية =====
        total_score = score_b + score_s
        max_possible_score = 50.0
        confidence = min(1.0, total_score / max_possible_score)
        
        # ===== 14. الحد الأدنى للثقة المحسنة =====
        min_confidence = 0.75  # 75% حد أدنى للجودة العالية
        if confidence < min_confidence:
            score_b *= 0.2  # تخفيض شديد للإشارات الضعيفة
            score_s *= 0.2
            logs.append(f"🛡️ Low Confidence ({confidence:.2f}) - Strong Signal Reduction")
        
        # ===== 15. مراعاة أداء التداول السابق =====
        performance = pro_trade_manager.analyze_trade_performance()
        if performance.get('win_rate', 0) < 40:
            # إذا كانت نسبة النجاح منخفضة، نكون أكثر تحفظاً
            score_b *= 0.8
            score_s *= 0.8
            logs.append("⚠️ Low Win Rate - Being More Conservative")
        
        return {
            "b": votes_b,
            "s": votes_s,
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "confidence": round(confidence, 2),
            "trade_type": trade_type,
            "logs": logs,
            "analysis": {
                "supply_demand": supply_demand,
                "manipulation": manipulation_analysis,
                "price_testing": price_testing,
                "smc": {
                    "order_blocks": len(current_obs),
                    "fvgs": len(current_fvgs),
                    "bos_choch": bos_choch
                }
            },
            "indicators": indicators
        }
        
    except Exception as e:
        log_e(f"Professional council error: {e}")
        return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "trade_type": "scalp", "logs": [f"Error: {e}"]}

# =================== DYNAMIC PROFIT MANAGEMENT ===================
class DynamicProfitManager:
    def __init__(self):
        self.profit_levels = []
        
    def calculate_dynamic_tps(self, trade_type, atr, entry_price, side, market_strength):
        """حساب أهداف ربح ديناميكية محترفة"""
        if trade_type == "trend":
            if market_strength == "strong":
                # أهداف كبيرة للترند القوي
                multipliers = [0.8, 1.5, 2.5, 4.0, 6.0, 8.0]
                close_fractions = [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]
            else:
                # أهداف متوسطة للترند العادي
                multipliers = [0.6, 1.2, 2.0, 3.0, 4.5]
                close_fractions = [0.15, 0.20, 0.25, 0.20, 0.20]
        else:
            # أهداف سكالب سريعة
            multipliers = [0.3, 0.6, 1.0]
            close_fractions = [0.40, 0.40, 0.20]
        
        tps = []
        for i, mult in enumerate(multipliers):
            if side == "long":
                tp_price = entry_price * (1 + mult / 100)
            else:
                tp_price = entry_price * (1 - mult / 100)
            
            tps.append({
                'level': i + 1,
                'price': tp_price,
                'close_fraction': close_fractions[i],
                'target_pct': mult,
                'atr_distance': mult / (atr / entry_price * 100) if atr > 0 else 0
            })
        
        return tps
    
    def update_trailing_stop(self, current_price, highest_profit, atr, trade_type, side):
        """تحديث الوقف المتحرك الذكي المحترف"""
        if trade_type == "trend":
            activation = 1.0  # 1% ربح لتفعيل الترند
            multiplier = 2.0  # ATR مضاعف كبير للترند
        else:
            activation = 0.5  # 0.5% ربح لتفعيل السكالب
            multiplier = 1.2  # ATR مضاعف ضيق للسكالب
        
        if highest_profit >= activation:
            if not STATE.get("trail_active", False):
                STATE["trail_active"] = True
                log_i("🎯 Professional Trailing Stop Activated")
            
            trail_distance = atr * multiplier
            if side == "long":
                new_trail = current_price - trail_distance
                if STATE.get("trail") is None or new_trail > STATE["trail"]:
                    STATE["trail"] = new_trail
                    if STATE["trail"] > STATE.get("entry", 0):
                        log_i(f"🔼 Professional trail updated: {STATE['trail']:.6f}")
            else:
                new_trail = current_price + trail_distance
                if STATE.get("trail") is None or new_trail < STATE["trail"]:
                    STATE["trail"] = new_trail
                    if STATE["trail"] < STATE.get("entry", float('inf')):
                        log_i(f"🔽 Professional trail updated: {STATE['trail']:.6f}")

# إنشاء مدير الأرباح الديناميكي
profit_manager = DynamicProfitManager()

# =================== PROFESSIONAL TRADE EXECUTION ===================
def execute_professional_trade(side, price, qty, council_data, market_analysis):
    """تنفيذ صفقة محترفة مع تحليل متقدم"""
    try:
        if not EXECUTE_ORDERS or DRY_RUN:
            log_i(f"DRY_RUN: {side.upper()} {qty:.4f} @ {price:.6f}")
            log_i(f"TRADE TYPE: {council_data.get('trade_type', 'scalp')}")
            return True
        
        if qty <= 0:
            log_e("❌ كمية غير صالحة للتنفيذ")
            return False
        
        trade_type = council_data.get('trade_type', 'scalp')
        
        log_i(f"🎯 PROFESSIONAL TRADE EXECUTION:")
        log_i(f"   SIDE: {side.upper()}")
        log_i(f"   TYPE: {trade_type.upper()}")
        log_i(f"   QTY: {qty:.4f}")
        log_i(f"   PRICE: {price:.6f}")
        log_i(f"   CONFIDENCE: {council_data.get('confidence', 0):.2f}")
        
        # عرض أسباب الدخول المفصلة
        log_i(f"   📋 ENTRY REASONS:")
        for i, log_msg in enumerate(council_data.get('logs', [])[-10:]):
            log_i(f"      {i+1}. {log_msg}")
        
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ PROFESSIONAL TRADE EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        # تسجيل الصفقة مع الأسباب
        entry_reason = " | ".join(council_data.get('logs', [])[-5:])
        pro_trade_manager.record_trade(
            side=side,
            entry=price,
            exit_price=price,
            quantity=qty,
            profit=0.0,
            duration=0,
            reason=entry_reason
        )
        
        return True
        
    except Exception as e:
        log_e(f"❌ PROFESSIONAL TRADE EXECUTION FAILED: {e}")
        return False

def compute_adaptive_position_size(balance, price, confidence, market_phase):
    """حساب حجم صفقة متكيف محترف"""
    base_size = (balance * RISK_ALLOC) / price
    
    # تعديل الحجم بناءً على الثقة
    confidence_multiplier = 0.6 + (confidence * 0.4)  # 0.6 إلى 1.0
    
    # تعديل الحجم بناءً على مرحلة السوق
    if market_phase in ["strong_bull", "strong_bear"]:
        market_multiplier = 1.3
    elif market_phase in ["bull", "bear"]:
        market_multiplier = 1.1
    else:
        market_multiplier = 0.8
    
    adaptive_size = base_size * confidence_multiplier * market_multiplier
    
    # التأكد من أن الحجم ضمن الحدود المعقولة
    max_position = balance * LEVERAGE * 0.8
    final_size = min(adaptive_size, max_position / price) if price > 0 else adaptive_size
    
    log_i(f"📊 PROFESSIONAL POSITION SIZING:")
    log_i(f"   Base: {base_size:.4f}")
    log_i(f"   Confidence Multiplier: {confidence_multiplier:.2f}")
    log_i(f"   Market Multiplier: {market_multiplier:.2f}")
    log_i(f"   Final: {final_size:.4f}")
    
    return safe_qty(final_size)

# =================== PROFESSIONAL POSITION MANAGEMENT ===================
def manage_professional_position(df, council_data, current_price):
    """إدارة محترفة للمراكز المفتوحة"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    try:
        entry_price = STATE["entry"]
        side = STATE["side"]
        qty = STATE["qty"]
        trade_type = STATE.get("trade_type", "scalp")
        
        # حساب الربح/الخسارة
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        STATE["pnl"] = pnl_pct
        
        if pnl_pct > STATE["highest_profit_pct"]:
            STATE["highest_profit_pct"] = pnl_pct
        
        # تحديث الوقف المتحرك المحترف
        atr = council_data.get('indicators', {}).get('atr', 0) or 0.001
        profit_manager.update_trailing_stop(current_price, STATE["highest_profit_pct"], atr, trade_type, side)
        
        # تحقيق أهداف الربح الديناميكية
        profit_targets = STATE.get("profit_targets", [])
        for target in profit_targets:
            target_key = f"tp_{target['level']}_done"
            if not STATE.get(target_key, False):
                if (side == "long" and current_price >= target['price']) or \
                   (side == "short" and current_price <= target['price']):
                    
                    close_qty = safe_qty(qty * target['close_fraction'])
                    if close_qty > 0:
                        close_side = "sell" if side == "long" else "buy"
                        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                            try:
                                params = exchange_specific_params(close_side, is_close=True)
                                ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                                log_g(f"🎯 PROFESSIONAL TP{target['level']} HIT: {target['target_pct']:.1f}% - Closed {close_qty:.4f}")
                                STATE["qty"] = safe_qty(qty - close_qty)
                                STATE[target_key] = True
                                STATE["profit_targets_achieved"] += 1
                                
                                # تسجيل الربح الجزئي
                                partial_profit = (abs(current_price - entry_price) * close_qty) * (1 if side == "long" else -1)
                                pro_trade_manager.record_trade(
                                    side=side,
                                    entry=entry_price,
                                    exit_price=current_price,
                                    quantity=close_qty,
                                    profit=partial_profit,
                                    duration=0,
                                    reason=f"TP{target['level']} Target Hit"
                                )
                            except Exception as e:
                                log_e(f"❌ PROFESSIONAL PARTIAL CLOSE FAILED: {e}")
        
        # التحقق من الوقف المتحرك
        if STATE.get("trail_active") and STATE.get("trail"):
            if (side == "long" and current_price <= STATE["trail"]) or \
               (side == "short" and current_price >= STATE["trail"]):
                log_i(f"🔴 PROFESSIONAL TRAILING STOP HIT: {STATE['trail']:.6f}")
                close_reason = "trailing_stop"
                close_market_strict(close_reason)
                return
        
        # إغلاق ذكي بناءً على إشارات الانعكاس القوية
        if should_close_from_professional_signals(council_data, side, pnl_pct, trade_type):
            close_reason = "professional_reversal_signal"
            log_i(f"🔴 PROFESSIONAL REVERSAL SIGNAL - Closing Position")
            close_market_strict(close_reason)
            return
        
        # إغلاق وقائي عند فقدان الزخم
        if should_close_from_momentum_loss(council_data, side, pnl_pct, trade_type):
            close_reason = "momentum_loss"
            log_i(f"🔴 MOMENTUM LOSS DETECTED - Closing Position")
            close_market_strict(close_reason)
            return
        
    except Exception as e:
        log_e(f"❌ PROFESSIONAL POSITION MANAGEMENT ERROR: {e}")

def should_close_from_professional_signals(council_data, current_side, pnl_pct, trade_type):
    """تحديد إذا كان يجب الإغلاق بناءً على إشارات الانعكاس المحترفة"""
    try:
        opposite_side_score = council_data["score_s"] if current_side == "long" else council_data["score_b"]
        current_side_score = council_data["score_b"] if current_side == "long" else council_data["score_s"]
        
        # إذا كانت إشارات الجانب المعاكس أقوى بكثير مع ثقة عالية
        if (opposite_side_score > current_side_score * 2.0 and 
            council_data["confidence"] > 0.8 and
            opposite_side_score > 20):
            return True
        
        # إذا كان هناك ربح جيد وإشارات انعكاس قوية
        if (pnl_pct > (3.0 if trade_type == "scalp" else 6.0) and 
            opposite_side_score > 15 and
            council_data["confidence"] > 0.75):
            return True
        
        # كشف الاختراقات الوهمية ضد المركز الحالي
        fake_breakouts = council_data.get('analysis', {}).get('manipulation', {}).get('fake_breakouts', [])
        current_price = price_now()
        for breakout in fake_breakouts:
            if (current_side == "long" and breakout['type'] == 'fake_breakout_high' and
                abs(breakout['level'] - current_price) / current_price < 0.005):
                return True
            elif (current_side == "short" and breakout['type'] == 'fake_breakout_low' and
                  abs(breakout['level'] - current_price) / current_price < 0.005):
                return True
        
        return False
    except Exception as e:
        log_w(f"Professional close signal error: {e}")
        return False

def should_close_from_momentum_loss(council_data, current_side, pnl_pct, trade_type):
    """كشف فقدان الزخم والإغلاق الوقائي"""
    try:
        indicators = council_data.get('indicators', {})
        adx = indicators.get('adx', 0)
        rsi = indicators.get('rsi', 50)
        macd_hist = indicators.get('macd_hist', 0)
        
        # فقدان الزخم في الترند
        if trade_type == "trend" and adx < 20 and pnl_pct > 2.0:
            return True
        
        # RSI في المنطقة المحايدة بعد حركة قوية
        if (pnl_pct > 4.0 and 40 < rsi < 60 and 
            abs(macd_hist) < 0.001):
            return True
        
        # إذا كان المركز مفتوحاً لفترة طويلة بدون تقدم
        if STATE.get("bars", 0) > 20 and pnl_pct < 1.0:
            return True
            
        return False
    except Exception as e:
        return False

def close_market_strict(reason=""):
    """إغلاق صارم للمركز مع تحليل السبب"""
    try:
        if not STATE["open"] or STATE["qty"] <= 0:
            return True

        side = STATE["side"]
        qty = STATE["qty"]
        close_side = "sell" if side == "long" else "buy"
        current_price = price_now()

        log_i(f"🔒 PROFESSIONAL CLOSE: {side.upper()} {qty:.4f} - Reason: {reason}")

        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            params = exchange_specific_params(close_side, is_close=True)
            ex.create_order(SYMBOL, "market", close_side, qty, None, params)
        
        # حساب الربح النهائي
        entry_price = STATE["entry"]
        if current_price and entry_price:
            if side == "long":
                profit = (current_price - entry_price) * qty
            else:
                profit = (entry_price - current_price) * qty
        else:
            profit = 0.0

        # تسجيل الصفقة المغلقة
        pro_trade_manager.record_trade(
            side=side,
            entry=entry_price,
            exit_price=current_price,
            quantity=qty,
            profit=profit,
            duration=0,
            reason=f"CLOSED: {reason}"
        )

        # إعادة تعيين الحالة
        STATE.update({
            "open": False, "side": None, "entry": None, "qty": 0.0,
            "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
            "tp1_done": False, "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0, "trail_active": False,
            "trade_type": None, "profit_targets": []
        })

        save_state({
            "in_position": False,
            "closed_at": int(time.time()),
            "close_reason": reason,
            "final_profit": profit
        })

        log_g(f"✅ PROFESSIONAL CLOSE COMPLETED: {reason} | PnL: {profit:.4f} USDT")
        
        # تحليل الصفقة المغلقة
        if profit > 0:
            log_g(f"🎉 SUCCESSFUL TRADE! Profit: {profit:.4f} USDT")
        else:
            log_w(f"📉 UNSUCCESSFUL TRADE! Loss: {abs(profit):.4f} USDT")

        return True

    except Exception as e:
        log_e(f"❌ PROFESSIONAL CLOSE FAILED: {e}")
        return False

# =================== PROFESSIONAL TRADING LOOP ===================
def professional_trading_loop():
    """الحلقة الرئيسية للتداول المحترف"""
    global wait_for_next_signal_side
    
    log_banner("STARTING ULTIMATE PROFESSIONAL TRADING BOT - TRADINGVIEW PRECISION")
    log_i(f"🤖 Bot Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    log_i(f"📊 Risk Allocation: {RISK_ALLOC*100}%")
    log_i(f"🎯 Indicators: TradingView/Bybit Precision Mode")
    
    # عرض إحصائيات البداية
    performance = pro_trade_manager.analyze_trade_performance()
    log_i(f"📈 Historical Performance: Win Rate: {performance.get('win_rate', 0):.1f}% | Total Profit: {performance.get('total_profit', 0):.2f} USDT")
    
    while True:
        try:
            # جمع البيانات الأساسية
            balance = balance_usdt()
            current_price = price_now()
            df = fetch_ohlcv(limit=500)  # زيادة الحد لضمان دقة المؤشرات
            
            if df.empty or current_price is None:
                log_w("📭 No data available - retrying...")
                time.sleep(BASE_SLEEP)
                continue
            
            # قرار مجلس الإدارة المحترف
            council_data = ultra_professional_council_ai(df)
            
            # تحديث الحالة
            STATE["last_council"] = council_data
            STATE["last_ind"] = council_data.get("indicators", {})
            STATE["last_spread_bps"] = orderbook_spread_bps()
            
            # عرض معلومات السوق المحترفة
            if LOG_ADDONS:
                log_i(f"🏪 MARKET ANALYSIS:")
                log_i(f"   Phase: {council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength', 'neutral').upper()}")
                log_i(f"   Trade Type: {council_data.get('trade_type', 'scalp').upper()}")
                log_i(f"   Confidence: {council_data['confidence']:.2f}")
                
                log_i(f"🎯 COUNCIL DECISION:")
                log_i(f"   Votes: B{council_data['b']}/S{council_data['s']}")
                log_i(f"   Scores: {council_data['score_b']:.1f}/{council_data['score_s']:.1f}")
                
                # عرض تحليل SMC
                smc_info = council_data.get('analysis', {}).get('smc', {})
                log_i(f"🔧 SMC ANALYSIS:")
                log_i(f"   Order Blocks: {smc_info.get('order_blocks', 0)}")
                log_i(f"   FVGs: {smc_info.get('fvgs', 0)}")
                log_i(f"   BOS: {'Y' if smc_info.get('bos_choch', {}).get('bos_bullish') or smc_info.get('bos_choch', {}).get('bos_bearish') else 'N'}")
                log_i(f"   CHoCH: {'Y' if smc_info.get('bos_choch', {}).get('choch_bullish') or smc_info.get('bos_choch', {}).get('choch_bearish') else 'N'}")
                
                # عرض أهم أسباب القرار
                log_i(f"📋 KEY REASONS:")
                for i, log_msg in enumerate(council_data.get("logs", [])[-5:]):
                    log_i(f"   {i+1}. {log_msg}")
            
            # إدارة المركز المفتوح
            if STATE["open"]:
                STATE["bars"] += 1
                manage_professional_position(df, council_data, current_price)
            
            # فتح صفقات جديدة
            if not STATE["open"]:
                signal_side = None
                signal_reason = ""
                
                # شروط دخول محترفة - عتبات عالية للجودة
                min_score = 18.0  # حد أدنى عالي للجودة
                min_confidence = 0.78  # ثقة عالية جداً
                
                if (council_data["score_b"] > council_data["score_s"] and 
                    council_data["score_b"] >= min_score and 
                    council_data["confidence"] >= min_confidence):
                    signal_side = "buy"
                    signal_reason = f"PROFESSIONAL BUY (Score: {council_data['score_b']:.1f}, Confidence: {council_data['confidence']:.2f})"
                elif (council_data["score_s"] > council_data["score_b"] and 
                      council_data["score_s"] >= min_score and 
                      council_data["confidence"] >= min_confidence):
                    signal_side = "sell"
                    signal_reason = f"PROFESSIONAL SELL (Score: {council_data['score_s']:.1f}, Confidence: {council_data['confidence']:.2f})"
                
                # فتح الصفقة إذا كانت الإشارة قوية جداً
                if signal_side:
                    position_size = compute_adaptive_position_size(
                        balance, current_price, council_data["confidence"], 
                        council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength', 'neutral')
                    )
                    
                    if position_size > 0:
                        log_i(f"🎯 PROFESSIONAL TRADE SIGNAL DETECTED:")
                        log_i(f"   Direction: {signal_side.upper()}")
                        log_i(f"   Type: {council_data.get('trade_type', 'scalp').upper()}")
                        log_i(f"   Size: {position_size:.4f}")
                        log_i(f"   Price: {current_price:.6f}")
                        log_i(f"   Confidence: {council_data['confidence']:.2f}")
                        log_i(f"   Reason: {signal_reason}")
                        
                        success = execute_professional_trade(
                            signal_side, current_price, position_size, council_data, {
                                "market_phase": council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength', 'neutral'),
                                "volatility": council_data.get('analysis', {}).get('manipulation', {}).get('current_volatility', 0)
                            }
                        )
                        
                        if success:
                            STATE.update({
                                "open": True,
                                "side": "long" if signal_side == "buy" else "short",
                                "entry": current_price,
                                "qty": position_size,
                                "pnl": 0.0,
                                "bars": 0,
                                "trail": None,
                                "breakeven": None,
                                "highest_profit_pct": 0.0,
                                "profit_targets_achieved": 0,
                                "trade_type": council_data.get('trade_type', 'scalp'),
                                "entry_reason": signal_reason
                            })
                            
                            # حساب أهداف الربح الديناميكية
                            atr = council_data.get('indicators', {}).get('atr', 0) or 0.001
                            market_strength = "strong" if council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength') == 'strong' else "normal"
                            STATE["profit_targets"] = profit_manager.calculate_dynamic_tps(
                                STATE["trade_type"], atr, current_price, STATE["side"], market_strength
                            )
                            
                            save_state({
                                "in_position": True,
                                "side": signal_side.upper(),
                                "entry_price": current_price,
                                "position_qty": position_size,
                                "opened_at": int(time.time()),
                                "trade_type": STATE["trade_type"],
                                "entry_reason": signal_reason
                            })
                            
                            print_position_snapshot("PROFESSIONAL_OPEN")
            
            # الانتظار للدورة التالية
            sleep_time = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            log_e(f"❌ PROFESSIONAL TRADING LOOP ERROR: {e}")
            log_e(traceback.format_exc())
            time.sleep(BASE_SLEEP * 2)

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "trail_active": False,
    "trade_type": None, "profit_targets": []
}

# =================== FLASK API (مبسط) ===================
app = Flask(__name__)

@app.route("/")
def home():
    return f"""
    <html>
        <head><title>SUI ULTRA PRO AI BOT</title></head>
        <body>
            <h1>🚀 SUI ULTRA PRO AI BOT - الإصدار المحترف مع مؤشرات TradingView</h1>
            <p><strong>Version:</strong> {BOT_VERSION}</p>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
            <p><strong>Symbol:</strong> {SYMBOL}</p>
            <p><strong>Status:</strong> {'🟢 LIVE' if MODE_LIVE else '🟡 PAPER'}</p>
            <p><strong>Position:</strong> {'🟢 OPEN' if STATE['open'] else '🔴 CLOSED'}</p>
            <p><strong>Indicators:</strong> TradingView/Bybit Precision Mode</p>
        </body>
    </html>
    """

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "position_open": STATE["open"],
        "indicators_mode": "TradingView Precision"
    })

@app.route("/performance")
def performance():
    performance_data = pro_trade_manager.analyze_trade_performance()
    return jsonify(performance_data)

# =================== STARTUP ===================
def startup_sequence():
    """تسلسل بدء التشغيل"""
    log_banner("PROFESSIONAL SYSTEM INITIALIZATION - TRADINGVIEW PRECISION")
    
    # تحميل الحالة السابقة
    loaded_state = load_state()
    if loaded_state:
        log_g("✅ Previous state loaded successfully")
        if loaded_state.get("in_position"):
            log_i(f"🔄 Resuming existing position: {loaded_state.get('side')} {loaded_state.get('position_qty')} @ {loaded_state.get('entry_price')}")
    
    # التحقق من اتصال البورصة
    try:
        balance = balance_usdt()
        price = price_now()
        log_g(f"✅ Exchange connection successful")
        log_g(f"💰 Balance: {balance:.2f} USDT")
        log_g(f"💰 Current price: {price:.6f}")
    except Exception as e:
        log_e(f"❌ Exchange connection failed: {e}")
        return False
    
    # عرض إحصائيات البداية
    performance = pro_trade_manager.analyze_trade_performance()
    log_i(f"📊 Historical Performance:")
    log_i(f"   Win Rate: {performance.get('win_rate', 0):.1f}%")
    log_i(f"   Total Profit: {performance.get('total_profit', 0):.2f} USDT")
    log_i(f"   Total Trades: {performance.get('total_trades', 0)}")
    
    if performance.get('suggestions'):
        log_i(f"💡 Suggestions: {', '.join(performance['suggestions'])}")
    
    log_g("🚀 ULTIMATE PROFESSIONAL TRADING BOT READY! - TRADINGVIEW PRECISION ACTIVE")
    return True

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    # إعداد معالجات الإشارات
    def signal_handler(signum, frame):
        log_i(f"🛑 Received signal {signum} - Shutting down gracefully...")
        save_state(STATE)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # بدء التشغيل
    if startup_sequence():
        # بدء خيوط التنفيذ
        import threading
        
        # خيط التداول الرئيسي
        trading_thread = threading.Thread(target=professional_trading_loop, daemon=True)
        trading_thread.start()
        
        # خيط الحفاظ على الحالة
        def state_saver():
            while True:
                time.sleep(300)  # حفظ كل 5 دقائق
                save_state(STATE)
        
        state_thread = threading.Thread(target=state_saver, daemon=True)
        state_thread.start()
        
        log_g(f"🌐 Starting web server on port {PORT}")
        
        # تشغيل سيرفل الويب
        try:
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ Web server failed: {e}")
    else:
        log_e("❌ Startup failed - check configuration and try again")
