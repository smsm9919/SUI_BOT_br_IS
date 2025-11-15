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
• نظام كشف التذبذب المتقدم
• نظام التسجيل المحترف المتكامل
• نظام توفير الموارد المحسن
• نظام سكالب محسن مع إشارات قوية وحجم صفقات مثالي
"""

import os, time, math, random, signal, sys, traceback, logging, json, gc
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

# =================== إعدادات توفير الموارد ===================
RESOURCE_SAVER_MODE = True  # 🔽 وضع توفير الموارد
MIN_CANDLES = 180           # 🔽 أقل عدد شموع للمؤشرات (كان 500)
BASE_SLEEP = 12             # 🔽 زيادة فترة الانتظار (كان 5)
NEAR_CLOSE_S = 3            # 🔽 زيادة قليلاً near close (كان 1)
MAX_LOOP_FREQUENCY = 18     # 🔽 أقصى تردد للمسح (ثانية)

# ====== POSITION SIZING CONFIG ======
# أقل كمية وخطوة تقريب لكل رمز (مظبوط لـ SUI)
MIN_QTY_BY_SYMBOL = {
    "SUI/USDT:USDT": 1.0,
    "SUI/USDT": 1.0,
    "SUIUSDT": 1.0,
}

# خطوة التقريب لكل رمز (SUI مسموح بعُشر عملة)
QTY_STEP_BY_SYMBOL = {
    "SUI/USDT:USDT": 0.1,
    "SUI/USDT": 0.1, 
    "SUIUSDT": 0.1,
}

DEFAULT_MIN_QTY = 1.0
DEFAULT_QTY_STEP = 0.1

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

# =================== ADVANCED VOLATILITY DETECTOR ===================
class AdvancedVolatilityDetector:
    """كاشف التذبذب المتقدم للحماية من الأسواق المتقلبة"""
    
    def __init__(self):
        self.volatility_history = []
        self.high_volatility_periods = []
        self.last_alert_time = 0
        self.cooldown_period = 300  # 5 دقائق بين التنبيهات
        
    def calculate_volatility_metrics(self, df):
        """حساب مقاييس التذبذب المتقدمة"""
        try:
            if len(df) < 50:
                return {"volatility_level": "low", "atr_ratio": 0, "price_oscillation": 0, "recommendation": "insufficient_data"}
            
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            # 1. نسبة ATR (المدى الحقيقي)
            atr = tv.tv_atr(high, low, close, 14)
            current_atr = atr.iloc[-1]
            avg_atr = atr.tail(50).mean()
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            # 2. تذبذب السعر (الانحراف المعياري للنسبة المئوية للتغير)
            price_changes = close.pct_change().dropna()
            price_volatility = price_changes.rolling(20).std().iloc[-1] * 100  # كنسبة مئوية
            
            # 3. نطاق بولينجر (عرض النطاق)
            bb_upper, bb_middle, bb_lower = tv.tv_bollinger_bands(close, 20, 2)
            bb_width = ((bb_upper - bb_lower) / bb_middle * 100).iloc[-1]
            
            # 4. تقلبات RSI
            rsi = tv.tv_rsi(close, 14)
            rsi_volatility = rsi.rolling(14).std().iloc[-1]
            
            # 5. حجم التقلبات النسبي
            volume = df['volume'].astype(float)
            volume_avg = volume.tail(20).mean()
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1.0
            
            # تحديد مستوى التذبذب
            volatility_score = 0
            volatility_score += 2.0 if atr_ratio > 1.8 else 1.0 if atr_ratio > 1.3 else 0
            volatility_score += 2.0 if price_volatility > 2.5 else 1.0 if price_volatility > 1.5 else 0
            volatility_score += 1.5 if bb_width > 6.0 else 0.5 if bb_width > 4.0 else 0
            volatility_score += 1.0 if rsi_volatility > 15 else 0
            volatility_score += 0.5 if volume_ratio > 2.0 else 0
            
            if volatility_score >= 5:
                volatility_level = "extreme"
                recommendation = "NO_TRADING"
                color = "🔴"
            elif volatility_score >= 3:
                volatility_level = "high"
                recommendation = "AVOID_NEW_TRADES"
                color = "🟡"
            elif volatility_score >= 1.5:
                volatility_level = "medium"
                recommendation = "CAUTION"
                color = "🟠"
            else:
                volatility_level = "low"
                recommendation = "SAFE"
                color = "🟢"
            
            # تسجيل فترة التذبذب العالي
            current_time = time.time()
            if volatility_level in ["high", "extreme"]:
                self.high_volatility_periods.append({
                    'timestamp': current_time,
                    'level': volatility_level,
                    'score': volatility_score,
                    'metrics': {
                        'atr_ratio': round(atr_ratio, 2),
                        'price_volatility': round(price_volatility, 2),
                        'bb_width': round(bb_width, 2),
                        'rsi_volatility': round(rsi_volatility, 2),
                        'volume_ratio': round(volume_ratio, 2)
                    }
                })
                # الاحتفاظ بسجل آخر 20 فترة فقط
                self.high_volatility_periods = self.high_volatility_periods[-20:]
            
            result = {
                "volatility_level": volatility_level,
                "volatility_score": round(volatility_score, 2),
                "atr_ratio": round(atr_ratio, 2),
                "price_volatility": round(price_volatility, 2),
                "bb_width": round(bb_width, 2),
                "rsi_volatility": round(rsi_volatility, 2),
                "volume_ratio": round(volume_ratio, 2),
                "recommendation": recommendation,
                "color": color,
                "timestamp": current_time
            }
            
            # إرسال تنبيه إذا كان التذبذب عالي ولم يمضِ وقت كافي منذ آخر تنبيه
            if volatility_level in ["high", "extreme"] and (current_time - self.last_alert_time) > self.cooldown_period:
                self.last_alert_time = current_time
                result["alert"] = True
            else:
                result["alert"] = False
                
            return result
            
        except Exception as e:
            return {"volatility_level": "unknown", "error": str(e), "recommendation": "ERROR"}
    
    def should_avoid_trading(self, volatility_data):
        """تحديد إذا كان يجب تجنب التداول بسبب التذبذب"""
        if not volatility_data or volatility_data.get("volatility_level") in ["unknown", "error"]:
            return False
        
        return volatility_data["volatility_level"] in ["high", "extreme"]
    
    def get_volatility_history_summary(self):
        """الحصول على ملخص تاريخ التذبذب"""
        if not self.high_volatility_periods:
            return "No high volatility periods recently"
        
        recent_periods = [p for p in self.high_volatility_periods 
                         if time.time() - p['timestamp'] < 3600]  # آخر ساعة
        
        if not recent_periods:
            return "No high volatility in the last hour"
        
        extreme_count = len([p for p in recent_periods if p['level'] == 'extreme'])
        high_count = len([p for p in recent_periods if p['level'] == 'high'])
        
        return f"Recent volatility: {extreme_count} extreme, {high_count} high periods in last hour"

# إنشاء كاشف التذبذب
volatility_detector = AdvancedVolatilityDetector()

# =================== PORTFOLIO TRACKER ===================
class PortfolioTracker:
    """تتبع محفظة التداول والأرباح التراكمية"""
    
    def __init__(self):
        self.initial_balance = None
        self.daily_profits = []
        self.hourly_balances = []
        self.peak_balance = 0
        self.drawdown = 0
        
    def update_balance(self, current_balance):
        """تحديد رصيد المحفظة"""
        try:
            if self.initial_balance is None and current_balance:
                self.initial_balance = current_balance
                self.peak_balance = current_balance
            
            if current_balance and self.initial_balance:
                # تحديث أعلى رصيد
                if current_balance > self.peak_balance:
                    self.peak_balance = current_balance
                
                # حساب ال drawdown
                self.drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
                
                # تسجيل الرصيد بالساعة
                current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
                self.hourly_balances.append({
                    'timestamp': current_hour,
                    'balance': current_balance,
                    'equity': current_balance
                })
                
                # الاحتفاظ بسجل 24 ساعة فقط
                twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
                self.hourly_balances = [b for b in self.hourly_balances 
                                      if b['timestamp'] >= twenty_four_hours_ago]
        
        except Exception as e:
            print(f"Portfolio update error: {e}")
    
    def record_trade_profit(self, profit):
        """تسجيل ربح صفقة جديدة"""
        try:
            if profit != 0:
                self.daily_profits.append({
                    'timestamp': datetime.now(),
                    'profit': profit,
                    'type': 'win' if profit > 0 else 'loss'
                })
                
                # الاحتفاظ بسجل 7 أيام فقط
                seven_days_ago = datetime.now() - timedelta(days=7)
                self.daily_profits = [p for p in self.daily_profits 
                                    if p['timestamp'] >= seven_days_ago]
        except Exception as e:
            print(f"Profit recording error: {e}")
    
    def get_portfolio_summary(self, current_balance):
        """الحصول على ملخص المحفظة"""
        try:
            if not self.initial_balance or not current_balance:
                return None
            
            total_profit = current_balance - self.initial_balance
            total_return = (total_profit / self.initial_balance) * 100
            
            # حساب الأرباح اليومية
            today = datetime.now().date()
            today_profits = [p for p in self.daily_profits 
                           if p['timestamp'].date() == today]
            daily_profit = sum(p['profit'] for p in today_profits)
            daily_return = (daily_profit / current_balance) * 100 if current_balance > 0 else 0
            
            # إحصائيات الأداء
            winning_trades = [p for p in self.daily_profits if p['type'] == 'win']
            losing_trades = [p for p in self.daily_profits if p['type'] == 'loss']
            
            win_rate = (len(winning_trades) / len(self.daily_profits)) * 100 if self.daily_profits else 0
            avg_win = sum(p['profit'] for p in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(p['profit'] for p in losing_trades) / len(losing_trades) if losing_trades else 0
            
            return {
                'current_balance': round(current_balance, 2),
                'initial_balance': round(self.initial_balance, 2),
                'total_profit': round(total_profit, 2),
                'total_return': round(total_return, 2),
                'daily_profit': round(daily_profit, 2),
                'daily_return': round(daily_return, 2),
                'peak_balance': round(self.peak_balance, 2),
                'drawdown': round(self.drawdown, 2),
                'win_rate': round(win_rate, 1),
                'total_trades': len(self.daily_profits),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            }
            
        except Exception as e:
            print(f"Portfolio summary error: {e}")
            return None

# إنشاء متتبع المحفظة
portfolio_tracker = PortfolioTracker()

# =================== PROFESSIONAL TRADING LOGGER ===================
class ProfessionalTradingLogger:
    """نظام تسجيل محترف للتداول"""
    
    @staticmethod
    def log_trading_session(balance, portfolio_summary, indicators, volatility_data, council_data, position_open=False):
        """تسجيل جلسة التداول الكاملة"""
        try:
            print("\n" + "="*80)
            print("🎯 SUI ULTRA PRO AI - TRADING SESSION REPORT")
            print("="*80)
            
            # قسم المحفظة والأرباح
            ProfessionalTradingLogger._log_portfolio_section(balance, portfolio_summary)
            
            # قسم المؤشرات الفنية
            ProfessionalTradingLogger._log_indicators_section(indicators)
            
            # قسم التذبذب
            ProfessionalTradingLogger._log_volatility_section(volatility_data)
            
            # قسم مجلس التصويت
            ProfessionalTradingLogger._log_council_section(council_data)
            
            # قسم المناطق الاستراتيجية
            ProfessionalTradingLogger._log_strategy_zones_section(council_data)
            
            print("="*80)
            
        except Exception as e:
            print(f"Professional logging error: {e}")

    @staticmethod
    def _log_portfolio_section(balance, portfolio_summary):
        """تسجيل قسم المحفظة"""
        print("💰 PORTFOLIO STATUS:")
        if portfolio_summary:
            current_balance = portfolio_summary.get('current_balance', 0)
            total_profit = portfolio_summary.get('total_profit', 0)
            total_return = portfolio_summary.get('total_return', 0)
            daily_profit = portfolio_summary.get('daily_profit', 0)
            win_rate = portfolio_summary.get('win_rate', 0)
            
            profit_color = "🟢" if total_profit >= 0 else "🔴"
            daily_color = "🟢" if daily_profit >= 0 else "🔴"
            
            print(f"   {profit_color} Balance: ${current_balance:.2f}")
            print(f"   {profit_color} Total PnL: ${total_profit:.2f} ({total_return:.2f}%)")
            print(f"   {daily_color} Today: ${daily_profit:.2f}")
            print(f"   📊 Win Rate: {win_rate:.1f}%")
            print(f"   📈 Peak: ${portfolio_summary.get('peak_balance', 0):.2f}")
            print(f"   📉 Drawdown: {portfolio_summary.get('drawdown', 0):.2f}%")
        else:
            print(f"   💰 Current Balance: ${balance:.2f}")
        print("   ───────────────────────────────────────────────────")

    @staticmethod
    def _log_indicators_section(indicators):
        """تسجيل قسم المؤشرات"""
        print("📊 TECHNICAL INDICATORS:")
        if indicators:
            # مؤشرات الزخم
            rsi = indicators.get('rsi', 0)
            rsi_status = "🟢" if rsi < 30 else "🔴" if rsi > 70 else "🟡"
            print(f"   {rsi_status} RSI: {rsi:.1f}")
            
            # مؤشرات الترند
            adx = indicators.get('adx', 0)
            plus_di = indicators.get('plus_di', 0)
            minus_di = indicators.get('minus_di', 0)
            adx_status = "🟢" if adx > 25 else "🔴" if adx < 15 else "🟡"
            print(f"   {adx_status} ADX: {adx:.1f} | +DI: {plus_di:.1f} | -DI: {minus_di:.1f}")
            
            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_hist = indicators.get('macd_hist', 0)
            macd_status = "🟢" if macd_hist > 0 else "🔴"
            print(f"   {macd_status} MACD: {macd:.4f} | Signal: {macd_signal:.4f} | Hist: {macd_hist:.4f}")
            
            # التقلب والمتوسطات
            atr = indicators.get('atr', 0)
            sma_20 = indicators.get('sma_20', 0)
            ema_20 = indicators.get('ema_20', 0)
            print(f"   📈 ATR: {atr:.4f} | SMA20: {sma_20:.4f} | EMA20: {ema_20:.4f}")
        else:
            print("   📭 No indicator data available")
        print("   ───────────────────────────────────────────────────")

    @staticmethod
    def _log_volatility_section(volatility_data):
        """تسجيل قسم التذبذب"""
        print("🛡️ VOLATILITY ANALYSIS:")
        if volatility_data:
            level = volatility_data.get('volatility_level', 'unknown')
            score = volatility_data.get('volatility_score', 0)
            color = volatility_data.get('color', '⚪')
            recommendation = volatility_data.get('recommendation', 'UNKNOWN')
            
            print(f"   {color} Level: {level.upper()} (Score: {score})")
            print(f"   📊 ATR Ratio: {volatility_data.get('atr_ratio', 0)}")
            print(f"   💹 Price Vol: {volatility_data.get('price_volatility', 0):.2f}%")
            print(f"   ⚠️  Recommendation: {recommendation}")
            
            if level in ["high", "extreme"]:
                print(f"   🚫 TRADING SUSPENDED - High Volatility Detected")
        else:
            print("   📭 No volatility data available")
        print("   ───────────────────────────────────────────────────")

    @staticmethod
    def _log_council_section(council_data):
        """تسجيل قسم مجلس التصويت"""
        print("🗳️ COUNCIL VOTING RESULTS:")
        if council_data:
            score_b = council_data.get('score_b', 0)
            score_s = council_data.get('score_s', 0)
            votes_b = council_data.get('b', 0)
            votes_s = council_data.get('s', 0)
            confidence = council_data.get('confidence', 0)
            trade_type = council_data.get('trade_type', 'scalp')
            
            # تحديد الفائز
            if score_b > score_s:
                winner = "🟢 BUY"
                winner_score = score_b
            elif score_s > score_b:
                winner = "🔴 SELL" 
                winner_score = score_s
            else:
                winner = "⚪ TIE"
                winner_score = 0
            
            print(f"   {winner} Signal - Score: {winner_score:.1f}")
            print(f"   🟢 BUY: {votes_b} votes | {score_b:.1f} points")
            print(f"   🔴 SELL: {votes_s} votes | {score_s:.1f} points")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   🎯 Trade Type: {trade_type.upper()}")
            
            # تحليل القرار
            min_score = 18.0
            min_confidence = 0.78
            
            if winner_score >= min_score and confidence >= min_confidence:
                print(f"   ✅ DECISION: STRONG SIGNAL - Ready to trade")
            else:
                print(f"   ❌ DECISION: WEAK SIGNAL - Waiting for better opportunity")
        else:
            print("   📭 No council data available")
        print("   ───────────────────────────────────────────────────")

    @staticmethod
    def _log_strategy_zones_section(council_data):
        """تسجيل قسم المناطق الاستراتيجية"""
        print("🎯 STRATEGY ZONES:")
        if council_data and council_data.get('analysis'):
            analysis = council_data['analysis']
            
            # مناطق العرض والطلب
            supply_demand = analysis.get('supply_demand', {})
            supply_zones = supply_demand.get('supply_zones', [])
            demand_zones = supply_demand.get('demand_zones', [])
            
            if demand_zones:
                print(f"   🛡️ DEMAND ZONES: {len(demand_zones)} active")
                for zone in demand_zones[:2]:  # أول منطقتين فقط
                    print(f"      - Price: {zone.get('price', 0):.4f} | Strength: {zone.get('strength', 0):.1f}%")
            
            if supply_zones:
                print(f"   🚧 SUPPLY ZONES: {len(supply_zones)} active") 
                for zone in supply_zones[:2]:
                    print(f"      - Price: {zone.get('price', 0):.4f} | Strength: {zone.get('strength', 0):.1f}%")
            
            # SMC Analysis
            smc = analysis.get('smc', {})
            order_blocks = smc.get('order_blocks', 0)
            fvgs = smc.get('fvgs', 0)
            bos_choch = smc.get('bos_choch', {})
            
            print(f"   🔧 SMC ANALYSIS:")
            print(f"      - Order Blocks: {order_blocks}")
            print(f"      - FVGs: {fvgs}")
            print(f"      - BOS: {'Y' if bos_choch.get('bos_bullish') or bos_choch.get('bos_bearish') else 'N'}")
            print(f"      - CHoCH: {'Y' if bos_choch.get('choch_bullish') or bos_choch.get('choch_bearish') else 'N'}")
            
        else:
            print("   📭 No strategy zones data available")

    @staticmethod
    def log_trade_signal(signal_side, current_price, position_size, council_data):
        """تسجيل إشارة التداول"""
        print("\n" + "⭐" * 40)
        if signal_side == "buy":
            print("🟢🟢🟢 BUY SIGNAL DETECTED 🟢🟢🟢")
        else:
            print("🔴🔴🔴 SELL SIGNAL DETECTED 🔴🔴🔴")
        print("⭐" * 40)
        
        print(f"🎯 TRADE DETAILS:")
        print(f"   Direction: {signal_side.upper()}")
        print(f"   Entry Price: {current_price:.6f}")
        print(f"   Position Size: {position_size:.4f}")
        print(f"   Trade Type: {council_data.get('trade_type', 'scalp').upper()}")
        print(f"   Confidence: {council_data.get('confidence', 0):.2f}")
        
        print(f"📋 ENTRY REASONS:")
        logs = council_data.get('logs', [])
        for i, log_msg in enumerate(logs[-8:]):  # آخر 8 أسباب
            print(f"   {i+1}. {log_msg}")
        
        print("⭐" * 40 + "\n")

    @staticmethod
    def log_no_trade_reasons(council_data, volatility_data):
        """تسجيل أسباب عدم الدخول في صفقة"""
        print("\n" + "❌" * 30)
        print("❌ NO TRADE - REASONS ANALYSIS ❌")
        print("❌" * 30)
        
        score_b = council_data.get('score_b', 0)
        score_s = council_data.get('score_s', 0) 
        confidence = council_data.get('confidence', 0)
        
        min_score = 18.0
        min_confidence = 0.78
        
        reasons = []
        
        # تحليل النقاط
        if score_b < min_score and score_s < min_score:
            reasons.append(f"Both scores below minimum ({min_score})")
        elif max(score_b, score_s) < min_score:
            reasons.append(f"Winning score {max(score_b, score_s):.1f} < {min_score}")
        
        # تحليل الثقة
        if confidence < min_confidence:
            reasons.append(f"Confidence {confidence:.2f} < {min_confidence}")
        
        # تحليل التذبذب
        if volatility_data and volatility_detector.should_avoid_trading(volatility_data):
            reasons.append(f"High volatility: {volatility_data.get('volatility_level')}")
        
        # تحليل الفارق
        score_diff = abs(score_b - score_s)
        if score_diff < 5:
            reasons.append(f"Score difference too small: {score_diff:.1f}")
        
        # عرض الأسباب
        if reasons:
            print("📊 DECISION ANALYSIS:")
            for i, reason in enumerate(reasons):
                print(f"   {i+1}. {reason}")
        else:
            print("   🤔 No specific reasons identified")
        
        # عرض توصيات المجلس
        logs = council_data.get('logs', [])
        if logs:
            print("📝 COUNCIL RECOMMENDATIONS:")
            for i, log_msg in enumerate(logs[-5:]):
                print(f"   • {log_msg}")
        
        print("❌" * 30 + "\n")

    @staticmethod
    def log_position_opened(position_data, council_data):
        """تسجيل فتح الصفقة"""
        side = position_data.get("side")
        entry = position_data.get("entry")
        qty = position_data.get("qty")
        trade_type = position_data.get("trade_type", "scalp")
        
        print("\n" + "💎" * 35)
        if side == "long":
            print("💎💎💎 BUY POSITION OPENED 💎💎💎")
            color_icon = "🟢"
        else:
            print("💎💎💎 SELL POSITION OPENED 💎💎💎") 
            color_icon = "🔴"
        print("💎" * 35)
        
        print(f"{color_icon} POSITION DETAILS:")
        print(f"   Direction: {side.upper()}")
        print(f"   Entry Price: {entry:.6f}")
        print(f"   Quantity: {qty:.4f}")
        print(f"   Trade Type: {trade_type.upper()}")
        print(f"   Leverage: {LEVERAGE}x")
        
        # أهداف الربح
        if position_data.get("profit_targets"):
            print(f"🎯 PROFIT TARGETS:")
            for target in position_data["profit_targets"][:3]:  # أول 3 أهداف
                print(f"   TP{target['level']}: {target['price']:.6f} ({target['target_pct']:.1f}%)")
        
        print(f"📊 ENTRY ANALYSIS:")
        logs = council_data.get('logs', [])
        for i, log_msg in enumerate(logs[-6:]):
            print(f"   {i+1}. {log_msg}")
        
        print("💎" * 35 + "\n")

# إنشاء الكائن
trading_logger = ProfessionalTradingLogger()

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
BOT_VERSION = f"SUI ULTRA PRO AI v10.0 — {EXCHANGE_NAME.upper()} - PROFESSIONAL LOGGER + RESOURCE SAVER + SMART SCALP"
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
LEVERAGE   = int(os.getenv("LEVERAGE", 15))
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

# Pacing - UPDATED FOR RESOURCE SAVING
BASE_SLEEP   = 12  # 🔽 كان 5
NEAR_CLOSE_S = 3   # 🔽 كان 1

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

# ===== VOLATILITY PROTECTION SETTINGS =====
VOLATILITY_PROTECTION = True
MAX_VOLATILITY_SCORE = 4.0
VOLATILITY_COOLDOWN_MIN = 10

# ===== SUPER SCALP ENGINE - ENHANCED =====
SCALP_MODE            = True
SCALP_EXECUTE         = True
SCALP_SIZE_FACTOR     = 0.45  # 🔼 زيادة حجم صفقات السكالب
SCALP_ADX_GATE        = 12.0
SCALP_MIN_SCORE       = 20.0  # 🔼 زيادة الحد الأدنى لنقاط السكالب
SCALP_IMB_THRESHOLD   = 1.00
SCALP_VOL_MA_FACTOR   = 1.50  # 🔼 زيادة عامل الحجم للسكالب
SCALP_COOLDOWN_SEC    = 8
SCALP_RESPECT_WAIT    = False
SCALP_TP_SINGLE_PCT   = 0.45  # 🔼 زيادة هدف الربح للسكالب
SCALP_BE_AFTER_PCT    = 0.15
SCALP_ATR_TRAIL_MULT  = 1.0

# ===== SMART SCALP ENTRY CONDITIONS =====
SCALP_REQUIRE_STRONG_SIGNALS = True
SCALP_MIN_VOLUME_RATIO = 1.8  # الحد الأدنى لنسبة الحجم للسكالب
SCALP_MIN_CONFIDENCE = 0.85   # 🔼 زيادة الثقة المطلوبة للسكالب
SCALP_ZERO_REVERSAL_CHECK = True  # تفعيل نظرية 0 انعكاس
SCALP_STRONG_ZONE_REQUIRED = True  # يتطلب مناطق انعكاس قوية

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

# =================== ENHANCED SCALP DETECTOR ===================
class EnhancedScalpDetector:
    """كاشف محسن لصفقات السكالب مع إشارات قوية"""
    
    def __init__(self):
        self.last_scalp_time = 0
        self.scalp_cooldown = 300  # 5 دقائق بين صفقات السكالب
        self.strong_signals_history = []
        
    def analyze_scalp_conditions(self, df, council_data, current_price):
        """تحليل شروط السكالب المحسنة"""
        try:
            if len(df) < 50:
                return {"qualified": False, "reason": "Insufficient data"}
            
            # فحص الوقت بين صفقات السكالب
            current_time = time.time()
            if current_time - self.last_scalp_time < self.scalp_cooldown:
                return {"qualified": False, "reason": "Scalp cooldown active"}
            
            volume = df['volume'].astype(float)
            close = df['close'].astype(float)
            
            # 🔼 شروط السكالب القوية
            conditions = {
                "high_volume": False,
                "strong_council_score": False,
                "zero_reversal": False,
                "key_level": False,
                "momentum_alignment": False
            }
            
            reasons = []
            
            # 1. شرط الحجم القوي
            current_volume = volume.iloc[-1]
            volume_ma = volume.tail(20).mean()
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
            
            if volume_ratio >= SCALP_MIN_VOLUME_RATIO:
                conditions["high_volume"] = True
                reasons.append(f"Strong volume: {volume_ratio:.2f}x")
            else:
                reasons.append(f"Weak volume: {volume_ratio:.2f}x")
            
            # 2. شرط نقاط المجلس القوية
            score_b = council_data.get('score_b', 0)
            score_s = council_data.get('score_s', 0)
            confidence = council_data.get('confidence', 0)
            
            if max(score_b, score_s) >= SCALP_MIN_SCORE and confidence >= SCALP_MIN_CONFIDENCE:
                conditions["strong_council_score"] = True
                reasons.append(f"Strong council score: {max(score_b, score_s):.1f}, confidence: {confidence:.2f}")
            else:
                reasons.append(f"Weak council score: {max(score_b, score_s):.1f}, confidence: {confidence:.2f}")
            
            # 3. شرط نظرية 0 انعكاس
            if SCALP_ZERO_REVERSAL_CHECK:
                zero_reversal = self.detect_zero_reversal(df, council_data)
                if zero_reversal["detected"]:
                    conditions["zero_reversal"] = True
                    reasons.append(f"Zero reversal detected: {zero_reversal['type']}")
                else:
                    reasons.append("No zero reversal detected")
            
            # 4. شرط المستويات الرئيسية
            if SCALP_STRONG_ZONE_REQUIRED:
                key_level = self.check_key_levels(df, current_price, council_data)
                if key_level["at_key_level"]:
                    conditions["key_level"] = True
                    reasons.append(f"At key level: {key_level['level_type']}")
                else:
                    reasons.append("Not at key level")
            
            # 5. محاذاة الزخم
            momentum_aligned = self.check_momentum_alignment(df, council_data)
            conditions["momentum_alignment"] = momentum_aligned
            reasons.append(f"Momentum aligned: {momentum_aligned}")
            
            # حساب إجمالي الشروط المحققة
            met_conditions = sum(conditions.values())
            total_conditions = len(conditions)
            
            qualified = met_conditions >= 4  # يتطلب تحقيق 4/5 شروط على الأقل
            
            if qualified:
                self.last_scalp_time = current_time
                self.strong_signals_history.append({
                    'timestamp': current_time,
                    'conditions': conditions,
                    'price': current_price,
                    'type': 'buy' if score_b > score_s else 'sell'
                })
                # الاحتفاظ بآخر 20 إشارة
                self.strong_signals_history = self.strong_signals_history[-20:]
            
            result = {
                "qualified": qualified,
                "met_conditions": met_conditions,
                "total_conditions": total_conditions,
                "conditions": conditions,
                "reasons": reasons,
                "volume_ratio": volume_ratio,
                "is_strong_scalp": qualified
            }
            
            return result
            
        except Exception as e:
            return {"qualified": False, "reason": f"Analysis error: {str(e)}"}
    
    def detect_zero_reversal(self, df, council_data):
        """كشف انعكاسات نظرية 0 انعكاس"""
        try:
            analysis = council_data.get('analysis', {})
            smc_data = analysis.get('smc', {})
            price_testing = analysis.get('price_testing', {})
            
            # فحص Order Blocks نشطة
            order_blocks = smc_data.get('order_blocks', 0)
            
            # فحص مناطق العرض والطلب
            supply_demand = analysis.get('supply_demand', {})
            demand_zones = supply_demand.get('demand_zones', [])
            supply_zones = supply_demand.get('supply_zones', [])
            
            current_price = float(df['close'].iloc[-1])
            
            # البحث عن مناطق انعكاس قريبة
            near_demand = any(abs(zone['price'] - current_price) / current_price < 0.005 for zone in demand_zones)
            near_supply = any(abs(zone['price'] - current_price) / current_price < 0.005 for zone in supply_zones)
            
            # فحص BOS/CHoCH
            bos_choch = smc_data.get('bos_choch', {})
            has_reversal_structure = (
                bos_choch.get('choch_bullish', False) or 
                bos_choch.get('choch_bearish', False) or
                bos_choch.get('bos_bullish', False) or
                bos_choch.get('bos_bearish', False)
            )
            
            detected = (near_demand or near_supply) and (order_blocks > 0 or has_reversal_structure)
            
            return {
                "detected": detected,
                "type": "demand" if near_demand else "supply" if near_supply else "none",
                "order_blocks": order_blocks,
                "has_structure": has_reversal_structure
            }
            
        except Exception as e:
            return {"detected": False, "type": "error"}
    
    def check_key_levels(self, df, current_price, council_data):
        """التحقق من المستويات الرئيسية"""
        try:
            analysis = council_data.get('analysis', {})
            supply_demand = analysis.get('supply_demand', {})
            price_testing = analysis.get('price_testing', {})
            
            demand_zones = supply_demand.get('demand_zones', [])
            supply_zones = supply_demand.get('supply_zones', [])
            
            # البحث عن أقرب منطقة طلب
            nearest_demand = min([abs(zone['price'] - current_price) for zone in demand_zones]) if demand_zones else float('inf')
            nearest_supply = min([abs(zone['price'] - current_price) for zone in supply_zones]) if supply_zones else float('inf')
            
            threshold = 0.005  # 0.5%
            
            if nearest_demand <= threshold:
                return {"at_key_level": True, "level_type": "demand", "distance": nearest_demand}
            elif nearest_supply <= threshold:
                return {"at_key_level": True, "level_type": "supply", "distance": nearest_supply}
            else:
                return {"at_key_level": False, "level_type": "none"}
                
        except Exception as e:
            return {"at_key_level": False, "level_type": "error"}
    
    def check_momentum_alignment(self, df, council_data):
        """التحقق من محاذاة الزخم"""
        try:
            indicators = council_data.get('indicators', {})
            
            rsi = indicators.get('rsi', 50)
            macd_hist = indicators.get('macd_hist', 0)
            adx = indicators.get('adx', 0)
            plus_di = indicators.get('plus_di', 0)
            minus_di = indicators.get('minus_di', 0)
            
            score_b = council_data.get('score_b', 0)
            score_s = council_data.get('score_s', 0)
            
            # محاذاة الزخم للشراء
            if score_b > score_s:
                return (rsi < 40 and macd_hist > 0 and plus_di > minus_di)
            # محاذاة الزخم للبيع
            elif score_s > score_b:
                return (rsi > 60 and macd_hist < 0 and minus_di > plus_di)
            else:
                return False
                
        except Exception as e:
            return False

# إنشاء كاشف السكالب المحسن
scalp_detector = EnhancedScalpDetector()

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
        
        # تحديث متتبع المحفظة
        portfolio_tracker.record_trade_profit(profit)
    
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

def log_v(msg): 
    print(f"📊 {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_p(msg): 
    print(f"💰 {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

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

def normalize_qty(symbol: str, qty: float) -> float:
    """
    - يقرّب الكمية لمضاعفات step
    - يتأكد إنها >= min_qty الخاص بالرمز
    - يرجّع 0.0 لو الكمية فعلاً صغيرة جدًا
    """
    if qty is None or qty <= 0:
        return 0.0

    min_qty = MIN_QTY_BY_SYMBOL.get(symbol, DEFAULT_MIN_QTY)
    step = QTY_STEP_BY_SYMBOL.get(symbol, DEFAULT_QTY_STEP)

    # تقريب لأسفل لأقرب step
    normalized = math.floor(qty / step) * step

    # نحمي من القيم العجيبة
    normalized = float(f"{normalized:.6f}")

    if normalized < min_qty:
        log_w(f"[SIZE] qty {normalized} < min_qty {min_qty} for {symbol} -> skip trade")
        return 0.0

    log_i(f"[SIZE] Normalized {qty:.4f} -> {normalized:.4f} (min={min_qty}, step={step})")
    return normalized

def safe_qty(q):
    """النسخة المحسنة من safe_qty باستخدام normalize_qty"""
    if q is None or q <= 0:
        return 0.0
    return normalize_qty(SYMBOL, q)

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

def fetch_ohlcv(limit=MIN_CANDLES):  # 🔽 استخدام MIN_CANDLES بدلاً من 600
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

# =================== دوال توفير الموارد الجديدة ===================
def should_run_analysis():
    """تحديد إذا كان يجب تشغيل التحليل الكامل أو الاقتصادي"""
    if not STATE["open"] and RESOURCE_SAVER_MODE:
        # إذا لا يوجد مركز مفتوح، قلل التحليل (70% تخطي)
        return random.random() > 0.3  # 30% تشغيل تحليل
    return True

def quick_analysis(df):
    """تحليل سريع خفيف للموارد"""
    try:
        if len(df) < 50:
            return {"b": 0, "s": 0, "score_b": 0, "score_s": 0, "confidence": 0}
        
        # 🔽 مؤشرات سريعة فقط
        close = df['close'].astype(float)
        rsi = tv.tv_rsi(close, 14).iloc[-1]
        
        # قرار مبسط بناء على RSI فقط
        score_b = 0
        score_s = 0
        
        if rsi < 30:
            score_b = 12
        elif rsi > 70:
            score_s = 12
        
        return {
            "b": 1 if score_b > 0 else 0,
            "s": 1 if score_s > 0 else 0, 
            "score_b": score_b,
            "score_s": score_s,
            "confidence": 0.5,
            "trade_type": "scalp",
            "logs": [f"Quick Analysis - RSI: {rsi:.1f}"]
        }
    except Exception as e:
        return {"b": 0, "s": 0, "score_b": 0, "score_s": 0, "confidence": 0}

def optimize_memory():
    """تحسين استخدام الذاكرة"""
    if RESOURCE_SAVER_MODE:
        gc.collect()  # تشغيل جامع القمامة

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
        
        # === تحليل التذبذب ===
        volatility_data = volatility_detector.calculate_volatility_metrics(df)
        
        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0
        logs = []
        confidence_factors = []
        
        current_price = float(df['close'].iloc[-1])
        
        # ===== 0. فحص التذبذب أولاً (حماية رئيسية) =====
        if VOLATILITY_PROTECTION and volatility_detector.should_avoid_trading(volatility_data):
            logs.append(f"🚫 VOLATILITY PROTECTION: {volatility_data.get('volatility_level', 'unknown').upper()} market - Trading suspended")
            # إرجاع نتيجة تمنع التداول
            return {
                "b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, 
                "confidence": 0.0, "trade_type": "avoid", "logs": logs,
                "volatility_alert": True,
                "analysis": {"volatility": volatility_data}
            }
        
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
                },
                "volatility": volatility_data
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
def compute_adaptive_position_size(balance, price, confidence, market_phase, is_scalp=False):
    """حساب حجم صفقة متكيف محترف مع التقريب الصحيح"""
    
    # الحساب الأساسي
    base_size = (balance * RISK_ALLOC * LEVERAGE) / price
    
    # تعديل الحجم بناءً على الثقة
    confidence_multiplier = 0.6 + (confidence * 0.4)  # 0.6 إلى 1.0
    
    # تعديل الحجم بناءً على مرحلة السوق
    if market_phase in ["strong_bull", "strong_bear"]:
        market_multiplier = 1.3
    elif market_phase in ["bull", "bear"]:
        market_multiplier = 1.1
    else:
        market_multiplier = 0.8
    
    # 🔼 زيادة حجم صفقات السكالب القوية
    if is_scalp:
        scalp_multiplier = SCALP_SIZE_FACTOR * 1.3  # زيادة 30% للسكالب القوية
    else:
        scalp_multiplier = 1.0
    
    raw_qty = base_size * confidence_multiplier * market_multiplier * scalp_multiplier
    
    # ✅ التقريب الصحيح باستخدام normalize_qty
    final_qty = normalize_qty(SYMBOL, raw_qty)
    
    log_i(f"📊 PROFESSIONAL POSITION SIZING:")
    log_i(f"   Balance: ${balance:.2f}")
    log_i(f"   Risk Alloc: {RISK_ALLOC*100}%")
    log_i(f"   Leverage: {LEVERAGE}x")  
    log_i(f"   Base Size: {base_size:.4f}")
    log_i(f"   Raw Qty: {raw_qty:.4f}")
    log_i(f"   Final Qty: {final_qty:.4f}")
    log_i(f"   Is Scalp: {is_scalp}")
    
    return final_qty

def execute_professional_trade(side, price, qty, council_data, market_analysis, is_strong_scalp=False):
    """تنفيذ صفقة محترفة مع تحقق من الكمية"""
    
    # ✅ تحقق نهائي من الكمية
    if qty <= 0:
        log_e(f"❌ INVALID QUANTITY: {qty} - Skipping trade")
        return False
        
    log_i(f"🎯 PROFESSIONAL TRADE EXECUTION:")
    log_i(f"   SIDE: {side.upper()}")
    log_i(f"   QTY: {qty:.4f} SUI")
    log_i(f"   PRICE: {price:.6f}")
    log_i(f"   VALUE: ${qty * price:.2f}")
    log_i(f"   TYPE: {council_data.get('trade_type', 'scalp').upper()}")
    log_i(f"   CONFIDENCE: {council_data.get('confidence', 0):.2f}")
    log_i(f"   STRONG SCALP: {is_strong_scalp}")
    
    # عرض أسباب الدخول المفصلة
    log_i(f"   📋 ENTRY REASONS:")
    for i, log_msg in enumerate(council_data.get('logs', [])[-10:]):
        log_i(f"      {i+1}. {log_msg}")
    
    if not EXECUTE_ORDERS or DRY_RUN:
        log_i(f"DRY_RUN: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
    
    if MODE_LIVE:
        try:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
            
            log_g(f"✅ PROFESSIONAL TRADE EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
            
            # تسجيل الصفقة مع الأسباب
            entry_reason = " | ".join(council_data.get('logs', [])[-5:])
            if is_strong_scalp:
                entry_reason += " | STRONG SCALP SIGNAL"
            
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
    
    return True

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

# =================== PROFESSIONAL TRADING LOOP - OPTIMIZED ===================
def professional_trading_loop():
    """الحلقة الرئيسية للتداول المحترف مع تحسينات توفير الموارد"""
    
    log_banner("STARTING ULTIMATE PROFESSIONAL TRADING BOT - RESOURCE SAVER + SMART SCALP MODE")
    log_i(f"🤖 Bot Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    log_i(f"📊 Risk Allocation: {RISK_ALLOC*100}%")
    log_i(f"🎯 Indicators: TradingView/Bybit Precision Mode")
    log_i(f"🛡️ Volatility Protection: {'ACTIVE' if VOLATILITY_PROTECTION else 'INACTIVE'}")
    log_i(f"💾 Resource Saver Mode: {'ACTIVE' if RESOURCE_SAVER_MODE else 'INACTIVE'}")
    log_i(f"🔥 Smart Scalp Mode: {'ACTIVE' if SCALP_MODE else 'INACTIVE'}")
    log_i(f"📉 Candles Limit: {MIN_CANDLES} (كان 500)")
    log_i(f"⏱️ Base Sleep: {BASE_SLEEP}s (كان 5s)")
    
    # عرض إحصائيات البداية
    performance = pro_trade_manager.analyze_trade_performance()
    log_i(f"📈 Historical Performance: Win Rate: {performance.get('win_rate', 0):.1f}% | Total Profit: {performance.get('total_profit', 0):.2f} USDT")
    
    cycle_count = 0
    consecutive_skips = 0
    
    while True:
        try:
            cycle_count += 1
            
            # 🔽 التحقق من وضع توفير الموارد
            if not should_run_analysis() and not STATE["open"]:
                consecutive_skips += 1
                if consecutive_skips % 10 == 0:  # كل 10 تخطيات سجل
                    log_v(f"💤 Resource saver mode - skip count: {consecutive_skips}")
                time.sleep(MAX_LOOP_FREQUENCY)
                continue
            
            consecutive_skips = 0
            
            # جمع البيانات الأساسية
            balance = balance_usdt()
            current_price = price_now()
            
            # 🔽 استخدام عدد شموع مخفض
            df = fetch_ohlcv(limit=MIN_CANDLES)
            
            if df.empty or current_price is None:
                log_w("📭 No data available - retrying...")
                time.sleep(BASE_SLEEP * 2)
                continue
            
            # تحديث متتبع المحفظة
            portfolio_tracker.update_balance(balance)
            portfolio_summary = portfolio_tracker.get_portfolio_summary(balance)
            
            # 🔽 تحديد إذا كان يجب تشغيل المجلس الكامل
            run_full_council = True
            if RESOURCE_SAVER_MODE and not STATE["open"]:
                # قلل تحليل المجلس عندما لا يوجد مركز مفتوح
                run_full_council = (time.time() % 300 < 30)  # كل 5 دقائق لمدة 30 ثانية
            
            if run_full_council:
                council_data = ultra_professional_council_ai(df)
            else:
                # 🔽 تحليل مبسط عندما لا يكون مطلوباً
                council_data = quick_analysis(df)
                log_v("🔽 Quick analysis mode (resource saving)")
            
            volatility_data = council_data.get('analysis', {}).get('volatility', {})
            
            # تحديث الحالة
            STATE["last_council"] = council_data
            STATE["last_ind"] = council_data.get("indicators", {})
            STATE["last_spread_bps"] = orderbook_spread_bps()
            
            # ✅ التسجيل المحترف لجلسة التداول (بتكرار أقل في وضع توفير الموارد)
            if run_full_council or STATE["open"] or cycle_count % 5 == 0:
                trading_logger.log_trading_session(
                    balance, portfolio_summary, 
                    council_data.get("indicators", {}), 
                    volatility_data, council_data, 
                    STATE["open"]
                )
            
            # إدارة المركز المفتوح
            if STATE["open"]:
                STATE["bars"] += 1
                manage_professional_position(df, council_data, current_price)
            
            # فتح صفقات جديدة
            if not STATE["open"]:
                signal_side = None
                signal_reason = ""
                is_strong_scalp = False
                
                # فحص الحماية من التذبذب أولاً
                if council_data.get("volatility_alert"):
                    log_w(f"🚫 VOLATILITY PROTECTION: Trading suspended")
                    time.sleep(BASE_SLEEP * 2)
                    continue
                
                # شروط دخول محترفة
                min_score = 18.0
                min_confidence = 0.78
                
                # 🔼 فحص شروط السكالب القوية
                if SCALP_MODE and council_data.get('trade_type') == 'scalp':
                    scalp_analysis = scalp_detector.analyze_scalp_conditions(df, council_data, current_price)
                    
                    if scalp_analysis.get("qualified"):
                        is_strong_scalp = True
                        log_i(f"🔥 STRONG SCALP SIGNAL DETECTED - Conditions: {scalp_analysis.get('met_conditions')}/{scalp_analysis.get('total_conditions')}")
                        for reason in scalp_analysis.get('reasons', []):
                            log_i(f"   📋 {reason}")
                
                if (council_data["score_b"] > council_data["score_s"] and 
                    council_data["score_b"] >= min_score and 
                    council_data["confidence"] >= min_confidence):
                    signal_side = "buy"
                elif (council_data["score_s"] > council_data["score_b"] and 
                      council_data["score_s"] >= min_score and 
                      council_data["confidence"] >= min_confidence):
                    signal_side = "sell"
                
                if signal_side:
                    # 🔼 استخدام حجم أكبر لصفقات السكالب القوية
                    is_scalp_trade = council_data.get('trade_type') == 'scalp' and is_strong_scalp
                    
                    position_size = compute_adaptive_position_size(
                        balance, current_price, council_data["confidence"], 
                        council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength', 'neutral'),
                        is_scalp=is_scalp_trade
                    )
                    
                    if position_size > 0:
                        # ✅ تسجيل إشارة التداول
                        trading_logger.log_trade_signal(
                            signal_side, current_price, position_size, council_data
                        )
                        
                        success = execute_professional_trade(
                            signal_side, current_price, position_size, council_data, {
                                "market_phase": council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength', 'neutral'),
                                "volatility": council_data.get('analysis', {}).get('manipulation', {}).get('current_volatility', 0)
                            },
                            is_strong_scalp=is_strong_scalp
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
                                "entry_reason": signal_reason,
                                "is_strong_scalp": is_strong_scalp
                            })
                            
                            # حساب أهداف الربح
                            atr = council_data.get('indicators', {}).get('atr', 0) or 0.001
                            market_strength = "strong" if council_data.get('analysis', {}).get('price_testing', {}).get('breakout_strength') == 'strong' else "normal"
                            
                            # 🔼 أهداف ربح أكبر للسكالب القوية
                            if is_strong_scalp:
                                STATE["profit_targets"] = [
                                    {'level': 1, 'price': current_price * (1 + 0.6/100) if signal_side == "buy" else current_price * (1 - 0.6/100), 
                                     'close_fraction': 0.5, 'target_pct': 0.6},
                                    {'level': 2, 'price': current_price * (1 + 1.0/100) if signal_side == "buy" else current_price * (1 - 1.0/100), 
                                     'close_fraction': 0.3, 'target_pct': 1.0},
                                    {'level': 3, 'price': current_price * (1 + 1.5/100) if signal_side == "buy" else current_price * (1 - 1.5/100), 
                                     'close_fraction': 0.2, 'target_pct': 1.5}
                                ]
                            else:
                                STATE["profit_targets"] = profit_manager.calculate_dynamic_tps(
                                    STATE["trade_type"], atr, current_price, STATE["side"], market_strength
                                )
                            
                            # ✅ تسجيل فتح الصفقة
                            trading_logger.log_position_opened(STATE, council_data)
                            
                            save_state({
                                "in_position": True,
                                "side": signal_side.upper(),
                                "entry_price": current_price,
                                "position_qty": position_size,
                                "opened_at": int(time.time()),
                                "trade_type": STATE["trade_type"],
                                "entry_reason": signal_reason,
                                "is_strong_scalp": is_strong_scalp
                            })
                else:
                    # ✅ تسجيل أسباب عدم الدخول (بتكرار أقل)
                    if run_full_council or cycle_count % 3 == 0:
                        trading_logger.log_no_trade_reasons(council_data, volatility_data)
            
            # 🔽 تحديد وقت الانتظار الديناميكي
            if STATE["open"]:
                sleep_time = NEAR_CLOSE_S
            elif RESOURCE_SAVER_MODE:
                sleep_time = MAX_LOOP_FREQUENCY
            else:
                sleep_time = BASE_SLEEP
                
            # 🔽 تحسين الذاكرة كل 10 دورات
            if cycle_count % 10 == 0:
                optimize_memory()
                
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
    "trade_type": None, "profit_targets": [],
    "is_strong_scalp": False
}

# =================== FLASK API (مبسط) ===================
app = Flask(__name__)

@app.route("/")
def home():
    portfolio_summary = portfolio_tracker.get_portfolio_summary(balance_usdt())
    return f"""
    <html>
        <head><title>SUI ULTRA PRO AI BOT - RESOURCE SAVER + SMART SCALP</title></head>
        <body>
            <h1>🚀 SUI ULTRA PRO AI BOT - الإصدار المحترف مع نظام توفير الموارد والسكالب الذكي</h1>
            <p><strong>Version:</strong> {BOT_VERSION}</p>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
            <p><strong>Symbol:</strong> {SYMBOL}</p>
            <p><strong>Status:</strong> {'🟢 LIVE' if MODE_LIVE else '🟡 PAPER'}</p>
            <p><strong>Position:</strong> {'🟢 OPEN' if STATE['open'] else '🔴 CLOSED'}</p>
            <p><strong>Indicators:</strong> TradingView/Bybit Precision Mode</p>
            <p><strong>Volatility Protection:</strong> {'🟢 ACTIVE' if VOLATILITY_PROTECTION else '🔴 INACTIVE'}</p>
            <p><strong>Resource Saver:</strong> {'🟢 ACTIVE' if RESOURCE_SAVER_MODE else '🔴 INACTIVE'}</p>
            <p><strong>Smart Scalp Mode:</strong> {'🟢 ACTIVE' if SCALP_MODE else '🔴 INACTIVE'}</p>
            <p><strong>Candles Limit:</strong> {MIN_CANDLES} (كان 500)</p>
            <p><strong>Base Sleep:</strong> {BASE_SLEEP}s (كان 5s)</p>
            <h2>Portfolio Summary</h2>
            <p><strong>Current Balance:</strong> ${portfolio_summary.get('current_balance', 0) if portfolio_summary else 'N/A'}</p>
            <p><strong>Total Profit:</strong> ${portfolio_summary.get('total_profit', 0) if portfolio_summary else 'N/A'}</p>
            <p><strong>Win Rate:</strong> {portfolio_summary.get('win_rate', 0) if portfolio_summary else 'N/A'}%</p>
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
        "indicators_mode": "TradingView Precision",
        "volatility_protection": VOLATILITY_PROTECTION,
        "resource_saver_mode": RESOURCE_SAVER_MODE,
        "smart_scalp_mode": SCALP_MODE,
        "candles_limit": MIN_CANDLES,
        "base_sleep": BASE_SLEEP
    })

@app.route("/performance")
def performance():
    performance_data = pro_trade_manager.analyze_trade_performance()
    portfolio_data = portfolio_tracker.get_portfolio_summary(balance_usdt())
    return jsonify({
        "trade_performance": performance_data,
        "portfolio": portfolio_data
    })

@app.route("/volatility")
def volatility_status():
    df = fetch_ohlcv(limit=100)
    volatility_data = volatility_detector.calculate_volatility_metrics(df)
    return jsonify(volatility_data)

@app.route("/scalp-analysis")
def scalp_analysis():
    df = fetch_ohlcv(limit=100)
    council_data = ultra_professional_council_ai(df)
    current_price = price_now()
    scalp_data = scalp_detector.analyze_scalp_conditions(df, council_data, current_price)
    return jsonify(scalp_data)

# =================== STARTUP ===================
def startup_sequence():
    """تسلسل بدء التشغيل"""
    log_banner("PROFESSIONAL SYSTEM INITIALIZATION - RESOURCE SAVER + SMART SCALP MODE")
    
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
        
        # تحديث متتبع المحفظة
        portfolio_tracker.update_balance(balance)
        portfolio_summary = portfolio_tracker.get_portfolio_summary(balance)
        if portfolio_summary:
            log_p(f"📈 Portfolio initialized: ${portfolio_summary.get('current_balance', 0):.2f}")
        
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
    
    # اختبار كاشف التذبذب
    try:
        df = fetch_ohlcv(limit=100)
        volatility_data = volatility_detector.calculate_volatility_metrics(df)
        log_i(f"🛡️ Volatility Detector: {volatility_data.get('color', '⚪')} {volatility_data.get('volatility_level', 'unknown').upper()} - {volatility_data.get('recommendation', 'UNKNOWN')}")
    except Exception as e:
        log_w(f"Volatility detector test failed: {e}")
    
    # اختبار كاشف السكالب المحسن
    try:
        df = fetch_ohlcv(limit=100)
        council_data = ultra_professional_council_ai(df)
        current_price = price_now()
        scalp_analysis = scalp_detector.analyze_scalp_conditions(df, council_data, current_price)
        log_i(f"🔥 Smart Scalp Detector: {scalp_analysis.get('qualified', False)} - Conditions: {scalp_analysis.get('met_conditions', 0)}/{scalp_analysis.get('total_conditions', 0)}")
    except Exception as e:
        log_w(f"Smart scalp detector test failed: {e}")
    
    log_g("🚀 ULTIMATE PROFESSIONAL TRADING BOT READY! - RESOURCE SAVER + SMART SCALP MODE ACTIVE")
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
