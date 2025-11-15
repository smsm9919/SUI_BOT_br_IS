# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المحترف المتكامل مع مؤشرات TradingView/Bybit
• نظام السكالب الدقيق - دخول في المناطق القوية فقط بنظرية 0 انعكاس
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح ديناميكي
• كشف التلاعب والتذبذب والكسر الحقيقي/الوهمي
• إدارة صفقات ذكية متكيفة مع قوة الترند
• حجم صفقات ذكي يضمن الربحية بعد العمولة
• Multi-Exchange Support: BingX & Bybit
• نظام مراقبة مستمرة طوال اليوم
• دخول فقط في الإشارات القوية المدروسة
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
RESOURCE_SAVER_MODE = True
MIN_CANDLES = 180
BASE_SLEEP = 12
NEAR_CLOSE_S = 3
MAX_LOOP_FREQUENCY = 18

# =================== EXCHANGE FEES & SCALP PROFITABILITY ===================
TAKER_FEE_RATE = 0.0006  # 0.06% عمولة
MAKER_FEE_RATE = 0.0002  # 0.02% للملاح
SCALP_EXTRA_NET_PCT = 0.004  # 0.4% ربح صافي إضافي
MIN_SCALP_PROFIT_PCT = 2 * TAKER_FEE_RATE + SCALP_EXTRA_NET_PCT  # 0.52% أدنى ربح

# =================== ENHANCED SCALP SECURITY ===================
SCALP_HIGH_CONFIDENCE_THRESHOLD = 0.88  # 88% ثقة كحد أدنى
SCALP_MIN_SCORE_ENHANCED = 26.0  # نقاط عالية جداً للسكالب
SCALP_CONFIRMATION_SIGNALS_REQUIRED = 5  # 5 إشارات تأكيد
SCALP_MIN_VOLUME_RATIO = 1.8  # حجم 1.8x المتوسط

# =================== ZERO REJECTION THEORY SETTINGS ===================
ZERO_REJECTION_MODE = True
ZR_MIN_ZONE_QUALITY = 8.5
ZR_REQUIRED_CONFIRMATIONS = 4
ZR_VOLUME_CONFIRMATION = 2.0
ZR_MOMENTUM_THRESHOLD = 0.85

# =================== DYNAMIC POSITION SIZING ===================
class IntelligentPositionSizer:
    """نظام ذكي لحساب حجم الصفقات يفرق بين السكالب والترند"""
    
    def __init__(self):
        self.trade_history = []
        self.performance_stats = {
            'scalp_wins': 0,
            'scalp_losses': 0,
            'trend_wins': 0,
            'trend_losses': 0
        }
    
    def calculate_adaptive_size(self, balance, current_price, trade_type, confidence, market_strength):
        """حساب حجم ذكي يتكيف مع نوع الصفقة وقوة السوق"""
        
        if trade_type == "trend":
            return self._calculate_trend_size(balance, current_price, confidence, market_strength)
        else:
            return self._calculate_scalp_size(balance, current_price, confidence, market_strength)
    
    def _calculate_trend_size(self, balance, current_price, confidence, market_strength):
        """حجم صفقات الترند - أكبر حجماً وأطول مدة"""
        
        # قاعدة المخاطرة للترند
        base_risk = 0.65  # 65% من رأس المال للترند
        
        if market_strength == "strong":
            strength_multiplier = 1.4
        elif market_strength == "weak":
            strength_multiplier = 0.7
        else:
            strength_multiplier = 1.0
        
        confidence_multiplier = 0.6 + (confidence * 0.6)  # 0.6 إلى 1.2
        
        raw_size = (balance * base_risk * LEVERAGE * strength_multiplier * confidence_multiplier) / current_price
        
        # تطبيق الحد الأدنى والخطوة
        final_qty = self._normalize_quantity(raw_size)
        
        log_i(f"📊 حجم الترند الذكي:")
        log_i(f"   الرصيد: ${balance:.2f}")
        log_i(f"   المخاطرة: {base_risk*100}%")
        log_i(f"   مضاعف القوة: {strength_multiplier:.2f}")
        log_i(f"   مضاعف الثقة: {confidence_multiplier:.2f}")
        log_i(f"   الحجم النهائي: {final_qty:.4f}")
        
        return final_qty
    
    def _calculate_scalp_size(self, balance, current_price, confidence, market_strength):
        """حجم صفقات السكالب - يركز على الربحية بعد العمولة"""
        
        # قاعدة مخاطرة أصغر للسكالب
        base_risk = 0.45  # 45% من رأس المال للسكالب
        
        # حساب الحد الأدنى للحجم لضمان الربحية
        min_trade_value = 20  # أقل قيمة صفقة $20 لضمان الربحية
        min_qty_by_value = min_trade_value / current_price
        
        if market_strength == "strong":
            strength_multiplier = 1.2
        else:
            strength_multiplier = 0.9
        
        confidence_multiplier = 0.7 + (confidence * 0.4)  # 0.7 إلى 1.1
        
        raw_size = (balance * base_risk * LEVERAGE * strength_multiplier * confidence_multiplier) / current_price
        
        # التأكد من أن الحجم يكفي لتغطية العمولة وتحقيق ربح
        final_qty = max(self._normalize_quantity(raw_size), min_qty_by_value)
        
        # حساب الربحية المتوقعة
        trade_value = final_qty * current_price
        total_fees = trade_value * TAKER_FEE_RATE * 2
        min_profit_needed = trade_value * MIN_SCALP_PROFIT_PCT
        
        log_i(f"📊 حجم السكالب الذكي:")
        log_i(f"   الرصيد: ${balance:.2f}")
        log_i(f"   المخاطرة: {base_risk*100}%")
        log_i(f"   القيمة: ${trade_value:.2f}")
        log_i(f"   العمولة: ${total_fees:.4f}")
        log_i(f"   أقل ربح مطلوب: ${min_profit_needed:.4f}")
        log_i(f"   الحجم النهائي: {final_qty:.4f}")
        
        return final_qty
    
    def _normalize_quantity(self, qty):
        """تقريب الكمية حسب متطلبات التداول"""
        if qty <= 0:
            return 0.0
        
        # لـ SUI عادة تكون الخطوة 0.1
        step = 0.1
        min_qty = 1.0
        
        normalized = math.floor(qty / step) * step
        normalized = max(normalized, min_qty)
        
        return float(f"{normalized:.4f}")

# إنشاء محاسب الحجم الذكي
position_sizer = IntelligentPositionSizer()

# =================== CONTINUOUS PRECISION MONITORING ===================
class PrecisionScalpMonitor:
    """نظام مراقبة مستمرة ذكي للدخول في المناطق القوية فقط"""
    
    def __init__(self):
        self.monitoring_active = True
        self.high_quality_signals = []
        self.last_signal_time = 0
        self.signal_cooldown = 300  # 5 دقائق بين الإشارات القوية
        
    def analyze_market_continuously(self, df, council_data, current_price, balance):
        """تحليل سوق مستمر للبحث عن فرص السكالب عالية الجودة"""
        try:
            # 1. فحص جودة المنطقة بنظرية 0 انعكاس
            zone_quality = self._analyze_zero_rejection_zone(df, council_data, current_price)
            
            # 2. فحص قوة الإشارة
            signal_strength = self._analyze_signal_strength(council_data)
            
            # 3. فحص ظروف السوق المناسبة
            market_conditions = self._analyze_market_conditions(df, council_data)
            
            # 4. فحص الربحية
            profitability_ok = self._check_scalp_profitability(current_price, council_data)
            
            # 5. قرار الدخول النهائي
            if (zone_quality['high_quality'] and 
                signal_strength['very_strong'] and 
                market_conditions['favorable'] and 
                profitability_ok and
                self._is_cooldown_over()):
                
                return self._execute_precision_scalp(council_data, current_price, balance, df, zone_quality)
            
            return {
                'monitoring': True,
                'signal_found': False,
                'zone_quality': zone_quality['score'],
                'signal_strength': signal_strength['level'],
                'message': f"🔍 مراقبة مستمرة - جودة: {zone_quality['score']:.1f}/10"
            }
            
        except Exception as e:
            return {
                'monitoring': True,
                'signal_found': False,
                'error': f"خطأ في المراقبة: {e}"
            }
    
    def _analyze_zero_rejection_zone(self, df, council_data, current_price):
        """تحليل المناطق بنظرية 0 انعكاس"""
        try:
            score = 0
            reasons = []
            
            # 1. تحليل SMC المتقدم
            smc_data = council_data.get('analysis', {}).get('smc', {})
            order_blocks = smc_data.get('order_blocks', 0)
            fvgs = smc_data.get('fvgs', 0)
            bos_choch = smc_data.get('bos_choch', {})
            
            if order_blocks >= 1:
                score += 2.5
                reasons.append("✅ Order Blocks نشطة")
            
            if fvgs >= 1:
                score += 2.0
                reasons.append("✅ FVGs نشطة")
            
            if bos_choch.get('bos_bullish') or bos_choch.get('bos_bearish'):
                score += 3.0
                reasons.append("✅ كسر هيكل قوي")
            
            # 2. تحليل العرض والطلب
            supply_demand = council_data.get('analysis', {}).get('supply_demand', {})
            demand_zones = supply_demand.get('demand_zones', [])
            supply_zones = supply_demand.get('supply_zones', [])
            
            active_demand = any(abs(z['price'] - current_price) / current_price < 0.004 for z in demand_zones[:2])
            active_supply = any(abs(z['price'] - current_price) / current_price < 0.004 for z in supply_zones[:2])
            
            if active_demand or active_supply:
                score += 2.5
                reasons.append("✅ مناطق عرض/طلب نشطة")
            
            # 3. تحليل الزخم
            indicators = council_data.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            adx = indicators.get('adx', 0)
            macd_hist = indicators.get('macd_hist', 0)
            
            if (rsi < 25 or rsi > 75) and adx > 22:
                score += 2.0
                reasons.append("✅ زخم انعكاسي قوي")
            
            if abs(macd_hist) > 0.002:
                score += 1.5
                reasons.append("✅ إشارة MACD قوية")
            
            # 4. تحليل الحجم
            volume_ok = self._check_volume_confirmation(df)
            if volume_ok:
                score += 2.0
                reasons.append("✅ تأكيد حجم قوي")
            
            high_quality = score >= ZR_MIN_ZONE_QUALITY
            
            return {
                'score': score,
                'high_quality': high_quality,
                'reasons': reasons,
                'order_blocks': order_blocks,
                'fvgs': fvgs,
                'active_zones': active_demand or active_supply
            }
            
        except Exception as e:
            return {'score': 0, 'high_quality': False, 'reasons': [f"خطأ: {e}"]}
    
    def _analyze_signal_strength(self, council_data):
        """تحليل قوة إشارة المجلس"""
        try:
            score_b = council_data.get('score_b', 0)
            score_s = council_data.get('score_s', 0)
            confidence = council_data.get('confidence', 0)
            
            winning_score = max(score_b, score_s)
            
            if winning_score >= 24 and confidence >= 0.85:
                level = "very_strong"
            elif winning_score >= 20 and confidence >= 0.78:
                level = "strong" 
            else:
                level = "weak"
            
            return {
                'level': level,
                'winning_score': winning_score,
                'confidence': confidence,
                'very_strong': level == "very_strong"
            }
            
        except Exception:
            return {'level': "weak", 'winning_score': 0, 'confidence': 0, 'very_strong': False}
    
    def _analyze_market_conditions(self, df, council_data):
        """تحليل ظروف السوق العامة"""
        try:
            conditions = {
                'favorable': True,
                'reasons': []
            }
            
            # 1. فحص التذبذب
            volatility_data = council_data.get('analysis', {}).get('volatility', {})
            if volatility_data.get('volatility_level') in ['high', 'extreme']:
                conditions['favorable'] = False
                conditions['reasons'].append("تذبذب عالي")
            
            # 2. فحص التلاعب
            manipulation = council_data.get('analysis', {}).get('manipulation', {})
            if manipulation.get('high_volatility_alert'):
                conditions['favorable'] = False
                conditions['reasons'].append("تلاعب مرتفع")
            
            # 3. فحص الانتشار
            spread = orderbook_spread_bps()
            if spread and spread > 8.0:  # انتشار عالي
                conditions['favorable'] = False
                conditions['reasons'].append(f"انتشار عالي: {spread:.1f}bps")
            
            # 4. فحص وقت الشمعة
            time_to_close = time_to_candle_close(df)
            if time_to_close < 30:  # نهاية الشمعة
                conditions['favorable'] = False
                conditions['reasons'].append("نهاية الشمعة")
            
            return conditions
            
        except Exception:
            return {'favorable': True, 'reasons': []}
    
    def _check_volume_confirmation(self, df):
        """التحقق من تأكيد الحجم"""
        try:
            volume = df['volume'].astype(float)
            current_volume = volume.iloc[-1]
            avg_volume = volume.tail(20).mean()
            
            return current_volume > avg_volume * ZR_VOLUME_CONFIRMATION
        except Exception:
            return False
    
    def _check_scalp_profitability(self, current_price, council_data):
        """فحص ربحية صفقة السكالب"""
        try:
            # تحديد الاتجاه
            if council_data.get('score_b', 0) > council_data.get('score_s', 0):
                side = "buy"
                target_price = current_price * (1 + MIN_SCALP_PROFIT_PCT + 0.002)  # هامش إضافي
            else:
                side = "sell" 
                target_price = current_price * (1 - MIN_SCALP_PROFIT_PCT - 0.002)
            
            # حساب الربحية
            is_profitable, gross_pct, min_required = is_scalp_profitable_enough(
                current_price, target_price, side
            )
            
            return is_profitable
            
        except Exception:
            return False
    
    def _is_cooldown_over(self):
        """التحقق من انتهاء فترة التبريد"""
        current_time = time.time()
        return (current_time - self.last_signal_time) >= self.signal_cooldown
    
    def _execute_precision_scalp(self, council_data, current_price, balance, df, zone_quality):
        """تنفيذ سكالب دقيق عالي الجودة"""
        try:
            # تحديد الاتجاه
            if council_data.get('score_b', 0) > council_data.get('score_s', 0):
                side = "buy"
            else:
                side = "sell"
            
            # حساب الحجم الذكي
            position_size = position_sizer.calculate_adaptive_size(
                balance, current_price, "scalp", 
                council_data.get('confidence', 0), "strong"
            )
            
            if position_size <= 0:
                return {
                    'monitoring': True,
                    'signal_found': True,
                    'executed': False,
                    'message': "❌ حجم غير صالح"
                }
            
            # حساب سعر الهدف
            if side == "buy":
                target_price = current_price * (1 + MIN_SCALP_PROFIT_PCT + 0.002)
            else:
                target_price = current_price * (1 - MIN_SCALP_PROFIT_PCT - 0.002)
            
            # تسجيل الإشارة
            self.last_signal_time = time.time()
            self.high_quality_signals.append({
                'timestamp': datetime.now(),
                'side': side,
                'price': current_price,
                'size': position_size,
                'zone_quality': zone_quality['score'],
                'reasons': zone_quality['reasons']
            })
            
            # الاحتفاظ بـ 50 إشارة فقط
            self.high_quality_signals = self.high_quality_signals[-50:]
            
            # التنفيذ
            success = execute_enhanced_scalp_trade(side, current_price, council_data, balance, df)
            
            if success:
                return {
                    'monitoring': True,
                    'signal_found': True,
                    'executed': True,
                    'side': side,
                    'size': position_size,
                    'price': current_price,
                    'message': f"🎯 تم تنفيذ سكالب دقيق: {side.upper()} {position_size:.4f} @ {current_price:.6f}"
                }
            else:
                return {
                    'monitoring': True,
                    'signal_found': True,
                    'executed': False,
                    'message': "❌ فشل التنفيذ"
                }
            
        except Exception as e:
            return {
                'monitoring': True,
                'signal_found': True,
                'executed': False,
                'error': f"خطأ في التنفيذ: {e}"
            }

# إنشاء نظام المراقبة
precision_monitor = PrecisionScalpMonitor()

# =================== ENHANCED SCALP TRADING SYSTEM ===================
def is_scalp_profitable_enough(entry_price: float, target_price: float, side: str) -> tuple[bool, float, float]:
    """
    فحص ربحية صفقة السكالب مع حساب العمولة
    """
    if side == "buy":
        gross_pct = (target_price - entry_price) / entry_price
    else:
        gross_pct = (entry_price - target_price) / entry_price

    fees_roundtrip_pct = 2 * TAKER_FEE_RATE
    min_required_pct = fees_roundtrip_pct + SCALP_EXTRA_NET_PCT

    is_ok = gross_pct >= min_required_pct
    return is_ok, gross_pct, min_required_pct

def enhanced_scalp_approval(council_data: dict, current_price: float, df: pd.DataFrame) -> tuple[bool, list]:
    """
    موافقة محسنة للسكالب - تتطلب إشارات قوية جداً
    """
    reasons = []
    confirmation_signals = 0
    
    indicators = council_data.get('indicators', {})
    analysis = council_data.get('analysis', {})
    
    # 1. ثقة عالية جداً
    if council_data.get('confidence', 0) >= SCALP_HIGH_CONFIDENCE_THRESHOLD:
        confirmation_signals += 1
        reasons.append(f"✅ ثقة عالية: {council_data['confidence']:.2f} >= {SCALP_HIGH_CONFIDENCE_THRESHOLD}")
    else:
        reasons.append(f"❌ ثقة منخفضة: {council_data.get('confidence', 0):.2f}")
    
    # 2. نقاط عالية جداً
    winning_score = max(council_data.get('score_b', 0), council_data.get('score_s', 0))
    if winning_score >= SCALP_MIN_SCORE_ENHANCED:
        confirmation_signals += 1
        reasons.append(f"✅ نقاط عالية: {winning_score:.1f} >= {SCALP_MIN_SCORE_ENHANCED}")
    else:
        reasons.append(f"❌ نقاط منخفضة: {winning_score:.1f}")
    
    # 3. حجم قوي
    volume_ok = check_volume_strength(df)
    if volume_ok:
        confirmation_signals += 1
        reasons.append("✅ حجم قوي مؤكد")
    else:
        reasons.append("❌ حجم ضعيف")
    
    # 4. زخم فوري قوي
    momentum_ok = check_instant_momentum(df, indicators)
    if momentum_ok:
        confirmation_signals += 1
        reasons.append("✅ زخم فوري قوي")
    else:
        reasons.append("❌ زخم ضعيف")
    
    # 5. مناطق SMC نشطة
    smc_ok = check_smc_activation(analysis, current_price)
    if smc_ok:
        confirmation_signals += 1
        reasons.append("✅ مناطق SMC نشطة")
    else:
        reasons.append("❌ مناطق SMC غير نشطة")
    
    # 6. ظروف سوق مناسبة
    market_ok = check_market_conditions_for_scalp(df, council_data)
    if market_ok:
        confirmation_signals += 1
        reasons.append("✅ ظروف سوق مناسبة")
    else:
        reasons.append("❌ ظروف سوق غير مناسبة")
    
    approved = confirmation_signals >= SCALP_CONFIRMATION_SIGNALS_REQUIRED
    
    if approved:
        reasons.append(f"🎯 السكالب معتمد: {confirmation_signals}/{SCALP_CONFIRMATION_SIGNALS_REQUIRED}")
    else:
        reasons.append(f"🚫 السكالب مرفوض: {confirmation_signals}/{SCALP_CONFIRMATION_SIGNALS_REQUIRED}")
    
    return approved, reasons

def check_volume_strength(df):
    """فحص قوة الحجم للسكالب"""
    try:
        volume = df['volume'].astype(float)
        current_volume = volume.iloc[-1]
        avg_volume = volume.tail(20).mean()
        
        return current_volume > avg_volume * SCALP_MIN_VOLUME_RATIO
    except Exception:
        return False

def check_instant_momentum(df: pd.DataFrame, indicators: dict) -> bool:
    """فحص الزخم الفوري القوي"""
    try:
        if len(df) < 10:
            return False
        
        # تحليل آخر 3 شموع
        recent_closes = df['close'].astype(float).tail(3)
        
        # اتجاه قوي في الشموع الأخيرة
        if all(recent_closes.iloc[i] > recent_closes.iloc[i-1] for i in range(1, 3)):
            return True
        if all(recent_closes.iloc[i] < recent_closes.iloc[i-1] for i in range(1, 3)):
            return True
        
        # RSI في مناطق التشبع
        rsi = indicators.get('rsi', 50)
        if (rsi < 25 or rsi > 75):
            return True
        
        return False
    except Exception:
        return False

def check_smc_activation(analysis, current_price):
    """فحص تنشيط مناطق SMC"""
    try:
        smc_data = analysis.get('smc', {})
        
        # فحص Order Blocks النشطة
        order_blocks = smc_data.get('order_blocks', 0)
        if order_blocks >= 1:
            return True
        
        # فحص FVGs النشطة
        fvgs = smc_data.get('fvgs', 0)
        if fvgs >= 1:
            return True
        
        # فحص BOS/CHoCH
        bos_choch = smc_data.get('bos_choch', {})
        if bos_choch.get('bos_bullish') or bos_choch.get('bos_bearish'):
            return True
            
        return False
    except Exception:
        return False

def check_market_conditions_for_scalp(df, council_data):
    """فحص ظروف السوق المناسبة للسكالب"""
    try:
        # فحص التذبذب
        volatility_data = council_data.get('analysis', {}).get('volatility', {})
        if volatility_data.get('volatility_level') in ['high', 'extreme']:
            return False
        
        # فحص الانتشار
        spread = orderbook_spread_bps()
        if spread and spread > 10.0:  # انتشار عالي
            return False
        
        # فحص وقت الشمعة
        time_to_close = time_to_candle_close(df)
        if time_to_close < 45:  # قرب نهاية الشمعة
            return False
            
        return True
    except Exception:
        return False

def execute_enhanced_scalp_trade(side: str, current_price: float, council_data: dict, balance: float, df: pd.DataFrame) -> bool:
    """تنفيذ سكالب محسن مع كل التحققيات"""
    
    # 1. حساب الهدف والربحية
    target_price = calculate_scalp_target_price(current_price, side, MIN_SCALP_PROFIT_PCT + 0.002)
    is_profitable, gross_pct, min_required = is_scalp_profitable_enough(
        current_price, target_price, side
    )
    
    if not is_profitable:
        log_w(f"🚫 السكالب مرفوض - غير مربح:")
        log_w(f"   إجمالي: {gross_pct*100:.3f}% < المطلوب: {min_required*100:.3f}%")
        return False
    
    # 2. الموافقة المحسنة
    approved, approval_reasons = enhanced_scalp_approval(council_data, current_price, df)
    
    if not approved:
        log_w(f"🚫 السكالب مرفوض - فشل الموافقة:")
        for reason in approval_reasons:
            if "❌" in reason or "🚫" in reason:
                log_w(f"   {reason}")
        return False
    
    # 3. حساب الحجم الذكي
    position_size = position_sizer.calculate_adaptive_size(
        balance, current_price, "scalp", council_data["confidence"], "strong"
    )
    
    if position_size <= 0:
        return False
    
    # 4. تسجيل أسباب الموافقة
    log_g(f"✅ السكالب معتمد:")
    for reason in approval_reasons:
        if "✅" in reason or "🎯" in reason:
            log_g(f"   {reason}")
    
    # 5. التنفيذ
    success = execute_professional_trade(
        side, current_price, position_size, council_data, {
            "market_phase": "enhanced_scalp",
            "target_price": target_price,
            "expected_net_pct": gross_pct - (2 * TAKER_FEE_RATE),
            "approval_signals": len([r for r in approval_reasons if "✅" in r])
        }
    )
    
    if success:
        log_g(f"🎯 تم تنفيذ السكالب:")
        log_g(f"   الدخول: {current_price:.6f}")
        log_g(f"   الهدف: {target_price:.6f}")
        log_g(f"   الربح المتوقع: {(gross_pct - (2 * TAKER_FEE_RATE))*100:.3f}%")
        log_g(f"   الحجم: {position_size:.4f}")
        log_g(f"   الثقة: {council_data.get('confidence', 0):.2f}")
        
        # تحديث الحالة
        STATE.update({
            "scalp_target": target_price,
            "min_required_pct": min_required,
            "expected_gross_pct": gross_pct,
            "enhanced_scalp": True,
            "approval_reasons": approval_reasons
        })
    
    return success

# =================== INTELLIGENT TRADE CLASSIFICATION ===================
class IntelligentTradeClassifier:
    """مصنف ذكي للتفريق بين صفقات السكالب والترند"""
    
    def __init__(self):
        self.trade_history = []
        
    def classify_trade_intelligently(self, council_data, df, current_price):
        """تصنيف ذكي للصفقة بناء على ظروف السوق"""
        try:
            indicators = council_data.get('indicators', {})
            analysis = council_data.get('analysis', {})
            
            trend_strength = self._calculate_trend_strength(indicators, df)
            breakout_conditions = self._analyze_breakout_conditions(analysis, df, current_price)
            momentum_analysis = self._analyze_momentum_conditions(df, indicators)
            
            trend_score = 0
            scalp_score = 0
            
            # تحليل الترند
            if trend_strength >= 0.7:
                trend_score += 3
            if breakout_conditions.get('strong_breakout'):
                trend_score += 3
            if momentum_analysis.get('sustained_momentum'):
                trend_score += 2
            
            # تحليل السكالب
            if trend_strength <= 0.4:
                scalp_score += 3
            if not breakout_conditions.get('strong_breakout'):
                scalp_score += 2
            if momentum_analysis.get('quick_momentum'):
                scalp_score += 2
            if council_data.get('confidence', 0) > 0.82:
                scalp_score += 2
            
            # اتخاذ القرار
            if trend_score >= 7 and trend_score > scalp_score:
                trade_type = "trend"
                reason = "🚀 ترند قوي - فرصة ركوب ترند محترف"
            elif scalp_score >= 6 and scalp_score > trend_score:
                trade_type = "scalp"
                reason = "⚡ ظروف سكالب مثالية - حركة سريعة"
            else:
                trade_type = "scalp"  # افتراضي للسكالب الآمن
                reason = "🔄 سوق جانبي - سكالب آمن"
            
            log_i(f"🎯 التصنيف الذكي: {trade_type.upper()}")
            log_i(f"   نقاط الترند: {trend_score} | نقاط السكالب: {scalp_score}")
            log_i(f"   السبب: {reason}")
            
            return {
                "trade_type": trade_type,
                "trend_score": trend_score,
                "scalp_score": scalp_score,
                "reason": reason
            }
            
        except Exception as e:
            log_w(f"خطأ في التصنيف: {e}")
            return {"trade_type": "scalp", "reason": f"Error: {e}"}
    
    def _calculate_trend_strength(self, indicators, df):
        """حساب قوة الترند"""
        try:
            adx = indicators.get('adx', 0)
            plus_di = indicators.get('plus_di', 0)
            minus_di = indicators.get('minus_di', 0)
            
            close = df['close'].astype(float)
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            # اتجاه المتوسطات
            trend_direction = 1 if sma_20.iloc[-1] > sma_50.iloc[-1] else -1
            
            adx_strength = min(adx / 50.0, 1.0)
            di_strength = min(abs(plus_di - minus_di) / 30.0, 1.0)
            
            trend_strength = (adx_strength * 0.4 + di_strength * 0.3 + 0.3)
            
            return trend_strength
            
        except Exception:
            return 0.5
    
    def _analyze_breakout_conditions(self, analysis, df, current_price):
        """تحليل ظروف الاختراق"""
        try:
            price_testing = analysis.get('price_testing', {})
            smc_data = analysis.get('smc', {})
            
            strong_breakout = False
            
            if (price_testing.get('breakout_confirmed') and 
                price_testing.get('breakout_strength') == 'strong'):
                strong_breakout = True
            
            elif (smc_data.get('bos_choch', {}).get('bos_bullish') or 
                  smc_data.get('bos_choch', {}).get('bos_bearish')):
                strong_breakout = True
            
            return {"strong_breakout": strong_breakout}
            
        except Exception:
            return {"strong_breakout": False}
    
    def _analyze_momentum_conditions(self, df, indicators):
        """تحليل ظروف الزخم"""
        try:
            close = df['close'].astype(float)
            
            # زخم سريع
            recent_moves = []
            for i in range(-3, 0):
                move_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * 100
                recent_moves.append(abs(move_pct))
            
            quick_momentum = all(move > 0.1 for move in recent_moves)
            
            # زخم مستمر
            medium_moves = []
            for i in range(-8, 0):
                move_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * 100
                medium_moves.append(move_pct)
            
            sustained_momentum = sum(1 for move in medium_moves if abs(move) > 0.08) >= 5
            
            return {
                "quick_momentum": quick_momentum,
                "sustained_momentum": sustained_momentum
            }
            
        except Exception:
            return {"quick_momentum": False, "sustained_momentum": False}

# إنشاء المصنف الذكي
trade_classifier = IntelligentTradeClassifier()

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

# إعدادات التنفيذ
EXECUTE_ORDERS = True
DRY_RUN = False

BOT_VERSION = f"SUI ULTRA PRO AI v12.0 — {EXCHANGE_NAME.upper()} - PRECISION SCALP + ZERO REJECTION"
print("🚀 Booting:", BOT_VERSION, flush=True)

# =================== SETTINGS ===================
SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = int(os.getenv("LEVERAGE", 15))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))

# ... (استمرار الكود مع بقية الدوال والأنظمة)

# =================== PROFESSIONAL TRADING LOOP ===================
def professional_trading_loop():
    """الحلقة الرئيسية للتداول المحترف مع نظام السكالب الذكي"""
    
    log_banner("بدء تشغيل البوت المحترف - نظام السكالب الذكي")
    log_i(f"🤖 إصدار البوت: {BOT_VERSION}")
    log_i(f"💱 المنصة: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 الزوج: {SYMBOL}")
    log_i(f"⏰ الفترة: {INTERVAL}")
    log_i(f"🎯 الرافعة: {LEVERAGE}x")
    log_i(f"📊 المخاطرة: {RISK_ALLOC*100}%")
    log_i(f"🛡️ نظام السكالب الآمن: نشط")
    log_i(f"🎯 نظرية 0 انعكاس: نشطة")
    log_i(f"📊 المراقبة المستمرة: نشطة")
    
    while True:
        try:
            # جمع البيانات
            balance = balance_usdt()
            current_price = price_now()
            df = fetch_ohlcv(limit=200)
            
            if df.empty or current_price is None:
                time.sleep(BASE_SLEEP)
                continue
            
            # قرار المجلس المحترف
            council_data = ultra_professional_council_ai(df)
            
            # تحديث الحالة
            STATE["last_council"] = council_data
            
            # المراقبة المستمرة للسكالب الدقيق
            if not STATE["open"] and ZERO_REJECTION_MODE:
                monitor_result = precision_monitor.analyze_market_continuously(
                    df, council_data, current_price, balance
                )
                
                if monitor_result.get('signal_found') and monitor_result.get('executed'):
                    log_g(f"🎯 تم تنفيذ سكالب دقيق via المراقبة المستمرة")
                    time.sleep(SCALP_COOLDOWN_SEC)
                    continue
            
            # إدارة المركز المفتوح
            if STATE["open"]:
                STATE["bars"] += 1
                manage_professional_position(df, council_data, current_price)
            
            # فتح صفقات جديدة
            if not STATE["open"]:
                signal_side = None
                
                # شروط دخول صارمة
                min_score = 20.0
                min_confidence = 0.80
                
                if (council_data["score_b"] > council_data["score_s"] and 
                    council_data["score_b"] >= min_score and 
                    council_data["confidence"] >= min_confidence):
                    signal_side = "buy"
                elif (council_data["score_s"] > council_data["score_b"] and 
                      council_data["score_s"] >= min_score and 
                      council_data["confidence"] >= min_confidence):
                    signal_side = "sell"
                
                if signal_side:
                    # التصنيف الذكي للصفقة
                    classification = trade_classifier.classify_trade_intelligently(
                        council_data, df, current_price
                    )
                    
                    trade_type = classification["trade_type"]
                    
                    if trade_type == "scalp":
                        # استخدام نظام السكالب المحسن
                        execute_enhanced_scalp_trade(signal_side, current_price, council_data, balance, df)
                    else:
                        # صفقات الترند
                        position_size = position_sizer.calculate_adaptive_size(
                            balance, current_price, "trend", 
                            council_data["confidence"], "strong"
                        )
                        
                        if position_size > 0:
                            execute_professional_trade(
                                signal_side, current_price, position_size, council_data, {
                                    "market_phase": "trend",
                                    "trade_type": "trend"
                                }
                            )
            
            # الانتظار للدورة التالية
            sleep_time = NEAR_CLOSE_S if STATE["open"] else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            log_e(f"❌ خطأ في الحلقة الرئيسية: {e}")
            time.sleep(BASE_SLEEP * 2)

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
    "trade_type": None, "profit_targets": [],
    "scalp_target": None, "min_required_pct": None, 
    "expected_gross_pct": None, "enhanced_scalp": False,
    "approval_reasons": None
}

# ... (استمرار الكود مع بقية الدوال المساعدة والتنفيذ)

if __name__ == "__main__":
    # إعداد معالجات الإشارات
    def signal_handler(signum, frame):
        log_i(f"🛑 إيقاف البوت...")
        save_state(STATE)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # بدء التشغيل
    if startup_sequence():
        import threading
        
        # بدء خيط التداول
        trading_thread = threading.Thread(target=professional_trading_loop, daemon=True)
        trading_thread.start()
        
        log_g(f"🌐 بدء السيرفر على المنفذ {PORT}")
        
        # تشغيل سيرفر الويب
        try:
            from flask import Flask
            app = Flask(__name__)
            
            @app.route("/")
            def home():
                return f"""
                <html>
                    <head><title>SUI ULTRA PRO AI - نظام السكالب الذكي</title></head>
                    <body>
                        <h1>🚀 SUI ULTRA PRO AI BOT - نظام السكالب الذكي</h1>
                        <p><strong>الإصدار:</strong> {BOT_VERSION}</p>
                        <p><strong>الحالة:</strong> {'🟢 يعمل' if MODE_LIVE else '🟡 تجريبي'}</p>
                        <p><strong>المركز:</strong> {'🟢 مفتوح' if STATE['open'] else '🔴 مغلق'}</p>
                        <p><strong>نظام السكالب الآمن:</strong> 🟢 نشط</p>
                        <p><strong>نظرية 0 انعكاس:</strong> 🟢 نشطة</p>
                    </body>
                </html>
                """
            
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ خطأ في سيرفر الويب: {e}")
    else:
        log_e("❌ فشل بدء التشغيل - الرجاء التحقق من الإعدادات")
