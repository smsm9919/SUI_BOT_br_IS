# talib_shim.py - بديل TA-Lib خفيف للتشغيل على Render/GitHub
import numpy as np
import pandas as pd

try:
    import ta as pta
except ImportError as e:
    raise ImportError("مكتبة 'ta' مطلوبة. أضفها إلى requirements.txt") from e


class _TalibShim:
    """بديل TA-Lib متوافق مع دوال البوت"""
    
    @staticmethod
    def _to_series(data):
        if isinstance(data, pd.Series):
            return data.astype(float)
        return pd.Series(np.asarray(data, dtype=float))
    
    # === المتوسطات المتحركة ===
    @staticmethod
    def SMA(close, timeperiod=14):
        s = _TalibShim._to_series(close)
        return s.rolling(window=timeperiod, min_periods=timeperiod).mean().values
    
    @staticmethod
    def EMA(close, timeperiod=14):
        s = _TalibShim._to_series(close)
        return s.ewm(span=timeperiod, adjust=False).mean().values
    
    # === مؤشرات الزخم ===
    @staticmethod
    def RSI(close, timeperiod=14):
        s = _TalibShim._to_series(close)
        return pta.momentum.RSIIndicator(s, window=timeperiod).rsi().values
    
    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        s = _TalibShim._to_series(close)
        ind = pta.trend.MACD(s, window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
        return (
            ind.macd().values,
            ind.macd_signal().values,
            ind.macd_diff().values
        )
    
    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        h = _TalibShim._to_series(high)
        l = _TalibShim._to_series(low) 
        c = _TalibShim._to_series(close)
        ind = pta.momentum.StochasticOscillator(
            high=h, low=l, close=c, window=fastk_period, smooth_window=slowk_period
        )
        k = ind.stoch()
        d = k.rolling(slowd_period).mean()
        return k.values, d.values
    
    # === مؤشرات التقلب ===
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        h = _TalibShim._to_series(high)
        l = _TalibShim._to_series(low)
        c = _TalibShim._to_series(close)
        ind = pta.volatility.AverageTrueRange(h, l, c, window=timeperiod)
        return ind.average_true_range().values
    
    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        s = _TalibShim._to_series(close)
        ind = pta.volatility.BollingerBands(s, window=timeperiod, window_dev=nbdevup)
        return (
            ind.bollinger_hband().values,
            ind.bollinger_mavg().values,
            ind.bollinger_lband().values
        )
    
    # === مؤشرات الاتجاه ===
    @staticmethod
    def ADX(high, low, close, timeperiod=14):
        h = _TalibShim._to_series(high)
        l = _TalibShim._to_series(low)
        c = _TalibShim._to_series(close)
        ind = pta.trend.ADXIndicator(h, l, c, window=timeperiod)
        return ind.adx().values
    
    @staticmethod
    def PLUS_DI(high, low, close, timeperiod=14):
        h = _TalibShim._to_series(high)
        l = _TalibShim._to_series(low)
        c = _TalibShim._to_series(close)
        ind = pta.trend.ADXIndicator(h, l, c, window=timeperiod)
        return ind.adx_pos().values
    
    @staticmethod
    def MINUS_DI(high, low, close, timeperiod=14):
        h = _TalibShim._to_series(high)
        l = _TalibShim._to_series(low)
        c = _TalibShim._to_series(close)
        ind = pta.trend.ADXIndicator(h, l, c, window=timeperiod)
        return ind.adx_neg().values
    
    # === مؤشرات الحجم ===
    @staticmethod
    def OBV(close, volume):
        c = _TalibShim._to_series(close)
        v = _TalibShim._to_series(volume)
        ind = pta.volume.OnBalanceVolumeIndicator(c, v)
        return ind.on_balance_volume().values


# نعرض واجهة متوافقة مع talib
talib = _TalibShim
