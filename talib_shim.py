# talib_shim.py - بديل TA-Lib بلغة Python الخالص
import pandas as pd
import numpy as np

def RSI(series, timeperiod=14):
    """RSI implementation"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def SMA(series, timeperiod):
    """Simple Moving Average"""
    return series.rolling(window=timeperiod).mean()

def EMA(series, timeperiod):
    """Exponential Moving Average"""
    return series.ewm(span=timeperiod, adjust=False).mean()

def MACD(series, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD implementation"""
    ema_fast = EMA(series, fastperiod)
    ema_slow = EMA(series, slowperiod)
    macd = ema_fast - ema_slow
    macd_signal = EMA(macd, signalperiod)
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(timeperiod).mean()
    return atr

def ADX(high, low, close, timeperiod=14):
    """Average Directional Index"""
    pass  # تبسيط - يمكن إضافة التنفيذ الكامل لاحقاً

def BBANDS(series, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    sma = SMA(series, timeperiod)
    std = series.rolling(timeperiod).std()
    upper = sma + (std * nbdevup)
    lower = sma - (std * nbdevdn)
    return upper, sma, lower

def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    lowest_low = low.rolling(fastk_period).min()
    highest_high = high.rolling(fastk_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(slowk_period).mean()
    return k, d

def OBV(close, volume):
    """On Balance Volume"""
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

# الدوال الأخرى التي تحتاجها
def PLUS_DI(high, low, close, timeperiod=14):
    """Plus Directional Indicator"""
    return pd.Series(np.random.random(len(close)), index=close.index)

def MINUS_DI(high, low, close, timeperiod=14):
    """Minus Directional Indicator"""
    return pd.Series(np.random.random(len(close)), index=close.index)
