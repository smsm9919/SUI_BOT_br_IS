# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO — Council + FVG + SMC(OB) + OrderBook Flow
Exchange: Bybit/BingX (USDT Perp via CCXT)
"""

import os, time, math, json, logging, traceback, statistics
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation

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

BOT_VERSION = f"SUI ULTRA PRO v6.2 — {EXCHANGE_NAME.upper()}"
print("🚀 BOOT:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True

# =================== SETTINGS ===================
# Range Filter (لو احتجته لاحقًا)
RF_PERIOD=18; RF_MULT=3.0; RF_HYST_BPS=6.0

# Indicators
RSI_LEN=14; ADX_LEN=14; ATR_LEN=14
ICHIMOKU=(9,26,52); BB=(20,2); MACD=(12,26,9); STOCH=(14,3,3)

# Profit Mgmt
SCALP_TP_TARGETS=[0.55,0.90,1.20]
SCALP_TP_FRACTIONS=[0.50,0.30,0.20]
TREND_TP_TARGETS=[0.60,1.50,2.50,4.00]
TREND_TP_FRACTIONS=[0.30,0.30,0.25,0.15]
BREAKEVEN_AFTER=0.35   # %
TRAIL_ACTIVATE_PCT=1.0 # %
ATR_TRAIL_MULT=1.8

# Council
TREND_STRENGTH_THRESHOLD=25
VOLUME_CONFIRMATION_THRESHOLD=1.2
REQUIRED_CONFIRMATIONS=4
GOLDEN_ENTRY_SCORE=7.0
GOLDEN_ENTRY_ADX=22.0

# Spread Gate (bps = 0.01%)
MAX_SPREAD_BPS=float(os.getenv("MAX_SPREAD_BPS", "8"))   # 8bps=0.08%

# FVG/SMC/Flow
FVG_LOOKBACK=60; FVG_MIN_SIZE=0.0015; FVG_NEAR_PCT=0.15
OB_LOOKBACK=80; OB_MIN_DISP=1.0; OB_NEAR_PCT=0.20
ORDERBOOK_LIMIT=50; IMBALANCE_THRESH=0.20; WALL_X_SIGMA=2.2

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
            return {"positionSide":"Long" if side=="buy" else "Short","reduceOnly":is_close}
        return {"positionSide":"Both","reduceOnly":is_close}
    else:
        if POSITION_MODE=="hedge":
            return {"positionSide":"LONG" if side=="buy" else "SHORT","reduceOnly":is_close}
        return {"positionSide":"BOTH","reduceOnly":is_close}

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
STATE = {"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,
         "bars":0,"trail":None,"breakeven":None,"tp_levels_hit":[],
         "highest_profit_pct":0.0,"profit_targets_achieved":0,"mode":None}

def save_state(st:dict):
    try:
        st["ts"]=int(time.time()); st["compound_pnl"]=round(float(compound_pnl),6)
        with open(STATE_PATH,"w",encoding="utf-8") as f: json.dump(st,f,ensure_ascii=False,indent=2)
        log_i(f"state saved (compound_pnl={compound_pnl:.4f})")
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
    # aggregate quick view
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

# =================== INDICATORS ===================
def wilder_ema(s:pd.Series,n:int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df:pd.DataFrame):
    if len(df)<max(ATR_LEN,RSI_LEN,ADX_LEN)+2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
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
    i=len(df)-1
    return {"rsi":float(rsi.iloc[i]),"plus_di":float(plus_di.iloc[i]),
            "minus_di":float(minus_di.iloc[i]),"dx":float(dx.iloc[i]),
            "adx":float(adx.iloc[i]),"atr":float(atr.iloc[i])}

def rsi_ma_context(df:pd.DataFrame):
    RSI_MA=9; NEUT=(45,55); PERSIST=3
    if len(df)<max(RSI_MA,14): return {"rsi":50,"rsi_ma":50,"cross":"none","trendZ":"none","in_chop":True}
    c=df["close"].astype(float); delta=c.diff(); up=delta.clip(lower=0); dn=(-delta).clip(lower=0)
    ru=up.ewm(span=14, adjust=False).mean(); rd=dn.ewm(span=14, adjust=False).mean()
    rsi=100-(100/(1+(ru/rd.replace(0,1e-12)))); rsi_ma=rsi.rolling(RSI_MA, min_periods=1).mean()
    cross="none"
    if len(rsi)>=2:
        if (rsi.iloc[-2]<=rsi_ma.iloc[-2]) and (rsi.iloc[-1]>rsi_ma.iloc[-1]): cross="bull"
        elif (rsi.iloc[-2]>=rsi_ma.iloc[-2]) and (rsi.iloc[-1]<rsi_ma.iloc[-1]): cross="bear"
    above=(rsi>rsi_ma); below=(rsi<rsi_ma)
    persist_b=above.tail(PERSIST).all() if len(above)>=PERSIST else False
    persist_s=below.tail(PERSIST).all() if len(below)>=PERSIST else False
    cur=float(rsi.iloc[-1]); in_chop = NEUT[0]<=cur<=NEUT[1]
    return {"rsi":cur,"rsi_ma":float(rsi_ma.iloc[-1]),"cross":cross,"trendZ":"bull" if persist_b else ("bear" if persist_s else "none"),"in_chop":in_chop}

def super_trend(df, period=10, multiplier=3):
    try:
        h=df['high'].astype(float); l=df['low'].astype(float); c=df['close'].astype(float)
        tr=pd.concat([(h-l),(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
        atr=tr.ewm(span=period, adjust=False).mean()
        hl2=(h+l)/2; ub=hl2+(multiplier*atr); lb=hl2-(multiplier*atr)
        st=[ub.iloc[0]]; trend=[-1]
        for i in range(1,len(df)):
            if (st[i-1]==ub.iloc[i-1] and c.iloc[i] <= ub.iloc[i]) or (st[i-1]==lb.iloc[i-1] and c.iloc[i] < lb.iloc[i]):
                st.append(ub.iloc[i]); trend.append(-1)
            else:
                st.append(lb.iloc[i]); trend.append(1)
        strength=abs(c.iloc[-1]-st[-1])/max(atr.iloc[-1],1e-9)
        sig="buy" if trend[-1]>0 else "sell"
        return {"trend":trend[-1],"strength":strength,"value":st[-1],"signal":sig}
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
        return {"tenkan":float(ten.iloc[-1]),"kijun":float(kij.iloc[-1]),"senkou_a":float(sA.iloc[-1]),"senkou_b":float(sB.iloc[-1]),
                "above_cloud":above,"below_cloud":below,"cloud_green":sA.iloc[-1]>sB.iloc[-1],"signal":sig,
                "strength":abs(price-kij.iloc[-1])/max(price,1e-9)*100}
    except Exception: return {"signal":"neutral","strength":0}

def bollinger_bands_advanced(df, period=20, std=2):
    try:
        c=df['close'].astype(float); sma=c.rolling(period).mean(); sd=c.rolling(period).std()
        ub=sma+(sd*std); lb=sma-(sd*std); cur=c.iloc[-1]
        bw=((ub-lb)/sma.replace(0,1e-12)*100).iloc[-1]; pctB=(cur-lb.iloc[-1])/max(ub.iloc[-1]-lb.iloc[-1],1e-12)
        squeeze=bw<10
        sig="buy" if (cur<=lb.iloc[-1] and not squeeze) else ("sell" if (cur>=ub.iloc[-1] and not squeeze) else "neutral")
        return {"upper":float(ub.iloc[-1]),"middle":float(sma.iloc[-1]),"lower":float(lb.iloc[-1]),
                "bandwidth":float(bw),"percent_b":float(pctB),"squeeze":bool(squeeze),"signal":sig,
                "strength":abs(pctB-0.5)*2}
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
        return {"macd":float(macd.iloc[-1]),"signal_line":float(macd_sig.iloc[-1]),
                "histogram":float(hist.iloc[-1]),"signal":sig,"strength":float(strength)}
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
        break_high=cur>prev_h; break_low=cur<prev_l; vol_ok=vol>avg*VOLUME_CONFIRMATION_THRESHOLD
        if break_high and vol_ok: return {"signal":"strong_buy","strength":(cur-prev_h)/max(prev_h,1e-9)*100,
                                          "break_high":True,"break_low":False,"volume_confirmation":True}
        if break_low and vol_ok:  return {"signal":"strong_sell","strength":(prev_l-cur)/max(prev_l,1e-9)*100,
                                          "break_high":False,"break_low":True,"volume_confirmation":True}
        return {"signal":"neutral","strength":0,"break_high":False,"break_low":False,"volume_confirmation":False}
    except Exception: return {"signal":"neutral","strength":0}

def smart_money_flow(df, period=20):
    try:
        h=df['high'].astype(float); l=df['low'].astype(float); c=df['close'].astype(float); v=df['volume'].astype(float)
        mfm=((c-l)-(h-c))/ (h-l).replace(0,1e-12); mfv=mfm*v
        pos=mfv.where(mfv>0,0).rolling(period).sum(); neg=abs(mfv.where(mfv<0,0)).rolling(period).sum()
        mfi=100-(100/(1+pos/ neg.replace(0,1e-12)))
        adl=(((c-l)-(h-c))/ (h-l).replace(0,1e-12)*v).cumsum(); trend="rising" if adl.iloc[-1]>adl.iloc[-2] else "falling"
        return {"mfi":float(mfi.iloc[-1]),"smart_money_bullish":trend=="rising" and mfi.iloc[-1]<30,
                "smart_money_bearish":trend=="falling" and mfi.iloc[-1]>70,"strength":abs(mfi.iloc[-1]-50)/50}
    except Exception: return {"mfi":50,"smart_money_bullish":False,"smart_money_bearish":False,"strength":0}

# -------- FVG / OB / FLOW --------
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

# =================== COUNCIL ===================
def ultra_intelligent_council(df, mtf_meta=None):
    try:
        st = super_trend(df); ichi = ichimoku_cloud(df); bb = bollinger_bands_advanced(df)
        macd = volume_weighted_macd(df); stoch = stochastic_rsi_advanced(df)
        structure = market_structure_break_advanced(df); money = smart_money_flow(df)
        ind_basic = compute_indicators(df); rsi_ctx = rsi_ma_context(df)
        fvg = detect_fvg(df); obx = detect_order_blocks(df); flow = fetch_orderbook_metrics()

        votes_b=votes_s=0; score_b=score_s=0.0; logs=[]; confirms=[]

        # SuperTrend
        if st["signal"]=="buy": votes_b+=2; score_b+=st["strength"]*2; logs.append(f"🚀 ST buy ({st['strength']:.2f})"); confirms.append("SuperTrend")
        if st["signal"]=="sell": votes_s+=2; score_s+=st["strength"]*2; logs.append(f"💥 ST sell ({st['strength']:.2f})"); confirms.append("SuperTrend")
        # Ichi
        if ichi["signal"]=="buy": votes_b+=2; score_b+=ichi["strength"]*1.5; logs.append(f"☁️ Ichi + ({ichi['strength']:.2f}%)"); confirms.append("Ichimoku")
        if ichi["signal"]=="sell": votes_s+=2; score_s+=ichi["strength"]*1.5; logs.append(f"☁️ Ichi - ({ichi['strength']:.2f}%)"); confirms.append("Ichimoku")
        # BB
        if bb["signal"]=="buy" and not bb["squeeze"]: votes_b+=2; score_b+=bb["strength"]*2; logs.append("📊 BB buy from lower"); confirms.append("Bollinger")
        if bb["signal"]=="sell" and not bb["squeeze"]: votes_s+=2; score_s+=bb["strength"]*2; logs.append("📊 BB sell from upper"); confirms.append("Bollinger")
        # MACD
        if macd["signal"]=="buy": votes_b+=2; score_b+=macd["strength"]*1.5; logs.append("📈 MACD bull"); confirms.append("MACD")
        if macd["signal"]=="sell": votes_s+=2; score_s+=macd["strength"]*1.5; logs.append("📉 MACD bear"); confirms.append("MACD")
        # StochRSI
        if stoch["signal"]=="buy": votes_b+=1; score_b+=stoch["strength"]*1.2; logs.append("🎯 StochRSI bull"); confirms.append("StochRSI")
        if stoch["signal"]=="sell": votes_s+=1; score_s+=stoch["strength"]*1.2; logs.append("🎯 StochRSI bear"); confirms.append("StochRSI")
        # Structure
        if structure["signal"]=="strong_buy": votes_b+=3; score_b+=structure["strength"]*0.5; logs.append(f"🔄 Break↑ {structure['strength']:.2f}%"); confirms.append("Structure")
        if structure["signal"]=="strong_sell": votes_s+=3; score_s+=structure["strength"]*0.5; logs.append(f"🔄 Break↓ {structure['strength']:.2f}%"); confirms.append("Structure")
        # Smart Money
        if money["smart_money_bullish"]: votes_b+=2; score_b+=money["strength"]*1.8; logs.append("💰 SmartMoney buy"); confirms.append("SmartMoney")
        if money["smart_money_bearish"]: votes_s+=2; score_s+=money["strength"]*1.8; logs.append("💰 SmartMoney sell"); confirms.append("SmartMoney")
        # MTF meta
        if mtf_meta:
            if mtf_meta["overall_signal"]=="bullish": votes_b+=2; score_b+=mtf_meta["confidence"]*2; logs.append(f"⏰ MTF bull ({mtf_meta['bull_count']})"); confirms.append("MultiTF")
            if mtf_meta["overall_signal"]=="bearish": votes_s+=2; score_s+=mtf_meta["confidence"]*2; logs.append(f"⏰ MTF bear ({mtf_meta['bear_count']})"); confirms.append("MultiTF")
        # Trend boost
        if ind_basic["adx"]>TREND_STRENGTH_THRESHOLD:
            if votes_b>votes_s: score_b*=1.3; logs.append(f"🔥 Trend+ ADX={ind_basic['adx']:.1f}")
            elif votes_s>votes_b: score_s*=1.3; logs.append(f"🔥 Trend- ADX={ind_basic['adx']:.1f}")
        # Volume boost
        v=df['volume'].astype(float); cur_v=float(v.iloc[-1]); avg=v.tail(20).mean()
        if cur_v>avg*VOLUME_CONFIRMATION_THRESHOLD:
            if votes_b>votes_s: score_b*=1.2; logs.append("📈 Volume↑ supports BUY")
            elif votes_s>votes_b: score_s*=1.2; logs.append("📉 Volume↑ supports SELL")
        # FVG
        if fvg.get("has"):
            if fvg["type"]=="bull": votes_b+=2; score_b+=1.0+(0.5 if fvg.get("near") else 0); logs.append(f"🧩 FVG bull (near={fvg.get('near')})"); confirms.append("FVG")
            if fvg["type"]=="bear": votes_s+=2; score_s+=1.0+(0.5 if fvg.get("near") else 0); logs.append(f"🧩 FVG bear (near={fvg.get('near')})"); confirms.append("FVG")
        # OB
        if obx.get("has"):
            if obx["type"]=="bull": votes_b+=3; score_b+=1.8 if obx.get("near") else 1.0; logs.append(f"🏦 OB bull (near={obx.get('near')})"); confirms.append("OrderBlock")
            if obx["type"]=="bear": votes_s+=3; score_s+=1.8 if obx.get("near") else 1.0; logs.append(f"🏦 OB bear (near={obx.get('near')})"); confirms.append("OrderBlock")
        # FLOW
        if flow.get("signal")=="buy": votes_b+=2; score_b+=1.2+abs(flow["imbalance"]); logs.append(f"📊 FLOW buy imb={flow['imbalance']:.2f}"); confirms.append("Flow")
        if flow.get("signal")=="sell": votes_s+=2; score_s+=1.2+abs(flow["imbalance"]); logs.append(f"📊 FLOW sell imb={flow['imbalance']:.2f}"); confirms.append("Flow")

        # Golden Zone gate (اختياري: اشتراط ADX)
        if ind_basic["adx"]<GOLDEN_ENTRY_ADX: pass

        return {
            "votes_buy":votes_b,"votes_sell":votes_s,
            "score_buy":round(score_b,2),"score_sell":round(score_s,2),
            "logs":logs,"confirmation_signals":confirms,
            "indicators":{
                "super_trend":st,"ichimoku":ichi,"bollinger":bb,"macd":macd,"stoch_rsi":stoch,
                "market_structure":structure,"money_flow":money,"basic":ind_basic,"rsi_context":rsi_ctx,
                "fvg":fvg,"order_block":obx,"orderbook_flow":flow
            },
            "mtf_analysis":mtf_meta
        }
    except Exception as e:
        log_w(f"council error: {e}")
        return {"votes_buy":0,"votes_sell":0,"score_buy":0,"score_sell":0,
                "logs":[],"confirmation_signals":[],"indicators":{},"mtf_analysis":None}

# =================== LOG MEDIUM PANEL ===================
def render_medium_log(c):
    try:
        ind=c.get("indicators",{}); basic=ind.get("basic",{})
        st=ind.get("super_trend",{}); ich=ind.get("ichimoku",{}); bb=ind.get("bollinger",{})
        macd=ind.get("macd",{}); stoch=ind.get("stoch_rsi",{}); mstr=ind.get("market_structure",{})
        money=ind.get("money_flow",{}); mtf=c.get("mtf_analysis",{}) or {}
        fvg=ind.get("fvg",{}); obx=ind.get("order_block",{}); flow=ind.get("orderbook_flow",{})

        bal=balance_usdt(); bal_fmt=f"{bal:.2f}" if (bal is not None) else "N/A"
        print(f"💼 BALANCE: {bal_fmt} USDT | 📦 COMPOUND PNL: {compound_pnl:+.4f} USDT", flush=True)

        if STATE.get("open"):
            entry=STATE.get("entry"); entry_fmt=f"{entry:.6f}" if isinstance(entry,(int,float)) else str(entry)
            print(f"🧭 MODE={STATE.get('mode','-').upper()} | POS={STATE.get('side','-').upper()} | PnL={STATE.get('pnl',0.0):.2f}% | TP_hits={int(STATE.get('profit_targets_achieved',0))} | entry={entry_fmt}", flush=True)
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
        fvg_txt = f"{fvg.get('type','-')}{'~' if fvg.get('near') else ''}" if fvg.get("has") else "-"
        obx_txt = f"{obx.get('type','-')}{'~' if obx.get('near') else ''}" if obx.get("has") else "-"
        flow_txt=f"{flow.get('signal','-')}(imb={float(flow.get('imbalance',0)):.2f},Wb={float(flow.get('bid_wall',0)):.1f}/Wa={float(flow.get('ask_wall',0)):.1f})"
        print(f"🧠 SMC: FVG={fvg_txt} | OB={obx_txt} | FLOW={flow_txt}", flush=True)
        print("—"*70, flush=True)
    except Exception as e:
        log_w(f"render_medium_log error: {e}")

# =================== ENTRY/EXIT ===================
def setup_trade_management(mode):
    if mode=="scalp":
        return {"tp_targets":SCALP_TP_TARGETS,"tp_fractions":SCALP_TP_FRACTIONS,
                "be_activate_pct":BREAKEVEN_AFTER/100.0,"trail_activate_pct":TRAIL_ACTIVATE_PCT/100.0,
                "atr_trail_mult":ATR_TRAIL_MULT,"close_aggression":"high","max_tp_levels":len(SCALP_TP_TARGETS)}
    return {"tp_targets":TREND_TP_TARGETS,"tp_fractions":TREND_TP_FRACTIONS,
            "be_activate_pct":BREAKEVEN_AFTER/100.0,"trail_activate_pct":TRAIL_ACTIVATE_PCT/100.0,
            "atr_trail_mult":ATR_TRAIL_MULT,"close_aggression":"medium","max_tp_levels":len(TREND_TP_TARGETS)}

def manage_take_profits(state,current_price,pnl_pct,mgmt,mode):
    if state["qty"]<=0: return
    tps=mgmt["tp_targets"]; frs=mgmt["tp_fractions"]
    if mode=="scalp":
        for i,(tp,frac) in enumerate(zip(tps,frs)):
            if not state.get(f"tp{i+1}_done") and pnl_pct>=tp/100.0:
                close_qty=safe_qty(state["qty"]*frac)
                if close_qty>0:
                    side="sell" if state["side"]=="long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params=exchange_specific_params(side, True)
                            ex.create_order(SYMBOL,"market",side,close_qty,None,params)
                            log_g(f"SCALP TP{i+1} hit ({tp}%)")
                        except Exception as e: log_e(f"Scalp TP{i+1} failed: {e}")
                    state["qty"]=safe_qty(state["qty"]-close_qty); state[f"tp{i+1}_done"]=True
                    state["profit_targets_achieved"]+=1
    else:
        hit=state.get("tp_levels_hit",[False]*len(tps))
        for i,(tp,frac) in enumerate(zip(tps,frs)):
            if not hit[i] and pnl_pct>=tp/100.0:
                close_qty=safe_qty(state["qty"]*frac)
                if close_qty>0:
                    side="sell" if state["side"]=="long" else "buy"
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params=exchange_specific_params(side, True)
                            ex.create_order(SYMBOL,"market",side,close_qty,None,params)
                            log_g(f"TREND TP{i+1} hit ({tp}%)")
                        except Exception as e: log_e(f"Trend TP{i+1} failed: {e}")
                    state["qty"]=safe_qty(state["qty"]-close_qty); hit[i]=True
                    state["profit_targets_achieved"]+=1
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
    if not EXECUTE_ORDERS or DRY_RUN: log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}"); return True
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
            log_g("SIMULATED EXECUTION")
            return True
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
                      "profit_targets_achieved":0,"mode":mode,"management":mgmt,"council_snapshot":council})
        save_state({"in_position":True,"side":STATE["side"].upper(),"entry_price":price,"position_qty":qty,
                    "leverage":LEVERAGE,"mode":mode,"management":mgmt,"council_snapshot":council,
                    "opened_at":int(time.time()),"tp_levels_hit":tp_hit})
        log_g(f"POSITION OPENED: {side.upper()} | mode={mode} | strength={strength:.1f}")
        return True
    return False

def smart_exit_analysis(df, council, price, pnl_pct, side, entry):
    # Council reversal
    cur = ultra_intelligent_council(df)
    cur_sig = "buy" if cur["score_buy"]>cur["score_sell"] else "sell"
    if (side=="long" and cur_sig=="sell") or (side=="short" and cur_sig=="buy"):
        return {"action":"close","reason":"council_reversal"}
    # Profit cap
    if pnl_pct >= (8 if STATE.get("mode")=="trend" else 4):
        return {"action":"close","reason":"excellent_profit"}
    # Momentum loss
    if council["indicators"]["basic"]["adx"]<15 and abs(pnl_pct)>1:
        return {"action":"close","reason":"momentum_loss"}
    return {"action":"hold","reason":"continue"}

def manage_after_entry_enhanced(df, council):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=price_now(); if not px: return
    entry=STATE["entry"]; side=STATE["side"]; qty=STATE["qty"]; mode=STATE.get("mode","trend"); mgmt=STATE.get("management",{})
    pnl_pct=(px-entry)/entry*(100 if side=="long" else -100); STATE["pnl"]=pnl_pct
    if pnl_pct>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=pnl_pct
    manage_take_profits(STATE, px, pnl_pct, mgmt, mode)
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
                side = "long" if "long" in side_raw or "both" in side_raw and STATE.get("side")=="long" else "short"
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
    # تقدير PnL البسيط (من STATE) — يمكن استبداله بقراءة من الإكستشينج إن أردت
    pnl_pct=STATE.get("pnl",0.0)
    bal=balance_usdt() or 0.0
    pnl_usdt = (bal* (pnl_pct/100.0) * 0.0)  # تقدير محافظ — اتركه 0 لو غير مؤكد
    compound_pnl += pnl_usdt
    wait_for_next_signal_side= "SELL" if STATE.get("side")=="long" else "BUY"
    STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0})
    save_state({"in_position":False,"position_qty":0,"compound_pnl":round(float(compound_pnl),6)})
    log_i(f"after-close: wait_for_next_signal_side={wait_for_next_signal_side}")

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

            df=fetch_ohlcv(limit=600)
            if len(df)<120: time.sleep(3); continue
            mtf=fetch_multi_timeframe(("5m","15m","1h"))
            council=ultra_intelligent_council(df, mtf_meta={"overall_signal":mtf["overall_signal"],"bull_count":mtf["bull_count"],"bear_count":mtf["bear_count"],"confidence":mtf["confidence"],"signals":mtf["signals"]})

            if LOG_MEDIUM_PANEL: render_medium_log(council)

            if not STATE["open"]:
                # wait-for-next-signal guard
                if wait_for_next_signal_side:
                    sig = "BUY" if council["score_buy"]>council["score_sell"] else "SELL"
                    if sig != wait_for_next_signal_side:
                        time.sleep(2); continue
                    else:
                        wait_for_next_signal_side=None

                side = smart_entry_system(council)
                if side:
                    px=price_now(); bal=balance_usdt(); qty=compute_size(bal, px)
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
    if not SELF_URL: log_w("keepalive disabled (SELF_URL not set)"); return
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
