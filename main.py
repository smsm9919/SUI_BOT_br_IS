# =================== EXECUTION SWITCHES ===================
LIVE_TRADING = True
EXECUTE_TRADES = True
ONE_ACTIVE_POS = True
COOLDOWN_SEC = 45

# =================== COUNCIL DECISIONS ===================
SCALP_MIN_VOTES = 2
SCALP_MIN_SCORE = 1.5
TREND_MIN_VOTES = 5
TREND_MIN_SCORE = 2.5
ADX_TREND_GATE = 20
ADX_SCALP_GATE = 14
ALLOW_GOLDEN_ZONE_OVERRIDE = True

# =================== GLOBAL EXECUTION STATE ===================
_last_entry_ts = 0
_active_side = None

# =================== EXECUTION FUNCTIONS ===================
def can_execute_now():
    from time import time
    return (time() - _last_entry_ts) >= COOLDOWN_SEC

def calc_qty(balance, price):
    base = balance * RISK_ALLOC * LEVERAGE
    qty = max(base / max(price, 1e-9), 10)
    return float(qty)

def open_market(side, qty, price):
    if not MODE_LIVE:
        print(colored(f"[PAPER] MARKET {side} {qty} @ {price}", "cyan"))
        return {"id": "paper_order"}
    
    order = ex.create_order(SYMBOL, 'market', side.lower(), qty, None, _params_open(side))
    return order

def log_banner(message):
    print(colored(f"\n{'='*60}", "yellow"))
    print(colored(f" {message}", "yellow"))
    print(colored(f"{'='*60}\n", "yellow"))

def log_g(message):
    print(colored(f"✅ {message}", "green"))

def log_r(message):
    print(colored(f"❌ {message}", "red"))

def classify_and_decide(df, council, signals, golden, ind):
    adx = ind.get('adx', 0)
    rf_side = 'BUY' if signals.get('rf_long') else 'SELL' if signals.get('rf_short') else None
    rsi_cross = signals.get('rsi_ma_cross')
    
    council_buy = council.get('buy', 0)
    council_sell = council.get('sell', 0)
    council_score = council.get('score', 0)
    
    # 1) TREND
    if adx >= ADX_TREND_GATE and council_score >= TREND_MIN_SCORE and max(council_buy, council_sell) >= TREND_MIN_VOTES and rf_side:
        side = 'BUY' if council_buy > council_sell else 'SELL'
        mode = 'TREND'
        return True, side, mode, "trend: council+adx+rf"

    # 2) GOLDEN ZONE (أولوية عالية)
    if ALLOW_GOLDEN_ZONE_OVERRIDE and golden.get('ok') and golden.get('score',0) >= 6.0 and adx >= 18:
        side = 'BUY' if golden.get('type') == 'bottom' else 'SELL'
        mode = 'TREND'
        return True, side, mode, f"golden_{golden.get('type')} score={golden.get('score',0):.1f}"

    # 3) SCALP
    if adx >= ADX_SCALP_GATE and rf_side and rsi_cross:
        if rsi_cross == 'bull' and rf_side == 'BUY' and council_buy >= SCALP_MIN_VOTES and council_score >= SCALP_MIN_SCORE:
            return True, 'BUY', 'SCALP', "scalp: rsi_cross+rf+votes"
        if rsi_cross == 'bear' and rf_side == 'SELL' and council_sell >= SCALP_MIN_VOTES and council_score >= SCALP_MIN_SCORE:
            return True, 'SELL', 'SCALP', "scalp: rsi_cross+rf+votes"

    return False, None, None, "sit_out"

def maybe_execute(df, council, signals, golden, ind, balance):
    global _last_entry_ts, _active_side
    
    ok, side, mode, why = classify_and_decide(df, council, signals, golden, ind)
    if not ok or not EXECUTE_TRADES or not LIVE_TRADING:
        return False

    if ONE_ACTIVE_POS and _active_side:
        return False

    if not can_execute_now():
        return False

    price = float(df['close'].iloc[-1])
    qty = calc_qty(balance, price)
    
    try:
        ord = open_market(side, qty, price)
        _last_entry_ts = time.time()
        _active_side = side
        
        log_banner("🚀 ENTRY")
        log_g(f"ENTRY | {mode} | {side} @ {price:.6f} | qty={qty:.2f} | why={why}")
        log_g(f"🧠 Council BUY={council.get('buy',0)} SELL={council.get('sell',0)} score={council.get('score',0):.2f} | RSI_cross={signals.get('rsi_ma_cross')} | RF={signals.get('rf_side')} | ADX={ind.get('adx',0):.1f}")
        
        return True
    except Exception as e:
        log_r(f"EXECUTION ERROR: {e}")
        return False

# =================== POSITION MANAGEMENT ===================
def manage_position(state, df, ind, council, mode):
    if not state["open"]:
        return
        
    price = float(df['close'].iloc[-1])
    entry = state["entry"]
    side = state["side"]
    
    # حساب الربح غير المحقق
    if side == "long":
        upl_pct = (price - entry) / entry * 100
    else:
        upl_pct = (entry - price) / entry * 100
        
    state["upl_pct"] = upl_pct
    
    # SCALP Management
    if mode == 'SCALP':
        if upl_pct >= 0.35:
            close_all("SCALP_TP", state)
        elif state.get("bars", 0) >= 8:  # ~20 دقيقة في إطار 15m
            close_all("SCALP_TIMEOUT", state)
        elif ind.get('evx', 0) < -1.0:
            close_all("SCALP_DEFENSE", state)

    # TREND Management  
    else:
        if upl_pct >= 0.30 and not state["trade_management"].get("break_even_moved", False):
            move_to_breakeven(state)
        apply_atr_trail(state, ind, mult=1.6)
        
        rf_opposite = False
        if side == "long" and ind.get('rf_short'):
            rf_opposite = True
        elif side == "short" and ind.get('rf_long'):
            rf_opposite = True
            
        if rf_opposite and max(council.get('buy',0), council.get('sell',0)) >= 5:
            close_all("TREND_STRICT_CLOSE", state)

def move_to_breakeven(state):
    state["trade_management"]["current_stop"] = state["entry"]
    state["trade_management"]["break_even_moved"] = True
    log_g(f"BREAKEVEN MOVED → {state['entry']}")

def apply_atr_trail(state, ind, mult=1.6):
    atr = ind.get('atr', 0)
    price = state.get('current_price', state['entry'])
    side = state["side"]
    
    if side == "long":
        new_trail = price - (atr * mult)
        current_stop = state["trade_management"].get("current_stop", 0)
        if new_trail > current_stop:
            state["trade_management"]["current_stop"] = new_trail
            state["trade_management"]["trailing_active"] = True
    else:
        new_trail = price + (atr * mult)
        current_stop = state["trade_management"].get("current_stop", float('inf'))
        if new_trail < current_stop:
            state["trade_management"]["current_stop"] = new_trail
            state["trade_management"]["trailing_active"] = True

def close_all(reason, state):
    global _active_side
    
    side_to_close = "sell" if state["side"] == "long" else "buy"
    qty_to_close = state["qty"]
    
    try:
        if MODE_LIVE:
            ex.create_order(SYMBOL, 'market', side_to_close, qty_to_close, None, _params_close())
        else:
            print(colored(f"[PAPER] MARKET CLOSE {side_to_close} {qty_to_close}", "cyan"))
            
        log_banner("✅ EXIT")
        log_g(f"EXIT | reason={reason} | side={state['side']} | pnl={state.get('upl_pct',0):.2f}% | total_pnl={compound_pnl:.2f}")
        
        _active_side = None
        _reset_after_close(reason, state["side"])
        
    except Exception as e:
        log_r(f"CLOSE ERROR: {e}")

# =================== RESUME POSITION ===================
def resume_live_position():
    try:
        # محاولة قراءة المركز من البورصة
        poss = ex.fetch_positions([SYMBOL])
        for pos in poss:
            if pos['symbol'] == SYMBOL and pos['contracts'] and float(pos['contracts']) > 0:
                side = 'long' if float(pos['contracts']) > 0 else 'short'
                qty = abs(float(pos['contracts']))
                entry = float(pos['entryPrice'])
                
                STATE.update({
                    "open": True,
                    "side": side,
                    "qty": qty,
                    "entry": entry,
                    "trading_mode": "TREND"  # افتراضي
                })
                _active_side = 'BUY' if side == 'long' else 'SELL'
                
                log_banner("🔄 POSITION RESUMED")
                log_g(f"Resumed {side} position: {qty} @ {entry}")
                return True
    except Exception as e:
        print(colored(f"⚠️ Cannot fetch positions: {e}", "yellow"))
    
    return False

# =================== ENHANCED SNAPSHOT ===================
def enhanced_snapshot(df, ind, council, signals, balance):
    price = float(df['close'].iloc[-1])
    rf_val = signals.get('rf_filter', price)
    spread_bps = orderbook_spread_bps() or 0
    rsi = ind.get('rsi', 50)
    adx = ind.get('adx', 0)
    di_p = ind.get('plus_di', 0)
    di_m = ind.get('minus_di', 0)
    atr = ind.get('atr', 0)
    rsi_cross = signals.get('rsi_ma_cross', 'none')
    rf_side = 'BUY' if signals.get('rf_long') else 'SELL' if signals.get('rf_short') else 'NONE'
    
    print(colored(f"\n{IC['hdr']}  ELITE COUNCIL • {SYMBOL} {INTERVAL} • {datetime.utcnow().strftime('%H:%M:%S UTC')}", "cyan"))
    print(colored(f"{IC['mk']}  MARKET", "white"))
    print(f"  $ Price={_num(price,6)} | {IC['rf']} RF={_num(rf_val,6)} | spread={_num(spread_bps,2)}bps")
    print(f"  {IC['ind']} RSI={_num(rsi,1)} cross={rsi_cross} | RF={rf_side} | ADX={_num(adx,1)} +DI={_num(di_p,1)} -DI={_num(di_m,1)}")
    print(f"  {IC['vote']} Council: BUY={council.get('buy',0)} SELL={council.get('sell',0)} score={council.get('score',0):.1f}")
    
    mode = STATE.get("trading_mode", "SIT_OUT")
    if STATE["open"]:
        mode = f"{STATE['trading_mode']} ({STATE['side']})"
    
    print(f"  {IC['strat']} Plan: {mode} | closes_in={time_to_candle_close(df)}s")
    print(f"  {IC['bal']} Balance={_num(balance,2)} | Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x | {IC['pnl']} TotalPnL={_num(compound_pnl,2)}")

# =================== INTEGRATION IN MAIN LOOP ===================
# في دالة elite_trade_loop، بعد حساب المؤشرات والإشارات، أضف:

def elite_trade_loop_enhanced():
    global _last_entry_ts, _active_side, compound_pnl
    
    # استعادة المركز النشط عند البدء
    if not STATE["open"]:
        resume_live_position()
    
    while True:
        try:
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            reconcile_state()

            info, ind, zones, candidates, trend, plan, elite_candidate = evaluate_all_elite(df)
            
            # بناء إشارات للمجلس التنفيذي
            council = {
                "buy": STATE.get("votes_b", 0),
                "sell": STATE.get("votes_s", 0), 
                "score": max(STATE.get("score_b", 0), STATE.get("score_s", 0))
            }
            
            signals = {
                "rf_long": info.get("long"),
                "rf_short": info.get("short"), 
                "rf_filter": info.get("filter"),
                "rsi_ma_cross": STATE.get("rsi_ma_signal", {}).get("cross"),
                "evx": ind.get("macd_hist", 0)  # مبسط
            }
            
            golden = {"ok": False}  # يمكن استبدال بالدالة الفعلية
            
            # طباعة السنابشوت المحسن
            enhanced_snapshot(df, ind, council, signals, bal)
            
            # التنفيذ الفعلي
            if not STATE["open"]:
                maybe_execute(df, council, signals, golden, ind, bal)
            
            # إدارة الصفقة النشطة
            if STATE["open"]:
                manage_position(STATE, df, ind, council, STATE.get("trading_mode", "SCALP"))
            
            time.sleep(2 if time_to_candle_close(df) <= 30 else 5)
            
        except Exception as e:
            print(colored(f"❌ Enhanced loop error: {e}", "red"))
            time.sleep(5)

# =================== REPLACE MAIN LOOP ===================
# استبدال الدورة الرئيسية بالنسخة المحسنة
def start_elite_system():
    print(colored("\n" + "🌟" * 50, "yellow"))
    print(colored("🚀 STARTING ELITE SUI COUNCIL PRO - EXECUTION SYSTEM", "yellow"))
    print(colored("🌟" * 50 + "\n", "yellow"))
    
    # استعادة المركز النشط
    resume_live_position()
    
    t1 = threading.Thread(target=elite_trade_loop_enhanced, name="elite_trade_loop", daemon=True)
    t1.start()
    
    t2 = threading.Thread(target=keepalive_loop, name="keepalive", daemon=True)
    t2.start()
    
    return t1, t2
