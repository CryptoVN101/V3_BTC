"""
V1_LiveAlts_3F.py — Bot Giao dịch Tự động Multi-Alts
=======================================================
Signal  : Stoch H1 + KAMA + ADX + TSI + Scoring (V1_TAlts)
Data    : M15 (entry) + H1 resample (filter, snapshot, ffill)
Live    : Bybit Auto-Trade | Telegram | X (Twitter)
Symbols : Nhiều symbol, tối đa 2 lệnh Bybit cùng lúc
Trail   : Breakeven 1R → Trailing 2R (TRIGGER=1.0, DIST=1.9)
"""

import os
import logging
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
import asyncio
from datetime import datetime, timezone, timedelta
from telegram import Bot
from telegram.constants import ParseMode
import tweepy

# --- Symbols ---
SYMBOLS = ["ETHUSDT"]

# --- API Keys (lấy từ biến môi trường) ---
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
CHANNEL_ID      = os.getenv("CHANNEL_ID", "")
BYBIT_API_KEY   = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
X_API_KEY       = os.getenv("X_API_KEY", "")
X_API_SECRET    = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN  = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET", "")
X_BEARER_TOKEN  = os.getenv("X_BEARER_TOKEN", "")

# --- Kích hoạt tính năng (tự động kiểm tra API key có hợp lệ không) ---
ENABLE_BYBIT_TRADING = os.getenv("ENABLE_BYBIT_TRADING", "true").lower() in ["true", "1"] and bool(BYBIT_API_KEY and BYBIT_API_SECRET)
ENABLE_TELEGRAM      = os.getenv("ENABLE_TELEGRAM", "true").lower()  in ["true", "1"] and bool(TELEGRAM_TOKEN and CHANNEL_ID)
ENABLE_X             = os.getenv("ENABLE_X", "false").lower()         in ["true", "1"] and bool(X_API_KEY and X_API_SECRET)

# --- Tham số giao dịch (tối ưu từ V1_TAlts backtest) ---
LEVERAGE         = 10
MAX_OPEN_POSITIONS = 2      # Giới hạn lệnh Bybit mở cùng lúc

MIN_SCORE_LONG  = 30        # Điểm tối thiểu để vào LONG
MIN_SCORE_SHORT = 30        # Điểm tối thiểu để vào SHORT
MIN_SCORE_REV   = 90        # Điểm tối thiểu để đảo chiều (REV)

SL_ATR_MULTIPLIER  = 0.8    # SL = ATR_H1 × 0.8
TP_ATR_MULTIPLIER  = 3.3    # TP tối thiểu theo ATR
MIN_SL_THRESHOLD   = 0.011  # SL tối thiểu 1.1% giá vào lệnh
TP_MULTIPLIER      = 1.8    # TP = max(SL × 1.8, ATR × 3.3)

# --- Trailing Stop (2 giai đoạn tuần tự) ---
# Tại 1R  : Breakeven → dời SL về entry (bảo hiểm vốn)
# Tại 2R  : Trailing bật → SL bám đỉnh với khoảng cách DIST_R
USE_TRAILING_STOP  = True
TRAILING_TRIGGER_R = 1.0    # Trailing bật sau Breakeven thêm 1R (tổng 2R)
TRAILING_DIST_R    = 1.9    # Khoảng cách SL từ đỉnh = 1.9 × SL ban đầu

# --- ADX độ mạnh xu hướng ---
ADX_TREND     = 20          # Có xu hướng
ADX_STRONG    = 30          # Xu hướng mạnh
ADX_EXHAUSTED = 45          # Xu hướng quá cọn, chặn lệnh thuận

# --- Thời gian & API Clients ---
VN_TZ         = timezone(timedelta(hours=7))
FETCH_LIMIT   = 800         # Số nến M15 cần fetch (~200 nến H1 warmup)
binance_client = ccxt.binance({'options': {'defaultType': 'future'}})
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("V1_LiveAlts")

def get_vn_time():
    return datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")

# ==============================================================================
# INDICATORS (resample M15 → H1, ffill, không shift)
# ==============================================================================

def rolling_tsi_numpy(window_close):
    """Tính TSI (Pearson Correlation giữa close và bar_index) trong cửảo 16 nến."""
    n = len(window_close)
    if n < 2: return 0
    x = np.arange(n)
    y = window_close
    num = n * np.sum(x*y) - np.sum(x) * np.sum(y)
    den = np.sqrt((n*np.sum(x**2) - np.sum(x)**2) * (n*np.sum(y**2) - np.sum(y)**2))
    return num / den if den != 0 else 0

def indicators_h1(df):
    """Chỉ báo H1: Stoch, ADX+DI, MACD, RSI, ATR"""
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k"]         = stoch.iloc[:, 0]
    df["stoch_slope"]     = df["stoch_k"].diff()
    # Đếm số nến H1 đóng đi xuống/lên liên tiếp (shift(1) để không dùng nến đang hình thành)
    df["stoch_neg_count"] = (df["stoch_slope"].shift(1) < 0).rolling(4).sum()
    df["stoch_pos_count"] = (df["stoch_slope"].shift(1) > 0).rolling(4).sum()

    adx_data         = ta.adx(df["high"], df["low"], df["close"])
    df["adx"]        = adx_data["ADX_14"]
    df["adx_slope"]  = df["adx"].diff()
    df["plus_di"]    = adx_data["DMP_14"]
    df["minus_di"]   = adx_data["DMN_14"]

    macd             = ta.macd(df["close"])
    df["macd_hist"]  = macd.iloc[:, 2]
    df["rsi"]        = ta.rsi(df["close"])
    df["atr_h1"]     = ta.atr(df["high"], df["low"], df["close"], length=14)
    return df

def indicators_m15(df):
    """Chỉ báo M15: Stoch, KAMA, TSI, Donchian, ATR, MFI, VWAP, ADX"""
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=16, smooth_k=16, d=8)
    df["stoch_k_m15"]    = stoch.iloc[:, 0]
    df["stoch_d_m15"]    = stoch.iloc[:, 1]
    df["stoch_slope_m15"] = df["stoch_k_m15"].diff()
    df["kama"]           = ta.kama(df["close"], length=10, fast=2, slow=20)
    df["kama_slope"]     = df["kama"].diff()
    df["tsi"]            = df["close"].rolling(16).apply(rolling_tsi_numpy, raw=True)
    dc                   = ta.donchian(df["high"], df["low"])
    df["dc_mid"]         = dc.iloc[:, 1]
    df["atr"]            = ta.atr(df["high"], df["low"], df["close"])
    df["atr_avg"]        = df["atr"].rolling(50).mean()
    df["mfi"]            = ta.mfi(df["high"], df["low"], df["close"], df["volume"])
    df["vwap"]           = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
    df["adx_m15"]        = ta.adx(df["high"], df["low"], df["close"])["ADX_14"]
    return df


def process_data(df_m15):
    """Resample M15 → H1, tính chỉ báo H1, join vào M15 rồi ffill.
    Không shift H1 — live trading thấy snapshot H1 đang hình thành.
    """
    df_m15 = indicators_m15(df_m15)
    df_h1  = df_m15.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    df_h1 = indicators_h1(df_h1)
    H1_COLS = [
        "stoch_k", "stoch_slope", "stoch_neg_count", "stoch_pos_count",
        "adx", "adx_slope", "plus_di", "minus_di",
        "macd_hist", "rsi", "atr_h1"
    ]
    return df_m15.join(df_h1[H1_COLS], how="left").ffill().dropna()

# ==============================================================================
# SIGNAL ANALYSIS
# ==============================================================================

def get_trend(row):
    """Xác định xu hướng dựa vào +DI / -DI của ADX H1."""
    if pd.isna(row.get("plus_di")) or pd.isna(row.get("minus_di")):
        return "NEUTRAL"
    return "UP" if row["plus_di"] > row["minus_di"] else "DOWN"


def allow_countertrend(adx_state, adx_slope):
    """Cho phép lệnh ngược xu hướng khi ADX yếu hoặc đang giảm."""
    if adx_state == "WEAK":     return True
    if adx_state == "MODERATE": return adx_slope < 0
    return False  # STRONG: không cho phép ngược xu hướng


def analyze_signal(row):
    """Phân tích tín hiệu vào lệnh. Trả về (direction, score) hoặc (None, 0).
    
    Logic: Stoch H1 đảo đầu (3+ nến xuống/lên) + KAMA xác nhận → hướng lệnh.
    Bộ lọc: ADX xu hướng, ADX M15 (10-50), TSI không quá cực.
    Điểm số: 5 tiêu chí × (15–20 điểm) = tối đa 100.
    """
    direction = None
    score = 0

    if pd.isna(row.get("stoch_k")) or pd.isna(row.get("stoch_slope")):
        return None, 0

    k = row["stoch_k"]
    adx = row.get("adx", 0) or 0
    trend = get_trend(row)

    stoch_neg = row.get("stoch_neg_count", 0) or 0
    stoch_pos = row.get("stoch_pos_count", 0) or 0
    stoch_slope = row.get("stoch_slope", 0) or 0
    kama_slope = row.get("kama_slope", 0) if pd.notna(row.get("kama_slope")) else 0

    # Entry Condition (Stoch H1 + KAMA)
    if stoch_slope > 0 and stoch_neg >= 3:
        if pd.notna(row.get("kama")) and row["close"] > row["kama"] and kama_slope >= 0:
            direction = "LONG"
    elif stoch_slope < 0 and stoch_pos >= 3:
        if pd.notna(row.get("kama")) and row["close"] < row["kama"] and kama_slope <= 0:
            direction = "SHORT"

    if direction is None:
        return None, 0

    # ADX Filters
    adx_state = "STRONG" if adx >= ADX_STRONG else ("MODERATE" if adx >= ADX_TREND else "WEAK")
    is_ct = (direction == "LONG" and trend == "DOWN") or (direction == "SHORT" and trend == "UP")

    if is_ct:
        adx_slope = row.get("adx_slope", 0) or 0
        if not allow_countertrend(adx_state, adx_slope):
            return None, 0
    else:
        if adx > ADX_EXHAUSTED:
            return None, 0

    # ADX M15 filter
    adx_m15 = row.get("adx_m15", 0) or 0
    if adx_m15 > 50 or adx_m15 < 10:
        return None, 0

    # TSI Filters
    tsi = row.get("tsi", 0) or 0
    if direction == "SHORT" and tsi < -0.6: return None, 0
    if direction == "LONG" and tsi > 0.6: return None, 0

    # Scoring
    k_m15 = row.get("stoch_k_m15", 50) or 50
    d_m15 = row.get("stoch_d_m15", 50) or 50
    k_slope_m15 = row.get("stoch_slope_m15", 0) or 0

    if direction == "LONG":
        if k_m15 > d_m15 and k_slope_m15 > 0: score += 20
        if (row.get("rsi", 50) or 50) < 50: score += 20
        if (row.get("macd_hist", 0) or 0) > 0: score += 15
        if pd.notna(row.get("dc_mid")) and row["close"] < row["dc_mid"]: score += 15
        if pd.notna(row.get("mfi")) and (row.get("mfi", 50) or 50) > 30: score += 15
        if pd.notna(row.get("atr")) and pd.notna(row.get("atr_avg")) and row["atr"] > row["atr_avg"]: score += 15

    if direction == "SHORT":
        if k_m15 < d_m15 and k_slope_m15 < 0: score += 20
        if (row.get("rsi", 50) or 50) > 50: score += 20
        if (row.get("macd_hist", 0) or 0) < 0: score += 15
        if pd.notna(row.get("dc_mid")) and row["close"] > row["dc_mid"]: score += 15
        if pd.notna(row.get("mfi")) and (row.get("mfi", 50) or 50) < 70: score += 15
        if pd.notna(row.get("atr")) and pd.notna(row.get("atr_avg")) and row["atr"] > row["atr_avg"]: score += 15

    return direction, max(0, min(score, 100))

def build_trade(row, direction, current_price):
    """Tính SL/TP cho lệnh mới.
    SL = max(ATR_H1 × 0.8, price × 1.1%)
    TP = max(SL × 1.8, ATR_H1 × 3.3)
    """
    atr_h1  = row.get("atr_h1", 0) or 0
    sl_dist = max(atr_h1 * SL_ATR_MULTIPLIER, current_price * MIN_SL_THRESHOLD)
    tp_dist = max(sl_dist * TP_MULTIPLIER, atr_h1 * TP_ATR_MULTIPLIER)
    if direction == "LONG":
        return current_price, current_price - sl_dist, current_price + tp_dist, sl_dist
    else:
        return current_price, current_price + sl_dist, current_price - tp_dist, sl_dist

# ==============================================================================
# TELEGRAM / X HELPERS
# ==============================================================================

async def send_telegram(bot, text):
    if not ENABLE_TELEGRAM or not bot or not CHANNEL_ID: return
    try:
        await bot.send_message(chat_id=CHANNEL_ID, text=text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"❌ Tele Error: {e}")


def format_signal(symbol, direction, entry, sl, tp, score):
    sl_pct = abs(entry - sl) / entry * 100
    tp_pct = abs(tp - entry) / entry * 100
    rr = tp_pct / sl_pct if sl_pct else 0
    return f"""
🤖 <b>{direction} {symbol}</b>

💰 Entry: <code>{entry:,.4f}</code>
🛑 SL: <code>{sl:,.4f}</code> ({sl_pct:.2f}%)
🎯 TP: <code>{tp:,.4f}</code> ({tp_pct:.2f}%)

📈 R:R = 1:{rr:.1f}
⭐ Score: {score}/100

⏰ {get_vn_time()}
"""

def format_close(symbol, side, result, price, entry):
    pnl_pct = (price - entry) / entry * 100 if side == "LONG" else (entry - price) / entry * 100
    icon = "✅" if pnl_pct > 0 else "❌"
    return f"""
{icon} <b>{result} {symbol}</b>

📌 {side} @ <code>{entry:,.4f}</code>
🚪 Exit: <code>{price:,.4f}</code>
💰 PnL: <code>{pnl_pct:+.2f}%</code>

⏰ {get_vn_time()}
"""

def format_trailing(symbol, position, new_sl):
    return f"""
🏃 <b>TRAILING STOP ACTIVE</b> {symbol}

📌 {position}
🔒 New SL: <code>{new_sl:,.4f}</code>

⏰ {get_vn_time()}
"""

async def send_to_x(text):
    if not ENABLE_X or not X_API_KEY or not X_ACCESS_TOKEN: return
    try:
        clean = text.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", "")
        client = tweepy.Client(
            bearer_token=X_BEARER_TOKEN,
            consumer_key=X_API_KEY, consumer_secret=X_API_SECRET,
            access_token=X_ACCESS_TOKEN, access_token_secret=X_ACCESS_SECRET
        )
        await asyncio.to_thread(client.create_tweet, text=clean[:280])
        logger.info("✅ Posted to X")
    except Exception as e:
        logger.error(f"❌ X Error: {e}")

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================

class SymbolState:
    """Lưu trạng thái riêng cho từng symbol.
    Telegram và Bybit hoạt động độc lập: tín hiệu có thể khác nhau do pending.
    """
    def __init__(self, symbol):
        self.symbol = symbol

        # --- Telegram (bot tự track giá để gửi thông báo) ---
        self.tele_pos           = None   # Hướng lệnh hiện tại: 'LONG'/'SHORT'/None
        self.tele_entry         = None   # Giá vào
        self.tele_sl            = None   # Stop Loss hiện tại (có thể thay đổi theo Trailing)
        self.tele_tp            = None   # Take Profit (999999 khi Trailing bật)
        self.tele_peak_price    = None   # Giá đỉnh/đáy tốt nhất từ lúc vào lệnh
        self.tele_sl_dist       = None   # Khoảng cách SL ban đầu (= 1R)
        self.tele_sl_moved      = False  # True khi Breakeven (SL → entry) đã xảy ra
        self.tele_trailing_active = False # True khi Trailing đã bật (giá đạt 2R)

        # --- Bybit (bot chỉ sync state, Bybit tự xử lý TP/SL/Trail) ---
        self.bybit_pos          = None   # Hướng lệnh đang mở trên Bybit
        self.bybit_qty          = None   # Số lượng hợp đồng
        self.bybit_entry        = None   # Giá vào
        self.bybit_sl_dist      = None   # SL ban đầu (1R) — để tính Breakeven và Trailing
        self.bybit_sl_moved     = False  # True khi đã gửi lệnh dời SL về entry
        self.bybit_trailing_active = False # True khi đã bật Native Trailing trên Bybit

        # --- Pending (chờ xác nhận tín hiệu 1 chu kỳ nữa) ---
        self.tele_pending_dir   = None
        self.tele_pending_score = 0
        self.bybit_pending_dir  = None
        self.bybit_pending_score = 0

# Global states
symbol_states = {sym: SymbolState(sym) for sym in SYMBOLS}

def count_bybit_positions():
    """Đếm số lệnh Bybit đang mở (từ bộ nhớ bot - được sync với Bybit mỗi chu kỳ)"""
    return sum(1 for s in symbol_states.values() if s.bybit_pos is not None)

def get_volume_multiplier(symbol):
    """Hệ số nhân khối lượng (ETH x2, còn lại x1)"""
    return 2.0 if symbol == "ETHUSDT" else 1.0

def calc_bybit_qty(symbol, equity, sl_dist, current_price):
    """Tính qty Bybit theo COMPOUND (lãi kép toàn vốn)"""
    volume_multiplier = get_volume_multiplier(symbol)
    target_qty = (equity * volume_multiplier) / current_price
    max_qty = (equity * 0.95 / current_price) * LEVERAGE
    return min(target_qty, max_qty)

async def bybit_open(exchange, symbol, direction, last, current_price, bot):
    """Mở lệnh Bybit: SL + TP cứng. Trailing và Breakeven sẽ được đặt sau khi giá chạy."""
    state = symbol_states[symbol]
    entry, sl, tp, sl_dist = build_trade(last, direction, current_price)
    pos_idx = 1 if direction == "LONG" else 2
    side_str = "buy" if direction == "LONG" else "sell"

    try:
        bal = await asyncio.to_thread(exchange.fetch_balance)
        equity = float(bal["USDT"]["total"])
        target_qty = calc_bybit_qty(symbol, equity, sl_dist, current_price)

        # Đặt lệnh market với SL + TP (chưa có trailing - sẽ đặt sau khi đạt mức)
        await asyncio.to_thread(exchange.create_market_order, symbol, side_str, target_qty, params={
            "positionIdx": pos_idx,
            "stopLoss": {"triggerPrice": str(round(sl, 6)), "type": "market"},
            "takeProfit": {"triggerPrice": str(round(tp, 6)), "type": "market"},
        })

        state.bybit_pos = direction
        state.bybit_qty = target_qty
        state.bybit_entry = entry
        state.bybit_sl_dist = sl_dist      # Lưu để tính Breakeven và Trailing sau
        state.bybit_sl_moved = False
        state.bybit_trailing_active = False
        logger.info(f"✅ [{symbol}] Bybit ENTRY: {direction} @ {entry:.4f} | Qty={target_qty:.4f} | SL={sl:.4f} | TP={tp:.4f}")
        logger.info(f"   [{symbol}] Breakeven sẽ bật tại 1R (+{sl_dist:.4f}), Trailing tại 2R (+{sl_dist*2:.4f})")

    except Exception as e:
        logger.error(f"❌ [{symbol}] Bybit Open Error: {e}")

# ==============================================================================
# SCAN LOGIC
# ==============================================================================

async def scan_symbol(symbol, bot, exchange):
    """Xử lý 1 symbol mỗi chu kỳ 15m"""
    state = symbol_states[symbol]

    # 1. Fetch M15 Data from Binance
    try:
        binance_sym = symbol.replace("USDT", "/USDT")
        ohlcv = await asyncio.to_thread(binance_client.fetch_ohlcv, binance_sym, "15m", limit=FETCH_LIMIT)
        if not ohlcv: return

        df_m15 = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df_m15["time"] = pd.to_datetime(df_m15["time"], unit="ms", utc=True).dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
        df_m15.set_index("time", inplace=True)

        df = process_data(df_m15)
        if df.empty: return

        last = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = float(last["close"])

        direction, score = analyze_signal(last)

    except Exception as e:
        logger.error(f"❌ [{symbol}] Analysis Error: {e}")
        return

    # ==========================================================================
    # TELEGRAM LOGIC
    # ==========================================================================

    if state.tele_pos:
        result = None
        exit_price = 0.0

        # Cập nhật peak price
        if state.tele_pos == "LONG":
            if state.tele_peak_price is None or float(prev["high"]) > state.tele_peak_price:
                state.tele_peak_price = float(prev["high"])
        else:
            if state.tele_peak_price is None or float(prev["low"]) < state.tele_peak_price:
                state.tele_peak_price = float(prev["low"])

        # ===== TRAILING STOP TELEGRAM - 2 GIAĐOẠN TUẦN TỰ =====
        if USE_TRAILING_STOP and state.tele_sl_dist and state.tele_peak_price:
            one_r = state.tele_sl_dist

            if state.tele_pos == "LONG":
                profit_r = (state.tele_peak_price - state.tele_entry) / one_r

                # Giai đoạn 1: Breakeven tại 1R
                if not state.tele_sl_moved and profit_r >= 1.0:
                    state.tele_sl = state.tele_entry
                    state.tele_sl_moved = True
                    logger.info(f"[{symbol}] Tele Breakeven LONG: SL → {state.tele_entry:.4f}")

                # Giai đoạn 2: Trailing bật SAU breakeven tại 1R + TRAILING_TRIGGER_R = 2.0R
                if state.tele_sl_moved and profit_r >= 1.0 + TRAILING_TRIGGER_R:
                    state.tele_tp = 999999.0
                    new_sl = state.tele_peak_price - one_r * TRAILING_DIST_R
                    if new_sl > state.tele_sl:
                        state.tele_sl = new_sl
                        if not state.tele_trailing_active:
                            state.tele_trailing_active = True
                            msg = format_trailing(symbol, state.tele_pos, new_sl)
                            await send_telegram(bot, msg)
                        else:
                            logger.info(f"[{symbol}] Tele Trailing LONG: SL → {new_sl:.4f}")

            else:  # SHORT
                profit_r = (state.tele_entry - state.tele_peak_price) / one_r

                # Giai đoạn 1: Breakeven tại 1R
                if not state.tele_sl_moved and profit_r >= 1.0:
                    state.tele_sl = state.tele_entry
                    state.tele_sl_moved = True
                    logger.info(f"[{symbol}] Tele Breakeven SHORT: SL → {state.tele_entry:.4f}")

                # Giai đoạn 2: Trailing bật SAU breakeven tại 1R + TRAILING_TRIGGER_R = 2.0R
                if state.tele_sl_moved and profit_r >= 1.0 + TRAILING_TRIGGER_R:
                    state.tele_tp = 0.0001
                    new_sl = state.tele_peak_price + one_r * TRAILING_DIST_R
                    if new_sl < state.tele_sl:
                        state.tele_sl = new_sl
                        if not state.tele_trailing_active:
                            state.tele_trailing_active = True
                            msg = format_trailing(symbol, state.tele_pos, new_sl)
                            await send_telegram(bot, msg)
                        else:
                            logger.info(f"[{symbol}] Tele Trailing SHORT: SL → {new_sl:.4f}")

        # Check TP / SL / Trailing SL exit
        if state.tele_pos == "LONG":
            if not state.tele_trailing_active and float(prev["high"]) >= state.tele_tp:
                result, exit_price = "TP", state.tele_tp
            elif float(prev["close"]) <= state.tele_sl:
                result, exit_price = ("Trailing SL" if state.tele_trailing_active else "SL"), state.tele_sl
        else:
            if not state.tele_trailing_active and float(prev["low"]) <= state.tele_tp:
                result, exit_price = "TP", state.tele_tp
            elif float(prev["close"]) >= state.tele_sl:
                result, exit_price = ("Trailing SL" if state.tele_trailing_active else "SL"), state.tele_sl

        # Check REV
        if not result and direction and direction != state.tele_pos:
            vwap = float(last.get("vwap", 0) or 0)
            vwap_ok = True
            if vwap > 0:
                if state.tele_pos == "LONG" and direction == "SHORT" and current_price >= vwap: vwap_ok = False
                if state.tele_pos == "SHORT" and direction == "LONG" and current_price <= vwap: vwap_ok = False
            if score >= MIN_SCORE_REV and vwap_ok:
                result, exit_price = "REV", current_price

        if result:
            msg = format_close(symbol, state.tele_pos, result, exit_price, state.tele_entry)
            await send_telegram(bot, msg)
            await send_to_x(msg)
            logger.info(f"[{symbol}] Tele EXIT: {result} {state.tele_pos} @ {exit_price:.4f}")

            state.tele_pos = None
            state.tele_peak_price = None
            state.tele_sl_moved = False
            state.tele_trailing_active = False
            state.tele_pending_dir = None

            # REV: Open new immediately
            if result == "REV" and direction:
                entry, sl, tp, sl_dist = build_trade(last, direction, current_price)
                min_sc = MIN_SCORE_SHORT if direction == "SHORT" else MIN_SCORE_LONG
                if score >= min_sc:
                    msg = format_signal(symbol, direction, entry, sl, tp, score)
                    await send_telegram(bot, msg)
                    await send_to_x(msg)
                    state.tele_pos = direction
                    state.tele_entry = entry
                    state.tele_sl = sl
                    state.tele_tp = tp
                    state.tele_sl_dist = sl_dist
                    state.tele_peak_price = entry
                    state.tele_sl_moved = False
                    state.tele_trailing_active = False
                    logger.info(f"[{symbol}] Tele REV ENTRY: {direction} @ {entry:.4f} Score={score}")

    else:  # Không có lệnh — tìm cơ hội vào mới
        min_sc = MIN_SCORE_SHORT if direction == "SHORT" else MIN_SCORE_LONG
        if direction and score >= min_sc:
            entry, sl, tp, sl_dist = build_trade(last, direction, current_price)
            msg = format_signal(symbol, direction, entry, sl, tp, score)
            await send_telegram(bot, msg)
            await send_to_x(msg)
            state.tele_pos = direction
            state.tele_entry = entry
            state.tele_sl = sl
            state.tele_tp = tp
            state.tele_sl_dist = sl_dist
            state.tele_peak_price = entry
            state.tele_sl_moved = False
            state.tele_trailing_active = False
            logger.info(f"[{symbol}] Tele ENTRY: {direction} @ {entry:.4f} Score={score}")

    # ==========================================================================
    # BYBIT LOGIC — đơn giản hoá: Bybit tự xử lý TP/SL/Trailing, bot chỉ sync state
    # ==========================================================================
    if not ENABLE_BYBIT_TRADING or not exchange:
        return

    try:
        # 1. Sync state thực tế từ Bybit (fetch positions)
        raw_pos = await asyncio.to_thread(exchange.fetch_positions, [symbol])
        long_open  = next((p for p in raw_pos if str(p.get("info", {}).get("positionIdx", "")) == "1" and float(p.get("contracts", 0)) > 0), None)
        short_open = next((p for p in raw_pos if str(p.get("info", {}).get("positionIdx", "")) == "2" and float(p.get("contracts", 0)) > 0), None)
        actual_pos = "LONG" if long_open else ("SHORT" if short_open else None)
        active_data = long_open or short_open

        if state.bybit_pos and actual_pos is None:
            # Bybit đã đóng lệnh (TP / SL / Trailing / thủ công)
            logger.info(f"✅ [{symbol}] Bybit: lệnh {state.bybit_pos} đã được đóng")
            state.bybit_pos = None
            state.bybit_qty = None
            state.bybit_entry = None
            state.bybit_sl_dist = None
            state.bybit_sl_moved = False
            state.bybit_trailing_active = False

        elif state.bybit_pos is None and actual_pos:
            # Bot restart / lệnh mở trước đó chưa vào state → sync lại
            state.bybit_pos = actual_pos
            state.bybit_qty = float(active_data["contracts"])
            state.bybit_entry = float(active_data["entryPrice"])
            # sl_dist không thể khôi phục sau restart → tính lại xấp xỉ
            if not state.bybit_sl_dist and state.bybit_entry:
                _atr = float(last.get("atr_h1", 0) or 0)
                state.bybit_sl_dist = max(_atr * SL_ATR_MULTIPLIER, state.bybit_entry * MIN_SL_THRESHOLD)
        
        # ===== BYBIT TRAILING STOP 2 GIAĐOẠN - KIỂM TRA MỔI CHU KỲ =====
        if state.bybit_pos and state.bybit_sl_dist and actual_pos and active_data:
            pos_idx = 1 if state.bybit_pos == "LONG" else 2
            one_r = state.bybit_sl_dist
            entry_price = state.bybit_entry or float(active_data["entryPrice"])

            # Lấy giá hiện tại từ thị trường
            mark_price = float(active_data.get("markPrice") or active_data.get("info", {}).get("markPrice") or current_price)

            if state.bybit_pos == "LONG":
                profit_r = (mark_price - entry_price) / one_r

                # Giai đoạn 1: Breakeven khi lời được 1R → dời SL về Entry
                if not state.bybit_sl_moved and profit_r >= 1.0:
                    be_price = round(entry_price, 6)
                    try:
                        await asyncio.to_thread(exchange.set_trading_stop, symbol, {
                            "stopLoss": str(be_price),
                            "positionIdx": pos_idx,
                        })
                        state.bybit_sl_moved = True
                        logger.info(f"🛡️ [{symbol}] Bybit Breakeven: SL → {be_price:.4f} (profit={profit_r:.2f}R)")
                    except Exception as e:
                        logger.warning(f"⚠️ [{symbol}] Bybit Breakeven failed: {e}")

                # Giai đoạn 2: Trailing bật sau breakeven, tính từ tổng 1R + TRAILING_TRIGGER_R
                if state.bybit_sl_moved and not state.bybit_trailing_active and profit_r >= 1.0 + TRAILING_TRIGGER_R:
                    trailing_dist = round(one_r * TRAILING_DIST_R, 6)
                    try:
                        await asyncio.to_thread(exchange.set_trading_stop, symbol, {
                            "trailingStop": str(trailing_dist),
                            "positionIdx": pos_idx,
                        })
                        state.bybit_trailing_active = True
                        logger.info(f"🔄 [{symbol}] Bybit Trailing bật: dist={trailing_dist:.4f} (profit={profit_r:.2f}R)")
                    except Exception as e:
                        logger.warning(f"⚠️ [{symbol}] Bybit Trailing failed: {e}")

            else:  # SHORT
                profit_r = (entry_price - mark_price) / one_r

                # Giai đoạn 1: Breakeven
                if not state.bybit_sl_moved and profit_r >= 1.0:
                    be_price = round(entry_price, 6)
                    try:
                        await asyncio.to_thread(exchange.set_trading_stop, symbol, {
                            "stopLoss": str(be_price),
                            "positionIdx": pos_idx,
                        })
                        state.bybit_sl_moved = True
                        logger.info(f"🛡️ [{symbol}] Bybit Breakeven: SL → {be_price:.4f} (profit={profit_r:.2f}R)")
                    except Exception as e:
                        logger.warning(f"⚠️ [{symbol}] Bybit Breakeven failed: {e}")

                # Giai đoạn 2: Trailing bật
                if state.bybit_sl_moved and not state.bybit_trailing_active and profit_r >= 1.0 + TRAILING_TRIGGER_R:
                    trailing_dist = round(one_r * TRAILING_DIST_R, 6)
                    try:
                        await asyncio.to_thread(exchange.set_trading_stop, symbol, {
                            "trailingStop": str(trailing_dist),
                            "positionIdx": pos_idx,
                        })
                        state.bybit_trailing_active = True
                        logger.info(f"🔄 [{symbol}] Bybit Trailing bật: dist={trailing_dist:.4f} (profit={profit_r:.2f}R)")
                    except Exception as e:
                        logger.warning(f"⚠️ [{symbol}] Bybit Trailing failed: {e}")
        # Log sau khi xử lý trailing (chỉ khi sync từ Bybit)
        if actual_pos and state.bybit_pos:
            logger.info(f"[{symbol}] Bybit synced: {actual_pos} | BE={state.bybit_sl_moved} | Trail={state.bybit_trailing_active}")

        # 2. Check REV (nếu đang có lệnh)
        if state.bybit_pos and direction and direction != state.bybit_pos:
            vwap = float(last.get("vwap", 0) or 0)
            vwap_ok = True
            if vwap > 0:
                if state.bybit_pos == "LONG"  and direction == "SHORT" and current_price >= vwap: vwap_ok = False
                if state.bybit_pos == "SHORT" and direction == "LONG"  and current_price <= vwap: vwap_ok = False
            if score >= MIN_SCORE_REV and vwap_ok:
                close_side = "sell" if state.bybit_pos == "LONG" else "buy"
                pos_idx    = 1 if state.bybit_pos == "LONG" else 2
                try:
                    await asyncio.to_thread(
                        exchange.create_market_order, symbol, close_side, state.bybit_qty,
                        params={"positionIdx": pos_idx, "reduceOnly": True}
                    )
                    logger.info(f"🔄 [{symbol}] Bybit REV: đóng {state.bybit_pos}")
                except Exception as e:
                    logger.error(f"❌ [{symbol}] Bybit REV close error: {e}")

                state.bybit_pos = None
                state.bybit_qty = None
                state.bybit_entry = None
                state.bybit_sl_dist = None       # BUG FIX: reset đủ khi REV
                state.bybit_sl_moved = False
                state.bybit_trailing_active = False
                state.bybit_pending_dir = None

                # Mở lệnh REV mới
                min_sc = MIN_SCORE_SHORT if direction == "SHORT" else MIN_SCORE_LONG
                if score >= min_sc and count_bybit_positions() < MAX_OPEN_POSITIONS:
                    await bybit_open(exchange, symbol, direction, last, current_price, bot)
                else:
                    logger.info(f"⚠️ [{symbol}] Bybit REV bỏ qua entry (score/limit)")

        # 3. Vào lệnh mới — vào ngay khi đủ điều kiện
        elif not state.bybit_pos:
            min_sc = MIN_SCORE_SHORT if direction == "SHORT" else MIN_SCORE_LONG
            if direction and score >= min_sc:
                if count_bybit_positions() < MAX_OPEN_POSITIONS:
                    await bybit_open(exchange, symbol, direction, last, current_price, bot)
                else:
                    logger.info(f"⚠️ [{symbol}] Bybit bỏ qua: đã đủ {MAX_OPEN_POSITIONS} lệnh")

    except Exception as e:
        logger.error(f"❌ [{symbol}] Bybit Error: {e}")



async def run_scan(bot, exchange):
    """Scan tất cả symbols đồng thời"""
    tasks = [scan_symbol(sym, bot, exchange) for sym in SYMBOLS]
    await asyncio.gather(*tasks)

# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    bot = Bot(token=TELEGRAM_TOKEN) if ENABLE_TELEGRAM else None
    exchange = None
    if ENABLE_BYBIT_TRADING and BYBIT_API_KEY:
        exchange = ccxt.bybit({
            "apiKey": BYBIT_API_KEY,
            "secret": BYBIT_API_SECRET,
            "options": {"defaultType": "linear"}
        })

    features = []
    if ENABLE_BYBIT_TRADING: features.append("📊 Bybit Auto-Trade")
    if ENABLE_TELEGRAM: features.append("📱 Telegram")
    if ENABLE_X: features.append("🐦 X (Twitter)")
    features_str = "\n".join([f"   • {f}" for f in features]) if features else "   • None"

    msg = f"""
🚀 <b>V1_LiveAlts KHỞI ĐỘNG</b>

📋 Symbols: {', '.join(SYMBOLS)}
⏱ Timeframe: M15 + H1
🔧 Features:
{features_str}
⚙️ Score: L{MIN_SCORE_LONG}/S{MIN_SCORE_SHORT}/R{MIN_SCORE_REV}
✅ Trailing: {TRAILING_TRIGGER_R}R → {TRAILING_DIST_R}R Dist

⏰ {get_vn_time()}
"""
    await send_telegram(bot, msg)
    await send_to_x(msg)
    logger.info(f"🚀 V1_LiveAlts started | Symbols: {SYMBOLS}")

    # Set Hedge Mode & Leverage on Bybit
    if exchange:
        for sym in SYMBOLS:
            for attempt in range(3):
                try:
                    await asyncio.to_thread(exchange.set_position_mode, True, sym)
                    await asyncio.to_thread(exchange.set_leverage, LEVERAGE, sym)
                    logger.info(f"✅ [{sym}] Hedge Mode & Leverage={LEVERAGE}x Set")
                    break
                except Exception as e:
                    if "110025" in str(e):
                        logger.info(f"✅ [{sym}] Hedge Mode Already Set")
                        try: await asyncio.to_thread(exchange.set_leverage, LEVERAGE, sym)
                        except: pass
                        break
                    logger.warning(f"⚠️ [{sym}] Retry {attempt+1}: {e}")
                    await asyncio.sleep(1)

    # Main Loop
    while True:
        try:
            now = datetime.now()
            wait = (15 - (now.minute % 15)) * 60 - now.second + 3
            if wait < 10: wait = 1
            logger.info(f"⏳ Waiting {wait}s for next candle...")
            await asyncio.sleep(wait)

            await run_scan(bot, exchange)

            # Log status
            if ENABLE_BYBIT_TRADING and exchange:
                try:
                    bal = await asyncio.to_thread(exchange.fetch_balance)
                    equity = float(bal["USDT"]["total"])
                    pos_summaries = [
                        f"{sym}: {symbol_states[sym].bybit_pos} ({symbol_states[sym].bybit_qty:.4f})"
                        for sym in SYMBOLS if symbol_states[sym].bybit_pos
                    ]
                    pos_info = ", ".join(pos_summaries) if pos_summaries else "None"
                    logger.info(f"💰 Balance: {equity:.2f} USDT | Pos: {pos_info}")
                except: pass

        except Exception as e:
            logger.error(f"❌ Main Loop Error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
