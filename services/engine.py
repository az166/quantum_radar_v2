import asyncio
import numpy as np
import os
import json
import time
from datetime import datetime
from utils.indicators import (
    calculate_ma, calculate_std_dev, calculate_obv_trend,
    detect_bullish_divergence, calculate_macd_efficient, calculate_pearson_correlation,
    calculate_technical_envelope_single_pass  # Memanggil fungsi optimasi Single-Pass
)
from services.binance_service import fetch_klines_safely_async, fetch_order_book_imbalance
from services.telegram_service import send_telegram_in_worker_thread

# ==============================================================================
# DATA CACHE STORE (MULTI-TIER CACHING)
# ==============================================================================
# Menyimpan data kline di RAM untuk menghindari pembatasan laju panggilan (rate limit) Binance
_kline_cache = {}

async def fetch_klines_cached(client, symbol, interval, limit, ttl_seconds):
    """
    Mengambil data kline dengan sistem penyimpanan sementara (cache).
    Mencegah panggilan API berulang untuk data historis yang jarang berubah.
    """
    key = f"{symbol}_{interval}_{limit}"
    now = time.time()

    if key in _kline_cache:
        cached_data, expire_time = _kline_cache[key]
        if now < expire_time:
            return cached_data

    data = await fetch_klines_safely_async(client, symbol, interval, limit)
    if data:
        _kline_cache[key] = (data, now + ttl_seconds)
    return data


# ==============================================================================
# 1. PERFORMANCE LOGGER & BACKTESTING ENGINE
# ==============================================================================
class TradingPerformanceLogger:
    def __init__(self, log_filepath="logs/signal_performance.json"):
        self.log_filepath = log_filepath
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        if not os.path.exists(self.log_filepath):
            with open(self.log_filepath, 'w') as f:
                json.dump([], f)

    def _write_entry_to_file(self, log_entry):
        try:
            with open(self.log_filepath, 'r+') as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error logging entry signal: {e}")

    async def log_entry_signal_async(self, symbol, entry_price, score, action, z_score, btc_risk_status, tp_level, cl_level):
        log_entry = {
            "signal_id": f"{symbol}_{int(datetime.now().timestamp())}",
            "timestamp_entry": datetime.now().isoformat(),
            "symbol": symbol,
            "entry_price": entry_price,
            "confidence_score": score,
            "engine_action": action,
            "volume_z_score": z_score,
            "btc_risk": btc_risk_status,
            "target_tp": tp_level,
            "target_cl": cl_level,
            "status": "OPEN",
            "exit_price": None,
            "pnl_pct": 0.0,
            "timestamp_exit": None
        }
        await asyncio.to_thread(self._write_entry_to_file, log_entry)
        return log_entry["signal_id"]

    def _write_close_to_file(self, symbol, exit_price, exit_time):
        try:
            with open(self.log_filepath, 'r+') as f:
                data = json.load(f)
                updated = False
                for entry in data:
                    if entry["symbol"] == symbol and entry["status"] == "OPEN":
                        entry["status"] = "CLOSED"
                        entry["exit_price"] = exit_price
                        entry["timestamp_exit"] = exit_time
                        pnl = ((exit_price - entry["entry_price"]) / entry["entry_price"]) * 100
                        entry["pnl_pct"] = round(pnl, 2)
                        updated = True
                        break
                if updated:
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error closing logged signal for {symbol}: {e}")

    async def close_logged_signal_async(self, symbol, exit_price, current_time=None):
        exit_time = current_time if current_time else datetime.now().isoformat()
        await asyncio.to_thread(self._write_close_to_file, symbol, exit_price, exit_time)

# Inisialisasi Logger Global
perf_logger = TradingPerformanceLogger()


# ==============================================================================
# 2. QUANTITATIVE HELPER FUNCTIONS (VEKTORISASI NUMPY & OPTIMASI)
# ==============================================================================
def detect_fair_value_gap(klines_1h):
    if len(klines_1h) < 3:
        return False, 0.0
    high_1 = float(klines_1h[-3][2])
    low_3 = float(klines_1h[-1][3])
    if low_3 > high_1:
        fvg_midpoint = (low_3 + high_1) / 2
        return True, fvg_midpoint
    return False, 0.0

def calculate_volume_metrics(klines_1h, window=20):
    if len(klines_1h) < window + 1:
        return 0.0, 50.0, 1.0
    historical_volumes = [float(k[7]) for k in klines_1h[-(window+1):-1]]
    current_volume = float(klines_1h[-1][7])

    mean_vol = np.mean(historical_volumes)
    std_vol = np.std(historical_volumes)

    z_score = (current_volume - mean_vol) / std_vol if std_vol > 0.00001 else 0.0
    z_score = max(-3.0, min(6.0, z_score))

    all_vols = historical_volumes + [current_volume]
    percentile = (np.sum(np.array(all_vols) <= current_volume) / len(all_vols)) * 100

    vol_spike_ratio = current_volume / mean_vol if mean_vol > 0 else 1.0

    return round(z_score, 2), round(percentile, 1), round(vol_spike_ratio, 2)

def analyze_market_structure(klines_1h, window=5):
    """
    Menganalisis struktur pasar menggunakan vektorisasi NumPy.
    Mengeliminasi perulangan lambat Python untuk deteksi swing high dan swing low.
    """
    n = len(klines_1h)
    if n < 30:
        return "CONSOLIDATION", 0.0, 0.0

    highs = np.array([float(k[2]) for k in klines_1h])
    lows = np.array([float(k[3]) for k in klines_1h])
    closes = np.array([float(k[4]) for k in klines_1h])

    # Deteksi puncak (Swing High) dan lembah (Swing Low) secara simultan menggunakan pergeseran array
    is_swing_high = np.ones(n, dtype=bool)
    is_swing_low = np.ones(n, dtype=bool)

    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        shifted_high = np.roll(highs, -offset)
        shifted_low = np.roll(lows, -offset)
        is_swing_high &= (highs >= shifted_high)
        is_swing_low &= (lows <= shifted_low)

    # Singkirkan elemen ujung yang terkena efek perputaran indeks dari np.roll
    valid_range = (np.arange(n) >= window) & (np.arange(n) < n - window)
    swing_high_indices = np.where(is_swing_high & valid_range)[0]
    swing_low_indices = np.where(is_swing_low & valid_range)[0]

    if len(swing_high_indices) < 2 or len(swing_low_indices) < 2:
        return "CONSOLIDATION", float(np.max(highs[-10:])), float(np.min(lows[-10:]))

    # Ambil nilai puncak dan lembah terakhir serta sebelumnya
    last_sh = float(highs[swing_high_indices[-1]])
    prev_sh = float(highs[swing_high_indices[-2]])
    last_sl = float(lows[swing_low_indices[-1]])
    prev_sl = float(lows[swing_low_indices[-2]])

    if last_sh > prev_sh and last_sl > prev_sl:
        structure = "BULLISH_STRUCTURE"
    elif last_sh < prev_sh and last_sl < prev_sl:
        structure = "BEARISH_STRUCTURE"
    else:
        structure = "CONSOLIDATION"

    current_close = closes[-1]
    if structure == "BEARISH_STRUCTURE" and current_close > last_sh:
        structure = "MSS_BULLISH_BREAKOUT"

    return structure, last_sh, last_sl

def verify_breakout_status(live_price, last_close, open_price, high_price, low_price, local_swing_high, z_score):
    if local_swing_high == 0:
        return "NO_BREAKOUT"

    candle_body = abs(live_price - open_price)
    min_range_guard = live_price * 0.0001 if live_price > 0 else 0.00000001
    candle_range = max(min_range_guard, high_price - low_price)
    body_to_range_ratio = candle_body / candle_range

    is_penetrating = live_price > local_swing_high
    is_closed_above = last_close > local_swing_high

    if is_penetrating:
        if not is_closed_above or body_to_range_ratio < 0.45 or z_score < 1.5:
            return "PENDING_BREAKOUT"
        if is_closed_above and body_to_range_ratio >= 0.45 and z_score >= 1.5:
            return "CONFIRMED_BREAKOUT"

    return "NO_BREAKOUT"

def calculate_atr_and_spread(klines_1d, klines_1h):
    true_ranges = []
    for i in range(1, len(klines_1d)):
        high = float(klines_1d[i][2])
        low = float(klines_1d[i][3])
        prev_close = float(klines_1d[i-1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0.0
    current_spread = float(klines_1h[-1][2]) - float(klines_1h[-1][3])
    recent_spreads = [float(k[2]) - float(k[3]) for k in klines_1h[-6:-1]]

    avg_recent_spread = sum(recent_spreads) / len(recent_spreads) if recent_spreads else 1.0
    if avg_recent_spread <= 0: 
        avg_recent_spread = 0.00001
    return atr, current_spread / avg_recent_spread

def calculate_btc_risk_level(btc_status_dict, btc_returns_24h):
    is_safe = btc_status_dict.get("is_safe", True)
    reason = str(btc_status_dict.get("reason", "")).upper()
    avg_return = np.mean(btc_returns_24h) if btc_returns_24h else 0.0

    if not is_safe and ("CRASH" in reason or "CAPITULATION" in reason or avg_return < -0.04):
        return {"level": 4, "status": "EXTREME RISK", "allocation_pct": 0.0, "allowed_trade": False}
    if not is_safe or "BREAKDOWN" in reason or avg_return < -0.02:
        return {"level": 3, "status": "HIGH RISK", "allocation_pct": 25.0, "allowed_trade": True}
    if "SQUEEZE" in reason or "CONSOLIDATION" in reason or -0.01 <= avg_return < 0.01:
        return {"level": 2, "status": "MEDIUM RISK", "allocation_pct": 50.0, "allowed_trade": True}
    return {"level": 1, "status": "LOW RISK", "allocation_pct": 100.0, "allowed_trade": True}

def calculate_confidence_score(market_struct, breakout_status, z_score, percentile, ob_ratio, ma_bullish, btc_risk):
    if btc_risk["level"] == 4:
        return 0, "WAIT & SEE"

    score = 0
    if ma_bullish: 
        score += 30

    if z_score >= 2.0 and percentile >= 85:
        score += 25
    elif z_score >= 1.0 and percentile >= 70:
        score += 15
    elif z_score >= 0.0:
        score += 5

    if ob_ratio >= 1.5:
        score += 20
    elif ob_ratio >= 1.1:
        score += 10

    if market_struct == "MSS_BULLISH_BREAKOUT" or breakout_status == "CONFIRMED_BREAKOUT":
        score += 15
    elif breakout_status == "PENDING_BREAKOUT":
        score += 7

    if btc_risk["level"] == 1:
        score += 10
    elif btc_risk["level"] == 2:
        score += 5
    elif btc_risk["level"] == 3:
        score -= 15

    score = max(0, min(100, score))

    if score >= 75:
        action = "STRONG BUY"
    elif score >= 55:
        action = "SPECULATIVE BUY / EARLY TREND"
    elif score >= 40:
        action = "HOLDING / MONITOR"
    else:
        action = "WAIT & SEE"

    return score, action

def hitung_matriks_atr_dinamis(live_price, entry_price, atr, vol_spike_ratio, whale_dominance, btc_risk_level, highest_peak=0.0):
    try:
        live_price = float(live_price or 0.0)
        entry_price = float(entry_price or 0.0)
        atr = float(atr or 0.0)
        vol_spike_ratio = float(vol_spike_ratio or 1.0)
        whale_dominance = float(whale_dominance or 50.0)
        btc_risk_level = int(btc_risk_level or 1)
        highest_peak = float(highest_peak or 0.0)

        if btc_risk_level == 4:
            base_multiplier = 1.2
        elif btc_risk_level == 3:
            base_multiplier = 1.5
        elif btc_risk_level == 2:
            base_multiplier = 2.0
        else:
            base_multiplier = 2.5

        if vol_spike_ratio > 2.0:
            vol_modifier = 1.3
        elif vol_spike_ratio > 1.5:
            vol_modifier = 1.15
        else:
            vol_modifier = 1.0

        whale_modifier = 0.9 if whale_dominance > 65.0 else 1.0
        final_multiplier = base_multiplier * vol_modifier * whale_modifier
        atr_distance = atr * final_multiplier
        reference_price = max(entry_price, highest_peak) if highest_peak > 0.0 else entry_price

        dynamic_tp = reference_price + (atr_distance * 1.5)
        dynamic_cl = reference_price - atr_distance

        if dynamic_cl < 0:
            dynamic_cl = entry_price * 0.95 

        return round(dynamic_tp, 6), round(dynamic_cl, 6)

    except Exception as e:
        print(f"[ENGINE RECOVERY CRITICAL] Fallback ATR terpicu akibat: {e}")
        fallback_tp = entry_price * 1.05
        fallback_cl = entry_price * 0.97
        return round(fallback_tp, 6), round(fallback_cl, 6)


# ==============================================================================
# 3. MAIN ASYNC PIPELINE ENGINE
# ==============================================================================
async def process_single_coin_pipeline(client, symbol, m_data, user_portfolio, semaphore, state_manager, device_id="default_guest_device"):
    async with semaphore:
        # Pipa Pengambilan Data Multi-Tier Caching (Sangat Menghemat I/O Jaringan)
        task_1w = fetch_klines_cached(client, symbol, '1w', 4, ttl_seconds=7200)    # Simpan 2 Jam
        task_1d = fetch_klines_cached(client, symbol, '1d', 105, ttl_seconds=3600)  # Simpan 1 Jam
        task_1h = fetch_klines_cached(client, symbol, '1h', 60, ttl_seconds=300)    # Simpan 5 Menit
        task_15m = fetch_klines_cached(client, symbol, '15m', 10, ttl_seconds=60)   # Simpan 1 Menit
        task_depth = fetch_order_book_imbalance(client, symbol)                     # Real-time (Tanpa Cache)

        klines_1w, klines_1d, klines_1h, klines_15m, order_book_ratio = await asyncio.gather(
            task_1w, task_1d, task_1h, task_15m, task_depth
        )
        if not klines_1w or not klines_1d or not klines_1h or not klines_15m or len(klines_1h) < 40 or len(klines_1d) < 99: 
            return None

        try:
            coin_name = symbol.replace("USDT", "")
            w1_close, w2_close = float(klines_1w[-2][4]), float(klines_1w[-3][4])
            is_macro_bullish = w1_close >= w2_close  

            live_price = state_manager.get_live_price(symbol, float(klines_1h[-1][4]))
            btc_returns_snapshot = state_manager.get_btc_returns()

            open_price = float(klines_1h[-1][1])
            price_pct_1h = (((live_price - open_price) / open_price) * 100) if open_price > 0 else 0.0
            atr, spread_ratio = calculate_atr_and_spread(klines_1d, klines_1h)

            daily_closes = [float(k[4]) for k in klines_1d]
            hourly_closes = [float(k[4]) for k in klines_1h]
            ma25_daily, ma99_daily = calculate_ma(daily_closes, 25), calculate_ma(daily_closes, 99)

            macd_line, signal_line, macd_hist, hist_list = await asyncio.to_thread(calculate_macd_efficient, hourly_closes)

            is_ma_trend_bullish = live_price > ma25_daily and live_price > ma99_daily
            is_macd_momentum_bullish = macd_hist > 0

            vol_z_score, vol_percentile, vol_spike_ratio = calculate_volume_metrics(klines_1h, window=20)

            volatility_based_threshold = max(0.2, min(1.5, (atr / live_price) * 100 * 0.15)) if live_price > 0 else 0.4

            v_15m_curr = float(klines_15m[-1][7])
            v_15m_ma = sum(float(k[7]) for k in klines_15m[-5:-1]) / 4
            if v_15m_ma == 0: v_15m_ma = 0.00001
            is_15m_volume_burst = v_15m_curr > (v_15m_ma * 2.5)

            is_obv_healthy = calculate_obv_trend(klines_1h)

            coin_returns = []
            for i in range(-24, 0):
                try:
                    c_open = float(klines_1h[i][1])
                    c_close = float(klines_1h[i][4])
                    if c_open > 0: coin_returns.append((c_close - c_open) / c_open)
                except:
                    pass

            btc_correlation = await asyncio.to_thread(calculate_pearson_correlation, coin_returns, btc_returns_snapshot)
            is_uncorrelated_or_decoupled = btc_correlation < 0.20

            has_fvg, fvg_target_price = detect_fair_value_gap(klines_1h)

            prev_volume = float(klines_1h[-2][7]) if len(klines_1h) >= 2 else 1.0
            vol_velocity = (float(klines_1h[-1][7]) - prev_volume) / prev_volume if prev_volume > 0 else 0.0

            # ==============================================================================
            # INTEGRASI OPTIMASI SINGLE-PASS (MENGGANTIKAN PERHITUNGAN BB & KC MANUAL)
            # ==============================================================================
            ma20_hourly, bb_upper, bb_lower, kc_upper, kc_lower, is_squeeze = calculate_technical_envelope_single_pass(
                prices=hourly_closes,
                atr=atr,
                period=20,
                num_std_dev=2.0,
                num_atr_mult=1.5
            )

            is_bullish_div = False
            if len(hourly_closes) >= 10 and max(hourly_closes[-10:]) != min(hourly_closes[-10:]):
                is_bullish_div = detect_bullish_divergence(hourly_closes, hist_list, period=10)

            # Perbaikan Struktur Pasar dengan Vektorisasi NumPy (Sangat Cepat di Level C)
            market_struct, last_sh, last_sl = await asyncio.to_thread(
                analyze_market_structure, klines_1h, 5
            )

            breakout_status = verify_breakout_status(
                live_price=live_price,
                last_close=float(klines_1h[-1][4]),
                open_price=open_price,
                high_price=float(klines_1h[-1][2]),
                low_price=float(klines_1h[-1][3]),
                local_swing_high=last_sh,
                z_score=vol_z_score
            )

            market_liquidity_pool = m_data.get("pure_vol_24h", 0)
            required_vol_spike = 1.5 if market_liquidity_pool >= 50000000 else (3.5 if market_liquidity_pool <= 5000000 else 2.0)

            body_to_range_ratio = abs(live_price - open_price) / max(0.0001, float(klines_1h[-1][2]) - float(klines_1h[-1][3]))
            is_confirmed_breakout = (breakout_status == "CONFIRMED_BREAKOUT")

            is_whale_churning = (vol_spike_ratio > required_vol_spike * 1.5) and (body_to_range_ratio < 0.20)
            base_whale = 30.0 + (vol_spike_ratio * 12.0)
            if is_whale_churning: base_whale -= 25.0  
            whale_dominance = round(max(10.0, min(99.0, base_whale)), 1)

            btc_risk = calculate_btc_risk_level(state_manager.btc_status, btc_returns_snapshot)

            momentum_score, status_rencana_otomatis = calculate_confidence_score(
                market_struct=market_struct,
                breakout_status=breakout_status,
                z_score=vol_z_score,
                percentile=vol_percentile,
                ob_ratio=order_book_ratio,
                ma_bullish=is_ma_trend_bullish,
                btc_risk=btc_risk
            )

            fase = "CONSOLIDATION"
            if btc_risk["level"] == 4 and coin_name not in user_portfolio:
                fase = f"ENGINE LOCKED ({state_manager.btc_status.get('reason','CRASH')})"
            else:
                if status_rencana_otomatis == "STRONG BUY":
                    fase = "INSTITUTIONAL BUY" if is_ma_trend_bullish and order_book_ratio > 1.2 else "VALID BREAKOUT"
                elif is_squeeze and (vol_velocity > 1.8 or is_15m_volume_burst) and price_pct_1h > volatility_based_threshold:
                    fase = "⚡ SQUEEZE BREAKOUT (EARLY TREND)"
                elif abs(price_pct_1h) < volatility_based_threshold and vol_velocity > 2.5 and vol_z_score < 0.5:
                    fase = "🐳 WHALE ACCUMULATION (SILENT)"
                elif is_bullish_div and price_pct_1h > volatility_based_threshold:
                    fase = "🔄 MOMENTUM REVERSAL (BOTTOMING)"

            # Pengiriman Telegram
            if state_manager.is_alert_state_differs(coin_name, fase):
                if status_rencana_otomatis in ["STRONG_BUY", "STRONG BUY"] or fase in ["VALID BREAKOUT", "INSTITUTIONAL BUY", "⚡ SQUEEZE BREAKOUT (EARLY TREND)"]:
                    if btc_risk["allowed_trade"] or is_uncorrelated_or_decoupled:
                        emoji = "👑 BRAND NEW MSS BREAKOUT!" if fase == "INSTITUTIONAL BUY" else "🔥 BREAKOUT SPIKE"
                        fvg_info = f"\n⚠️ Fair Value Gap Spotted: Yes (Retest Area: ${fvg_target_price:.4f})" if has_fvg else ""
                        decouple_info = f"\n🔄 BTC Correlation: {btc_correlation:.2f} (Decoupled)" if is_uncorrelated_or_decoupled else f"\n🔄 BTC Correlation: {btc_correlation:.2f}"

                        harga_terformat = f"${live_price:.8f}" if live_price < 1.0 else f"${live_price:.4f}"
                        send_telegram_in_worker_thread(
                            f"{emoji}\n\nCoin: *{coin_name}*\nConfidence Score: `{momentum_score}/100` (`{status_rencana_otomatis}`)\n"
                            f"Vol Z-Score: `{vol_z_score}` (Pct: {vol_percentile}%)\n"
                            f"Structure: `{market_struct}` | Breakout: `{breakout_status}`\n"
                            f"Whale Dominance: `{whale_dominance}%`{decouple_info}{fvg_info}\n"
                            f"Live Price: *{harga_terformat}*"
                        )
                state_manager.update_alert_state(coin_name, fase)

            coin_p_data = user_portfolio.get(coin_name, {})
            entry_price = float(coin_p_data.get("costPrice") or 0.0)
            amount = float(coin_p_data.get("amount") or 0.0)

            current_peak = 0.0
            if entry_price > 0 and amount > 0:
                current_peak = state_manager.update_trailing_peak(device_id, coin_name, entry_price, live_price)

            dynamic_tp, dynamic_cl = hitung_matriks_atr_dinamis(
                live_price=live_price,
                entry_price=entry_price,
                atr=atr,
                vol_spike_ratio=vol_spike_ratio,
                whale_dominance=whale_dominance,
                btc_risk_level=btc_risk["level"],
                highest_peak=current_peak
            )

            if status_rencana_otomatis == "STRONG BUY" and entry_price == 0:
                await perf_logger.log_entry_signal_async(
                    symbol=symbol, entry_price=live_price, score=momentum_score, 
                    action=status_rencana_otomatis, z_score=vol_z_score, 
                    btc_risk_status=btc_risk["status"], tp_level=dynamic_tp, cl_level=dynamic_cl
                )

            max_allowed_atr = live_price * 0.15
            smoothed_atr = max(0.000001, min(atr, max_allowed_atr) if atr > 0 else (live_price * 0.02))

            if status_rencana_otomatis == "STRONG BUY" and has_fvg:
                saran_entry = fvg_target_price
            elif status_rencana_otomatis == "STRONG BUY" and is_confirmed_breakout:
                saran_entry = last_sh + (0.15 * smoothed_atr)
            else:
                saran_entry = live_price - (0.5 * smoothed_atr)

            if entry_price > 0:
                if live_price >= dynamic_tp: 
                    status_rencana_otomatis = "TAKE PROFIT"
                    await perf_logger.close_logged_signal_async(symbol, live_price)
                elif live_price <= dynamic_cl: 
                    status_rencana_otomatis = "CUT LOSS"
                    await perf_logger.close_logged_signal_async(symbol, live_price)
                else: 
                    status_rencana_otomatis = "HOLDING"
                saran_entry = live_price - (0.75 * smoothed_atr)

            pnl_val, pnl_pct, current_value = 0.0, 0.0, 0.0
            if entry_price > 0 and amount > 0:
                current_value = amount * live_price
                initial_value = amount * entry_price
                pnl_val = current_value - initial_value
                pnl_pct = (pnl_val / initial_value) * 100

            return {
                "koin": coin_name, "harga": live_price, "persen_harga": price_pct_1h,
                "rasio": vol_spike_ratio, "fase": fase, "atr": atr, "whale": whale_dominance, "skor": momentum_score,
                "is_portfolio": coin_name in user_portfolio,
                "amount": amount, "entry": entry_price, "tp": dynamic_tp, "cl": dynamic_cl,
                "status_aksi": status_rencana_otomatis, "saran_entry": saran_entry,
                "pnl_val": pnl_val, "pnl_pct": pnl_pct, "current_value": current_value,
                "vol_velocity_pct": f"{round(vol_velocity * 100, 1)}%", "z_score": round(vol_z_score, 2)
            }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None
