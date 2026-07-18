import copy
import time
from threading import Thread, Lock
from flask import Flask, render_template, jsonify, request
import httpx
import asyncio
import numpy as np
import nest_asyncio
from asgiref.wsgi import WsgiToAsgi

# Pengaman mutlak agar pipeline async tetap berjalan di dalam worker sync Gunicorn
nest_asyncio.apply()

# Inisialisasi loop global agar persisten dan efisien
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

from config import CACHE_TTL_SECONDS
from services.binance_service import (
    check_bitcoin_circuit_breaker, get_combined_tickers_data_async
)
from services.engine import process_single_coin_pipeline, hitung_matriks_atr_dinamis
from services.telegram_service import send_telegram_in_worker_thread

flask_app = Flask(__name__)

class GlobalStateManager:
    """Mengelola data global terbagi menggunakan pengaman Thread Lock."""
    def __init__(self):
        self.lock = Lock()
        self.market_data_cache = []
        self.last_alerts_state = {}
        self.last_successful_scan_time = 0
        self.btc_returns = []
        self.live_price_map = {}
        self.btc_status = {"is_safe": True, "reason": "Connecting"}
        self.trailing_peaks = {}
        self.portfolio_dynamics = {}

    def get_live_price(self, symbol, default):
        with self.lock:
            return self.live_price_map.get(symbol, default)

    def get_btc_returns(self):
        with self.lock:
            return list(self.btc_returns)

    def is_alert_state_differs(self, coin_name, fase):
        with self.lock:
            return coin_name not in self.last_alerts_state or self.last_alerts_state[coin_name] != fase

    def update_alert_state(self, coin_name, fase):
        with self.lock:
            self.last_alerts_state[coin_name] = fase

    def update_trailing_peak(self, device_id, coin_name, entry_price, live_price):
        with self.lock:
            if device_id not in self.trailing_peaks:
                self.trailing_peaks[device_id] = {}
            old_peak = self.trailing_peaks[device_id].get(coin_name, entry_price)
            current_peak = max(old_peak, live_price)
            self.trailing_peaks[device_id][coin_name] = current_peak
            return current_peak

state = GlobalStateManager()
ENGINE_INITIALIZED = False

async def execute_one_market_scan(target_device_id=None, minimal_bootstrap=False):
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        timeout=httpx.Timeout(5.0)
    ) as brass_client:
        try:
            semaphore = asyncio.Semaphore(4)  

            # 1. Update BTC Circuit Breaker
            btc_status, btc_returns = await check_bitcoin_circuit_breaker(brass_client)
            with state.lock:
                state.btc_status = btc_status
                state.btc_returns = btc_returns

            # 2. Get Combined Ticker Data
            with state.lock:
                # Perbaikan 1: Ambil potret portofolio default agar tidak mencampuradukkan perangkat
                dev_key = target_device_id if target_device_id else "default_guest_device"
                portfolio_snapshot = copy.deepcopy(state.portfolio_dynamics.get(dev_key, {}))
                
            ticker_master_data, prices_update = await get_combined_tickers_data_async(brass_client, portfolio_snapshot)

            if not ticker_master_data:
                return

            with state.lock:
                state.live_price_map.update(prices_update)

            if minimal_bootstrap:
                ticker_master_data = dict(list(ticker_master_data.items())[:4])

            active_portfolio = {}
            with state.lock:
                if target_device_id and target_device_id in state.portfolio_dynamics:
                    active_portfolio = dict(state.portfolio_dynamics[target_device_id])

            tasks = [
                process_single_coin_pipeline(brass_client, symbol, m_data, active_portfolio, semaphore, state, dev_key) 
                for symbol, m_data in ticker_master_data.items()
            ]
            results = await asyncio.gather(*tasks)
            temp_data = [r for r in results if r is not None]
            temp_data.sort(key=lambda x: (x['is_portfolio'], x['skor']), reverse=True)

            with state.lock:
                if minimal_bootstrap and state.market_data_cache:
                    existing_coins = {x['koin'] for x in temp_data}
                    for old_item in state.market_data_cache:
                        if old_item['koin'] not in existing_coins:
                            temp_data.append(old_item)

                state.market_data_cache = temp_data
                state.last_successful_scan_time = time.time()
        except Exception as e:
            print(f"Error during core scan execution: {e}")

def run_loop_in_bg():
    # Perbaikan 3: Gunakan sub-loop terisolasi yang melekat pada thread utama agar tidak re-create event loop
    asyncio.set_event_loop(loop)
    while True:
        try:
            loop.run_until_complete(execute_one_market_scan())
        except Exception as e:
            print(f"Background Loop Error: {e}")
        time.sleep(30)  

@flask_app.before_request
def trigger_engine_startup():
    global ENGINE_INITIALIZED
    if not ENGINE_INITIALIZED:
        Thread(target=run_loop_in_bg, daemon=True).start()
        ENGINE_INITIALIZED = True

@flask_app.route('/')
def index(): 
    return render_template('index.html')

@flask_app.route('/api/data', methods=['POST'])
def get_data():
    req = request.json or {}
    device_id = req.get("device_id", "default_guest_device")

    try:
        with state.lock:
            state.portfolio_dynamics[device_id] = req.get("portfolio", {})
            if device_id in state.trailing_peaks:
                active_coins = state.portfolio_dynamics[device_id].keys()
                state.trailing_peaks[device_id] = {
                    k: v for k, v in state.trailing_peaks[device_id].items() if k in active_coins
                }
    except Exception as e:
        print(f"Failed to synchronize device dynamic cache: {e}")

    try:
        with state.lock:
            active_portfolio = dict(state.portfolio_dynamics.get(device_id, {}))
            cache_snapshot = copy.deepcopy(state.market_data_cache)
            btc_safe_snapshot = state.btc_status["is_safe"]
            btc_reason_snapshot = state.btc_status["reason"].upper()
            btc_returns_snapshot = list(state.btc_returns)

        avg_btc_return = np.mean(btc_returns_snapshot) if btc_returns_snapshot else 0.0

        if not btc_safe_snapshot and ("CRASH" in btc_reason_snapshot or "CAPITULATION" in btc_reason_snapshot or avg_btc_return < -0.04):
            btc_risk_level = 4
        elif not btc_safe_snapshot or "BREAKDOWN" in btc_reason_snapshot or avg_btc_return < -0.02:
            btc_risk_level = 3
        elif "SQUEEZE" in btc_reason_snapshot or "CONSOLIDATION" in btc_reason_snapshot or -0.01 <= avg_btc_return < 0.01:
            btc_risk_level = 2
        else:
            btc_risk_level = 1

        user_market_data = []

        for original_item in cache_snapshot:
            item = copy.deepcopy(original_item)  
            coin = item["koin"]

            if coin in active_portfolio:
                item["is_portfolio"] = True
                coin_p_data = active_portfolio[coin]
                item["amount"] = coin_p_data.get("amount", 0.0)
                item["entry"] = coin_p_data.get("costPrice", 0.0)

                current_peak = 0.0
                if item["entry"] > 0 and item["amount"] > 0:
                    # Perbaikan 2: Fungsi dipanggil dengan proteksi teratur untuk mencegah tabrakan data internal thread
                    current_peak = state.update_trailing_peak(device_id, coin, item["entry"], item["harga"])

                if item["entry"] > 0 and item["amount"] > 0:
                    dtp, dcl = hitung_matriks_atr_dinamis(
                        live_price=item["harga"],
                        entry_price=item["entry"],
                        atr=item["atr"],
                        vol_spike_ratio=item["rasio"],
                        whale_dominance=item["whale"],
                        btc_risk_level=btc_risk_level,
                        highest_peak=current_peak
                    )
                    item["tp"] = dtp
                    item["cl"] = dcl
                    item["current_value"] = item["amount"] * item["harga"]
                    initial_val = item["amount"] * item["entry"]
                    item["pnl_val"] = item["current_value"] - initial_val
                    item["pnl_pct"] = (item["pnl_val"] / initial_val) * 100

                    if item["harga"] >= item["tp"]: 
                        item["status_aksi"] = "TAKE PROFIT"
                    elif item["harga"] <= item["cl"]: 
                        item["status_aksi"] = "CUT LOSS"
                    else: 
                        item["status_aksi"] = "HOLDING"
            else:
                item["is_portfolio"] = False
                item["amount"] = 0.0
                item["entry"] = 0.0

                dtp, dcl = hitung_matriks_atr_dinamis(
                    live_price=item["harga"],
                    entry_price=0.0,
                    atr=item["atr"],
                    vol_spike_ratio=item["rasio"],
                    whale_dominance=item["whale"],
                    btc_risk_level=btc_risk_level,
                    highest_peak=0.0
                )
                item["tp"] = dtp
                item["cl"] = dcl
                item["pnl_val"] = 0.0
                item["pnl_pct"] = 0.0
                item["current_value"] = 0.0

                if "ENGINE LOCKED" not in item["fase"]:
                    if item["fase"] in ["INSTITUTIONAL BUY", "VALID BREAKOUT", "EARLY RALLY", "⚡ SQUEEZE BREAKOUT (EARLY TREND)", "🐳 WHALE ACCUMULATION (SILENT)", "🔄 MOMENTUM REVERSAL (BOTTOMING)"]:
                        item["status_aksi"] = "BUY STAGE"
                    elif item["fase"] == "OVERBOUGHT PEAK":
                        item["status_aksi"] = "TAKE PROFIT"
                    else:
                        item["status_aksi"] = "WAIT & SEE"

            user_market_data.append(item)

        user_market_data.sort(key=lambda x: (x['is_portfolio'], x['skor']), reverse=True)

        with state.lock:
            btc_status_response = dict(state.btc_status)
            btc_status_response["risk_level"] = btc_risk_level

        return jsonify({
            "btc_status": btc_status_response,
            "market": user_market_data
        }), 200

    except Exception as e:
        flask_app.logger.error(f"Error executing quantitative data route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@flask_app.route('/api/telegram/send_manual', methods=['POST'])
def send_manual_alert():
    try:
        req = request.json
        coin = req.get("koin", "").strip().upper()
        fase = req.get("fase", "MONITORING")
        harga = float(req.get("harga", 0))
        rasio = float(req.get("rasio", 0))
        whale = float(req.get("whale", 0))
        status_aksi = req.get("status_aksi", "WAIT & SEE")
        saran = float(req.get("saran_entry", 0))

        harga_fmt = f"${harga:.8f}" if harga < 1.0 else f"${harga:.4f}"
        saran_fmt = f"${saran:.8f}" if saran < 1.0 else f"${saran:.4f}"

        msg = (
            f"📢 *MANUAL QUANTUM SIGNAL ALERT*\n\n"
            f"Coin: *{coin}*\n"
            f"Market Phase: {fase}\n"
            f"Current Price: {harga_fmt}\n"
            f"Vol vs MA20: {rasio:.1f}x\n"
            f"Whale Dominance: {whale}%\n"
            f"Action Strategy: *{status_aksi}*\n"
            f"Suggested Entry Trigger: {saran_fmt}"
        )

        send_telegram_in_worker_thread(msg)
        return jsonify({"status": "success", "message": "Signal broadcast initiated!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# Ekspos aplikasi sebagai komponen ASGI untuk Gunicorn Uvicorn Worker
app = WsgiToAsgi(flask_app)

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000, debug=False)
