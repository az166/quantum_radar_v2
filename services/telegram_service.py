import asyncio
import httpx
from threading import Thread
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

async def send_telegram_alert_async(message):
    if not TELEGRAM_TOKEN or "ENTER_TOKEN" in TELEGRAM_TOKEN:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload, timeout=5)
            return res.status_code == 200
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")
        return False

def send_telegram_in_worker_thread(message):
    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(send_telegram_alert_async(message))
        finally:
            loop.close()
    Thread(target=worker, daemon=True).start()
