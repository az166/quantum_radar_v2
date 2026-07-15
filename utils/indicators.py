import math

def calculate_ma(prices, period):
    if len(prices) < period: 
        return sum(prices) / len(prices) if prices else 0.0
    return sum(prices[-period:]) / period

def calculate_std_dev(prices, period):
    if len(prices) < period: 
        return 0.00001
    ma = sum(prices[-period:]) / period
    variance = sum((x - ma) ** 2 for x in prices[-period:]) / period
    return math.sqrt(variance) if variance > 0 else 0.00001

def calculate_obv_trend(klines_1h):
    if len(klines_1h) < 10: 
        return True
    obv = 0
    obv_values = []
    for i in range(1, len(klines_1h)):
        close_curr = float(klines_1h[i][4])
        close_prev = float(klines_1h[i-1][4])
        vol = float(klines_1h[i][7])
        if close_curr > close_prev:
            obv += vol
        elif close_curr < close_prev:
            obv -= vol
        obv_values.append(obv)

    recent_obv = sum(obv_values[-3:]) / 3
    base_obv = sum(obv_values[-10:]) / 10
    return recent_obv >= base_obv

def detect_bullish_divergence(prices, macd_hists, period=10):
    if len(prices) < period or len(macd_hists) < period: 
        return False
    recent_prices = prices[-period:]
    recent_macd = macd_hists[-period:]
    min_p_idx = recent_prices.index(min(recent_prices))

    if 0 < min_p_idx < period - 1:
        if recent_prices[-1] <= recent_prices[min_p_idx] and recent_macd[-1] > recent_macd[min_p_idx]:
            return True
    return False

def calculate_macd_efficient(prices):
    if len(prices) < 35: 
        return 0.0, 0.0, 0.0, []

    def get_ema_list(data, period):
        alpha = 2 / (period + 1)
        ema_res = []
        current_ema = sum(data[:period]) / period
        ema_res.append(current_ema)
        for price in data[period:]:
            current_ema = (price * alpha) + (current_ema * (1 - alpha))
            ema_res.append(current_ema)
        return ema_res

    ema12_list = get_ema_list(prices, 12)
    ema26_list = get_ema_list(prices, 26)
    offset = len(ema12_list) - len(ema26_list)
    macd_line = [e12 - e26 for e12, e26 in zip(ema12_list[offset:], ema26_list)]

    if len(macd_line) < 9:
        return 0.0, 0.0, 0.0, []

    signal_alpha = 2 / (9 + 1)
    current_signal = sum(macd_line[:9]) / 9
    for m_val in macd_line[9:]:
        current_signal = (m_val * signal_alpha) + (current_signal * (1 - signal_alpha))

    current_macd = macd_line[-1]
    hist_list = [m_v - current_signal for m_v in macd_line]
    return current_macd, current_signal, current_macd - current_signal, hist_list

def calculate_pearson_correlation(coin_returns, btc_returns):
    if not coin_returns or not btc_returns or len(coin_returns) != len(btc_returns):
        return 1.0  

    n = len(coin_returns)
    mean_x = sum(coin_returns) / n
    mean_y = sum(btc_returns) / n

    num = sum((coin_returns[i] - mean_x) * (btc_returns[i] - mean_y) for i in range(n))
    den_x = sum((coin_returns[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((btc_returns[i] - mean_y) ** 2 for i in range(n))

    if den_x == 0 or den_y == 0:
        return 1.0
    return num / math.sqrt(den_x * den_y)
