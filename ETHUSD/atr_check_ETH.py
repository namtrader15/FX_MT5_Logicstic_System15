import MetaTrader5 as mt5
import pandas as pd  # Đảm bảo pandas đã được import
from datetime import datetime

# Hàm tính toán smoothing (EMA, SMA, WMA, RMA) cho ATR
def ma_function(source, length, smoothing="RMA"):
    if smoothing == "RMA":
        return rma(source, length)
    elif smoothing == "SMA":
        return sma(source, length)
    elif smoothing == "EMA":
        return ema(source, length)
    elif smoothing == "WMA":
        return wma(source, length)

# Hàm tính RMA (Relative Moving Average)
def rma(source, length):
    alpha = 1 / length
    rma_val = [source[0]]  # Giá trị ban đầu là giá trị đầu tiên
    for i in range(1, len(source)):
        rma_val.append(alpha * source[i] + (1 - alpha) * rma_val[i - 1])
    return rma_val[-1]

# Hàm tính ATR Stop Loss Finder
def atr_stop_loss_finder(symbol, length=14, multiplier=1.5, smoothing="RMA"): 
    # Lấy dữ liệu nến từ MetaTrader 5 (MT5) cho XAUUSD, khung thời gian 1 giờ
    timeframes = {
        "1h": mt5.TIMEFRAME_H1,  # 1 giờ
    }

    # Lấy dữ liệu từ MT5
    rates = mt5.copy_rates_from_pos(symbol, timeframes["1h"], 0, length + 1)
    if rates is None:
        print(f"Không lấy được dữ liệu cho {symbol}")
        return None

    # Chuyển đổi dữ liệu sang DataFrame
    data = pd.DataFrame(rates)  # Đảm bảo rằng pandas đã được import
    highs = data['high'].values
    lows = data['low'].values
    closes = data['close'].values

    # Tính True Range (TR)
    tr_values = []
    for i in range(1, len(rates)):
        high = highs[i]
        low = lows[i]
        close_prev = closes[i - 1]
        tr = max(high - low, abs(high - close_prev), abs(low - close_prev))
        tr_values.append(tr)

    # Tính ATR bằng công thức smoothing
    atr_value = ma_function(tr_values, length, smoothing) * multiplier

    # Tính toán các mức Stop Loss
    atr_short_stop_loss = round(highs[-1] + atr_value)  # ATR cho Short Stop Loss
    atr_long_stop_loss = round(lows[-1] - atr_value)    # ATR cho Long Stop Loss

    # In ra các giá trị để anh kiểm tra
    print(f"Giá trị ATR: {atr_value:.2f}")
    print(f"ATR Short Stop Loss: {atr_short_stop_loss}")
    print(f"ATR Long Stop Loss: {atr_long_stop_loss}")
    print(f"Giá cao nhất cây nến cuối: {highs[-1]}")
    print(f"Giá thấp nhất cây nến cuối: {lows[-1]}")

    return atr_short_stop_loss, atr_long_stop_loss

# Hàm chính để chạy chương trình tính ATR
def main():
    # Kết nối với MetaTrader 5
    if not mt5.initialize():
        print("Không thể kết nối tới MetaTrader 5!")
        return

    symbol = "XAUUSD"  # Thay thế thành cặp giao dịch XAUUSD
    atr_stop_loss_finder(symbol)

    # Ngắt kết nối MT5 sau khi sử dụng
    mt5.shutdown()

if __name__ == "__main__":
    main()
