from binance.client import Client

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
def atr_stop_loss_finder(client, symbol, length=14, multiplier=1.5, smoothing="RMA"): 
    # Lấy dữ liệu nến từ Binance theo khung 1 giờ
    klines = client.futures_klines(symbol=symbol, interval='1h', limit=length + 1)

    # Tách dữ liệu giá cao nhất, thấp nhất và giá đóng cửa
    highs = [float(kline[2]) for kline in klines]
    lows = [float(kline[3]) for kline in klines]
    closes = [float(kline[4]) for kline in klines]

    # Tính True Range (TR)
    tr_values = []
    for i in range(1, len(klines)):
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
    api_key = 'api_key'  # Thay thế bằng API key của anh
    api_secret = 'api_secret'  # Thay thế bằng API secret của anh
    client = Client(api_key, api_secret, tld='com', testnet=False)

    symbol = 'BTCUSDT'  # Thay thế bằng cặp giao dịch anh muốn
    atr_stop_loss_finder(client, symbol)

if __name__ == "__main__":
    main()
