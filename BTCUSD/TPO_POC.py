import numpy as np
from binance.client import Client

def calculate_poc_value(client):
    # Lấy dữ liệu giá lịch sử BTC/USDT, khung thời gian 5 phút
    candlesticks = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_5MINUTE)

    # Trích xuất giá cao và thấp
    highs = np.array([float(candle[2]) for candle in candlesticks])
    lows = np.array([float(candle[3]) for candle in candlesticks])

    # Số lượng phân đoạn kênh giá
    num_channels = 20

    # Tính giá cao nhất và thấp nhất
    highest = np.max(highs)
    lowest = np.min(lows)

    # Tính chiều rộng của mỗi kênh giá
    channel_width = (highest - lowest) / num_channels

    # Tính TPO cho mỗi kênh giá
    def get_tpo(lower, upper, highs, lows):
        count = 0
        for high, low in zip(highs, lows):
            if (low <= upper and high >= lower):
                count += 1
        return count

    # Tạo mảng để lưu trữ TPO của mỗi kênh giá
    tpos = []
    for i in range(num_channels):
        lower = lowest + i * channel_width
        upper = lower + channel_width
        tpo = get_tpo(lower, upper, highs, lows)
        tpos.append(tpo)

    # Tìm POC (kênh giá có nhiều TPO nhất)
    poc_index = np.argmax(tpos)
    poc_lower = lowest + poc_index * channel_width
    poc_upper = poc_lower + channel_width
    poc_value = (poc_lower + poc_upper) / 2

    return poc_value

# Hàm chính để kiểm tra TPO POC cho BTCUSDT
def main():
    # Tạo client Binance
    client = Client("your_binance_api_key", "your_binance_secret_key")  # Thay thế với API key của bạn

    # Tính giá trị POC
    poc_value = calculate_poc_value(client)

    # Hiển thị giá trị POC
    if poc_value:
        print(f"Giá trị POC của BTCUSDT: {poc_value:.2f}")

if __name__ == "__main__":
    main()
