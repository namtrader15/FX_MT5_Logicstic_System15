import numpy as np
import MetaTrader5 as mt5

def calculate_poc_value_USDJPY(symbol="USDJPY", lookback=500, num_channels=20):
    # Lấy dữ liệu giá lịch sử USDJPY từ MetaTrader 5, khung thời gian 5 phút
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, lookback)
    if rates is None:
        print(f"Không lấy được dữ liệu {symbol}")
        return None

    # Trích xuất giá cao và thấp
    highs = np.array([rate[2] for rate in rates])  # Giá cao (high)
    lows = np.array([rate[3] for rate in rates])   # Giá thấp (low)

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

# Hàm chính để kiểm tra TPO POC cho USDJPY
def main():
    if not mt5.initialize():
        print("Không thể kết nối đến MetaTrader 5.")
        mt5.shutdown()
        return
    
    poc_value = calculate_poc_value_USDJPY(symbol="USDJPY", lookback=500, num_channels=20)
    if poc_value:
        print(f"Giá trị POC của USDJPY: {poc_value}")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
