from binance.client import Client
import pandas as pd
import numpy as np

# Kết nối API Binance
api_key = "YOUR_API_KEY"  # Thay bằng API Key của anh
api_secret = "YOUR_API_SECRET"  # Thay bằng API Secret của anh
client = Client(api_key, api_secret)

# Hàm lấy dữ liệu nến từ Binance API
def get_klines(symbol, interval, lookback):
    """
    Lấy dữ liệu nến từ Binance API.
    :param symbol: Tên cặp giao dịch (VD: BTCUSDT)
    :param interval: Khung thời gian nến (VD: 5m, 15m)
    :param lookback: Số lượng nến cần lấy
    :return: DataFrame chứa dữ liệu nến
    """
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        # Chuyển đổi các cột cần thiết sang kiểu số
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu nến: {e}")
        return pd.DataFrame()

# Hàm tính toán VWAP và các vùng Over Sold / Over Bought
def calculate_vwap_and_zones(data, dev_up=1.28, dev_dn=1.28):
    """
    Tính toán VWAP và các vùng Over Sold / Over Bought.
    :param data: DataFrame chứa dữ liệu nến
    :param dev_up: Hệ số độ lệch chuẩn để xác định vùng Overbought
    :param dev_dn: Hệ số độ lệch chuẩn để xác định vùng Oversold
    :return: DataFrame với các cột mới cho VWAP, oversold_zone, overbought_zone
    """
    if data.empty or len(data) < 2:
        print("Dữ liệu không đủ để tính toán VWAP.")
        return pd.DataFrame()
    
    try:
        # Tính giá trung bình (HLC Avg)
        data['hlc_avg'] = (data['high'] + data['low'] + data['close']) / 3
        # Tính toán VWAP
        data['vwap_cumsum'] = (data['hlc_avg'] * data['volume']).cumsum()
        data['vol_cumsum'] = data['volume'].cumsum()
        data['vwap'] = data['vwap_cumsum'] / data['vol_cumsum']
        
        # Tính độ lệch chuẩn
        data['price_squared'] = data['hlc_avg'] ** 2
        data['price_vol_cumsum'] = (data['price_squared'] * data['volume']).cumsum()
        
        # Đảm bảo không có chia cho 0 hoặc giá trị âm trong căn bậc hai
        data['std_dev'] = np.sqrt(
            np.maximum(
                data['price_vol_cumsum'] / data['vol_cumsum'] - data['vwap'] ** 2,
                0  # Đảm bảo giá trị >= 0
            )
        )
        
        # Xác định các vùng Over Sold và Over Bought
        data['oversold_zone'] = data['vwap'] - dev_dn * data['std_dev']
        data['overbought_zone'] = data['vwap'] + dev_up * data['std_dev']
        
        return data[['vwap', 'oversold_zone', 'overbought_zone', 'close']]
    except Exception as e:
        print(f"Lỗi khi tính toán VWAP: {e}")
        return pd.DataFrame()


# Hàm lấy giá real-time từ Binance API
def check_realtime_price(symbol):
    """
    Lấy giá real-time từ Binance API.
    :param symbol: Tên cặp giao dịch (VD: BTCUSDT)
    :return: Giá hiện tại của cặp giao dịch
    """
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Lỗi khi lấy giá real-time: {e}")
        return None

# Hàm trả về tín hiệu giao dịch dựa trên VWAP
def get_vwap_signal(symbol, dev_up=1.28, dev_dn=1.28, interval='5m', lookback=50):
    """
    Trả về tín hiệu giao dịch dựa trên VWAP.
    :param symbol: Tên cặp giao dịch (VD: BTCUSDT)
    :param dev_up: Hệ số độ lệch chuẩn để xác định vùng Overbought
    :param dev_dn: Hệ số độ lệch chuẩn để xác định vùng Oversold
    :param interval: Khung thời gian nến (VD: 5m, 15m)
    :param lookback: Số lượng nến cần lấy
    :return: 1 (Mua), 0 (Bán), None (Không hành động)
    """
    try:
        # Lấy dữ liệu nến
        data = get_klines(symbol, interval, lookback)
        if data.empty:
            return None
        
        # Tính toán VWAP và các vùng Over Sold / Over Bought
        data = calculate_vwap_and_zones(data, dev_up, dev_dn)
        if data.empty or 'oversold_zone' not in data.columns:
            return None

        # Lấy giá real-time
        real_time_price = check_realtime_price(symbol)
        if real_time_price is None:
            return None
        
        # Lấy thông tin từ nến cuối cùng
        latest_data = data.iloc[-1]
        oversold = latest_data['oversold_zone']
        overbought = latest_data['overbought_zone']
        
        # So sánh giá real-time với các vùng
        if real_time_price <= oversold:
            return 1  # Giá nằm trong vùng Over Sold
        elif real_time_price >= overbought:
            return 0  # Giá nằm trong vùng Over Bought
        else:
            return None  # Giá không nằm trong các vùng này
    except Exception as e:
        print(f"Lỗi trong hàm get_vwap_signal: {e}")
        return None

# Ví dụ sử dụng hàm
if __name__ == "__main__":
    symbol = "BTCUSDT"  # Thay bằng cặp giao dịch của anh
    signal = get_vwap_signal(symbol, dev_up=1.28, dev_dn=1.28, interval='5m', lookback=50)
    print(f"Trading Signal: {signal}")
