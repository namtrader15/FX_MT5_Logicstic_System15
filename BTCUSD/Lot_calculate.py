import MetaTrader5 as mt5
from atr_check import atr_stop_loss_finder  # Hàm tính ATR từ Binance
from binance.client import Client

# Thông tin tài khoản MT5
MT5_ACCOUNT = 24492270
MT5_PASSWORD = 'obpaPLEJ.~39'
MT5_SERVER = 'FivePercentOnline-Real'

# Thông tin API Binance
BINANCE_API_KEY = "your_binance_api_key"
BINANCE_SECRET_KEY = "your_binance_secret_key"

# Hàm kết nối MT5
def connect_mt5():
    if not mt5.initialize():
        print("Lỗi khi khởi động MT5:", mt5.last_error())
        return False
    
    authorized = mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
    if not authorized:
        error_code, error_message = mt5.last_error()
        print(f"Lỗi kết nối đến MT5: Mã lỗi {error_code} - {error_message}")
        mt5.shutdown()
        return False
    
    print("Kết nối thành công đến MT5 với tài khoản:", MT5_ACCOUNT)
    return True

# Hàm lấy giá mark từ MT5
def get_realtime_price_mt5(symbol="BTCUSD"):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return tick.ask  # Giá mua (ask)
    else:
        print(f"Không thể lấy giá hiện tại cho {symbol}.")
        return None

# Hàm tính volume dựa trên ATR
def calculate_volume_based_on_risk(symbol, risk_amount, market_price, stop_loss_price):
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Không thể lấy thông tin cho {symbol}")
        return None

    # Kích thước hợp đồng giao dịch
    contract_size = symbol_info.trade_contract_size

    # Khoảng cách từ giá vào lệnh đến Stop Loss
    distance = abs(market_price - stop_loss_price)

    # Tính khối lượng giao dịch (Volume)
    volume = risk_amount / (distance * contract_size)

    # Làm tròn volume theo bước lot tối thiểu của broker
    volume_step_decimal_places = len(str(symbol_info.volume_step).split(".")[-1])
    volume = max(symbol_info.volume_min, round(volume, volume_step_decimal_places))
    
    return volume

# Công cụ tính volume
def calculate_volume_tool():
    # Nhập Risk Amount từ bàn phím
    risk_amount = float(input("Nhập mức rủi ro (Risk Amount, USD): "))

    # Khởi tạo Binance client
    client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)

    # Kết nối MT5
    if not connect_mt5():
        return
    
    # Lấy giá hiện tại của BTCUSD từ MT5
    market_price = get_realtime_price_mt5(symbol="BTCUSD")
    if market_price is None:
        return

    # Lấy ATR và tính Stop Loss từ Binance
    atr_symbol = "BTCUSDT"  # Tương ứng trên Binance
    atr_short_stop_loss, atr_long_stop_loss = atr_stop_loss_finder(client, atr_symbol)

    # Tính volume cho lệnh Buy
    buy_volume = calculate_volume_based_on_risk("BTCUSD", risk_amount, market_price, atr_long_stop_loss)
    if buy_volume is None:
        print("Không thể tính toán volume cho lệnh Buy.")
        return

    # Tính volume cho lệnh Sell
    sell_volume = calculate_volume_based_on_risk("BTCUSD", risk_amount, market_price, atr_short_stop_loss)
    if sell_volume is None:
        print("Không thể tính toán volume cho lệnh Sell.")
        return

    # Hiển thị kết quả
    print(f"\nVolume tính toán cho lệnh Buy (Long): {buy_volume} lots")
    print(f"Volume tính toán cho lệnh Sell (Short): {sell_volume} lots")

    # Dừng màn hình để xem kết quả
    input("\nNhấn Enter để thoát...")

    # Đóng kết nối MT5
    mt5.shutdown()

# Chạy chương trình
if __name__ == "__main__":
    calculate_volume_tool()
