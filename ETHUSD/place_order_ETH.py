import MetaTrader5 as mt5
from atr_check_ETH import atr_stop_loss_finder  # Hàm tính ATR từ MetaTrader 5 (MT5)
import random

# Thông tin tài khoản MT5
MT5_ACCOUNT = 7510016
MT5_PASSWORD = "7lTa+zUw"
MT5_SERVER = "VantageInternational-Demo"

# Hàm kết nối với MT5
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
def get_realtime_price_mt5(symbol="ETHUSD"):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return tick.ask  # Giá mua (ask)
    else:
        print(f"Không thể lấy giá hiện tại cho {symbol}.")
        return None

# Hàm tính khối lượng giao dịch dựa trên mức rủi ro mong muốn
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
    
    print(f"Volume tính toán: {volume} lots cho rủi ro {risk_amount} USD")
    return volume

# Hàm thực hiện lệnh Market trên MT5 với tính toán volume, Stop Loss và Take Profit
def place_order_mt5(client, order_type, symbol="ETHUSD", risk_amount=60, risk_reward_ratio=1.7):
    global last_order_status
    
    # Lấy giá mark hiện tại từ MT5 để đặt lệnh
    mark_price = get_realtime_price_mt5(symbol="ETHUSD")
    if mark_price is None:
        return

    # Sử dụng hàm ATR để lấy stop_loss dựa trên ATR từ MetaTrader 5 (MT5)
    atr_symbol = "ETHUSD"  # Đổi thành ETHUSD cho MT5
    atr_short_stop_loss, atr_long_stop_loss = atr_stop_loss_finder(atr_symbol)
    
    # Xác định giá trị SL
    if order_type == "buy":
        stop_loss_price = atr_long_stop_loss
    else:
        stop_loss_price = atr_short_stop_loss
    
    # Định dạng giá trị SL thành dạng thập phân
    stop_loss_price = float(f"{stop_loss_price:.2f}")

    # Tính khối lượng giao dịch dựa trên mức rủi ro và giá trị ATR
    volume = calculate_volume_based_on_risk("ETHUSD", risk_amount, mark_price, stop_loss_price)
    if volume is None or volume <= 0:
        print("Số lượng giao dịch không hợp lệ. Hủy giao dịch.")
        return

    # Tính toán Take Profit
    risk_distance = abs(mark_price - stop_loss_price)  # Khoảng cách Risk
    reward_distance = risk_distance * risk_reward_ratio  # Khoảng cách Reward
    
    if order_type == "buy":
        take_profit_price = mark_price + reward_distance
    else:
        take_profit_price = mark_price - reward_distance

    # Định dạng TP thành dạng thập phân
    take_profit_price = float(f"{take_profit_price:.2f}")

    # In các giá trị để kiểm tra
    print(f"Giá hiện tại từ MT5: {mark_price}")
    print(f"Stop Loss dựa trên ATR: {stop_loss_price}")
    print(f"Take Profit: {take_profit_price}")
    print(f"Khối lượng giao dịch: {volume} lots")

    # Thiết lập các chế độ điền lệnh
    filling_modes = [
        mt5.ORDER_FILLING_IOC,  # Immediate or Cancel (IOC)
        mt5.ORDER_FILLING_FOK,  # Fill or Kill (FOK)
        mt5.ORDER_FILLING_RETURN  # Return (mặc định)
    ]
    
    # Thử gửi lệnh với các chế độ điền lệnh khác nhau
    for mode in filling_modes:
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "ETHUSD", 
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": mark_price,
            "sl": stop_loss_price,
            "tp": take_profit_price,
            "deviation": 20,
            "magic": 234000,
            "type_filling": mode,  # Thử từng chế độ điền lệnh
        }

        # Gửi lệnh và kiểm tra kết quả
        result = mt5.order_send(order)
        
        # Kiểm tra xem result có phải là None không
        if result is None:
            print(f"Lệnh không thể gửi với chế độ {mode}. Kiểm tra thông số lệnh.")
            continue  # Tiếp tục thử với chế độ điền lệnh khác
        elif result.retcode == mt5.TRADE_RETCODE_DONE:
            last_order_status = f"Đã {order_type} {volume} lots ETHUSD ở giá {mark_price:.2f} với Stop Loss: {stop_loss_price:.2f} và Take Profit: {take_profit_price:.2f}."
            print(last_order_status)
            break  # Nếu lệnh thành công thì thoát khỏi vòng lặp
        else:
            print(f"Lệnh không thành công với chế độ {mode}. Mã lỗi: {result.retcode}")
            print("Thông tin chi tiết:", result)
    
    # Nếu tất cả các chế độ đều thất bại, in thông báo lỗi
    print("Không thể gửi lệnh với bất kỳ chế độ điền lệnh nào.")

# Chương trình chính để kiểm tra
if __name__ == "__main__":
    # Khởi tạo Binance client (nếu cần)
    # client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    
    # Kết nối MT5 và thực hiện lệnh Market mẫu
    if connect_mt5():
        print("Thực hiện lệnh Market với tính toán volume từ mức rủi ro và ATR stop loss.")
        place_order_mt5(None, "buy", "ETHUSD", risk_amount=60)  # Thay symbol thành "ETHUSD"
        mt5.shutdown()
    else:
        print("Không thể kết nối đến MT5.")
