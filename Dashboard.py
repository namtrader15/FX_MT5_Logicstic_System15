from flask import Flask, render_template, request
import MetaTrader5 as mt5

app = Flask(__name__)

# Hàm kết nối MT5
def connect_mt5(account_number, password, server):
    if not mt5.initialize():
        print(f"Không thể khởi tạo MT5, lỗi: {mt5.last_error()}")
        return False

    if not mt5.login(account_number, password=password, server=server):
        print(f"Đăng nhập thất bại, lỗi: {mt5.last_error()}")
        return False

    return True

# Hàm đóng lệnh
def close_trade(symbol):
    # Lấy danh sách lệnh đang mở với symbol được chỉ định
    positions = mt5.positions_get(symbol=symbol)
    if not positions or len(positions) == 0:
        return f"Không tìm thấy lệnh mở cho {symbol}."

    # Đóng từng lệnh
    for position in positions:
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask,
            "deviation": 10,
            "type_time": mt5.ORDER_TIME_GTC,  # Chuyển sang ORDER_TIME_GTC
            "type_filling": mt5.ORDER_FILLING_IOC,  # Giữ chế độ IOC cho việc khớp lệnh
        }

        # Gửi yêu cầu đóng lệnh
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return f"Lỗi đóng lệnh {symbol}: {result.retcode} - {result.comment}"
    
    return f"Đóng lệnh thành công cho {symbol}."

#Route đóng lệnh
@app.route("/close_trade", methods=["POST"])
def close_trade_route():
    symbol = request.form.get("symbol")
    if not symbol:
        return "Không có thông tin tài sản để đóng lệnh.", 400

    # Kết nối tới MT5
    ACCOUNT_NUMBER = 7510016
    PASSWORD = "7lTa+zUw"
    SERVER = "VantageInternational-Demo"

    if not connect_mt5(ACCOUNT_NUMBER, PASSWORD, SERVER):
        return "Không thể kết nối đến MetaTrader 5.", 500

    # Đóng lệnh và xử lý kết quả
    result = close_trade(symbol)
    mt5.shutdown()  # Đảm bảo tắt kết nối
    return result

# Route chính hiển thị giao dịch
@app.route("/")
def index():
    # Kết nối tới MT5 và lấy dữ liệu
    ACCOUNT_NUMBER = 7510016
    PASSWORD = "7lTa+zUw"
    SERVER = "VantageInternational-Demo"

    if not connect_mt5(ACCOUNT_NUMBER, PASSWORD, SERVER):
        return "Không thể kết nối đến tài khoản MT5!"

    # Lấy danh sách các lệnh đang mở
    positions = mt5.positions_get()
    trades = []
    if positions:
        for pos in positions:
            trades.append({
                "Loại tài sản": pos.symbol,
                "Giá vào lệnh": pos.price_open,
                "Loại lệnh": "Buy" if pos.type == mt5.ORDER_TYPE_BUY else "Sell",
                "TP": pos.tp,
                "SL": pos.sl,
                "Profit": pos.profit,
            })

    mt5.shutdown()  # Đóng kết nối sau khi lấy dữ liệu
    return render_template("index.html", trades=trades)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
