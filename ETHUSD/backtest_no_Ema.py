import MetaTrader5 as mt5
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime

# Kết nối MetaTrader 5
if not mt5.initialize():
    print("Kết nối MT5 thất bại!")
    mt5.shutdown()

# Đăng nhập tài khoản MT5 nếu cần
account = 7510016  # Thay bằng số tài khoản của anh
password = "7lTa+zUw"  # Thay bằng mật khẩu tài khoản
server = "VantageInternational-Demo"  # Thay bằng tên server
if not mt5.login(account, password, server):
    print(f"Đăng nhập thất bại, lỗi: {mt5.last_error()}")
    mt5.shutdown()

# Hàm lấy dữ liệu XAU/USD từ MT5
def get_realtime_klines(symbol, timeframe, lookback):
    # Map khung thời gian
    timeframes = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,  # Thêm khung thời gian 15 phút
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    
    # Chuyển đổi khung thời gian
    mt5_timeframe = timeframes.get(timeframe, mt5.TIMEFRAME_H1)
    
    # Lấy dữ liệu từ MT5
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, lookback)
    if rates is None:
        print(f"Không lấy được dữ liệu {symbol}")
        return None

    # Chuyển dữ liệu sang DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    
    # Đổi tên cột cho phù hợp với các tính toán
    data.rename(columns={
        'open': 'open', 
        'high': 'high', 
        'low': 'low', 
        'close': 'close', 
        'tick_volume': 'volume'
    }, inplace=True)
    
    return data


# Hàm tính nến Heikin-Ashi
def calculate_heikin_ashi(data):
    # Tính toán các giá trị Heikin-Ashi
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
    ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2  # Khởi tạo giá trị đầu tiên
    ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)
    
    # Thay thế dữ liệu bằng nến Heikin-Ashi
    data['open'] = ha_open
    data['high'] = ha_high
    data['low'] = ha_low
    data['close'] = ha_close
    return data

# Hàm tính RSI
def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Hàm tính MACD
def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Hàm tính Parabolic SAR
def calculate_parabolic_sar(data, acceleration=0.02, maximum=0.2):
    high = data['high']
    low = data['low']
    close = data['close']
    
    sar = [close.iloc[0]]  # Bắt đầu bằng giá đóng cửa đầu tiên
    ep = high.iloc[0]  # Extreme Point (Điểm cực đại)
    af = acceleration  # Hệ số gia tốc ban đầu
    trend = 1  # Bắt đầu với giả định xu hướng tăng
    
    for i in range(1, len(close)):
        if trend == 1:  # Xu hướng tăng
            sar.append(sar[i-1] + af * (ep - sar[i-1]))
            if low.iloc[i] < sar[i]:  # Đảo chiều sang xu hướng giảm
                trend = -1
                sar[i] = ep
                af = acceleration
                ep = low.iloc[i]
        else:  # Xu hướng giảm
            sar.append(sar[i-1] + af * (ep - sar[i-1]))
            if high.iloc[i] > sar[i]:  # Đảo chiều sang xu hướng tăng
                trend = 1
                sar[i] = ep
                af = acceleration
                ep = high.iloc[i]
                
        # Điều chỉnh Extreme Point và hệ số gia tốc
        if trend == 1 and high.iloc[i] > ep:
            ep = high.iloc[i]
            af = min(af + acceleration, maximum)
        elif trend == -1 and low.iloc[i] < ep:
            ep = low.iloc[i]
            af = min(af + acceleration, maximum)
    
    data['parabolic_sar'] = sar
    return data


# Hàm phân tích xu hướng
def analyze_trend(interval, name):
    # Lấy dữ liệu XAU/USD
    symbol = "ETHUSD"
    lookback = 1500
    data = get_realtime_klines(symbol, interval, lookback)
    if data is None:
        return None

    # Tính nến Heikin-Ashi
    data = calculate_heikin_ashi(data)

    # Tính RSI, MACD và Parabolic SAR
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)
    data = calculate_parabolic_sar(data)

    # Thêm cột chỉ báo vào DataFrame
    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    data['sar'] = data['parabolic_sar']
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Chuẩn bị dữ liệu cho mô hình học máy
    features = data[['rsi', 'macd', 'signal_line', 'sar']].dropna()
    target = data['target'].dropna()

    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Huấn luyện Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Dự đoán xu hướng thời gian thực
    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = model.predict_proba(latest_features)[0]  # Lấy xác suất cho các lớp

    # Hiển thị mức độ ngưỡng (xác suất) cho xu hướng giảm (0) và xu hướng tăng (1)
    trend = "Tăng" if prediction_prob[1] >= 0.55 else "Giảm"
    print(f"{name}: Xu hướng {trend}, Xác suất xu hướng tăng: {prediction_prob[1]:.2f}, Xác suất xu hướng giảm: {prediction_prob[0]:.2f}")
    print(f"Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
    return prediction_prob[1]  # Trả về xác suất của xu hướng tăng

# Hàm chính
def main():
    trend_h1 = analyze_trend("1h", "H1")
    trend_h4 = analyze_trend("4h", "H4")
    trend_15m = analyze_trend("15m", "15m")  # Bổ sung khung thời gian 15 phút

    if trend_h1 >= 0.55 and trend_h4 >= 0.55 and trend_15m >= 0.55:
        print("Xu hướng tăng!")
    elif trend_h1 <= 0.45 and trend_h4 <= 0.45 and trend_15m <= 0.45:
        print("Xu hướng giảm!")
    else:
        print("Xu hướng không rõ ràng!")


if __name__ == "__main__":
    main()
    mt5.shutdown()
