import MetaTrader5 as mt5
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Hàm tính xác suất kết hợp của hai mô hình không hoàn toàn độc lập
def combined_probability(p1, p2):
    combined_prob = p1 + p2 - (p1 * p2)
    return combined_prob

# Hàm lấy dữ liệu XAU/USD từ MT5
def get_realtime_klines(symbol, interval, lookback, end_time=None):
    # Map khung thời gian
    timeframes = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    # Chuyển đổi khung thời gian
    mt5_timeframe = timeframes.get(interval, mt5.TIMEFRAME_H1)

    # Lấy dữ liệu từ MT5
    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, lookback)
    if rates is None or len(rates) == 0:
        print(f"Không lấy được dữ liệu cho {symbol} với khung thời gian {interval}.")
        return None

    # Chuyển dữ liệu sang DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)

    # Tính giá trị Heikin-Ashi
    ha_open = (data['open'].shift(1) + data['close'].shift(1)) / 2
    ha_open.iloc[0] = (data['open'].iloc[0] + data['close'].iloc[0]) / 2
    ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
    ha_high = pd.concat([data['high'], ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([data['low'], ha_open, ha_close], axis=1).min(axis=1)

    data['open'] = ha_open
    data['high'] = ha_high
    data['low'] = ha_low
    data['close'] = ha_close

    # Tính các giá trị EMA (Exponential Moving Average)
    data['ema1'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema2'] = data['close'].ewm(span=11, adjust=False).mean()
    data['ema3'] = data['close'].ewm(span=15, adjust=False).mean()
    data['ema8'] = data['close'].ewm(span=34, adjust=False).mean()

    # Sinh tín hiệu EMA Ribbon
    data['Longema'] = (data['ema2'] > data['ema8']).astype(int)  # Longema: EMA2 > EMA8
    data['Redcross'] = (data['ema1'] < data['ema2']).astype(int)  # Redcross: EMA1 < EMA2
    data['Bluetriangle'] = (data['ema2'] > data['ema3']).astype(int)  # Bluetriangle: EMA2 > EMA3

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
                
        if trend == 1 and high.iloc[i] > ep:
            ep = high.iloc[i]
            af = min(af + acceleration, maximum)
        elif trend == -1 and low.iloc[i] < ep:
            ep = low.iloc[i]
            af = min(af + acceleration, maximum)
    
    data['parabolic_sar'] = sar
    return data

# Hàm phân tích xu hướng
def analyze_trend(symbol, interval, lookback):
    # Lấy dữ liệu
    data = get_realtime_klines(symbol, interval, lookback)
    if data is None:
        print(f"Lỗi: Không thể phân tích xu hướng cho {symbol} với khung thời gian {interval}.")
        return None, None, None

    # Tính RSI, MACD và Parabolic SAR
    rsi = calculate_rsi(data, 14)
    macd, signal_line = calculate_macd(data)
    data = calculate_parabolic_sar(data)

    data['rsi'] = rsi
    data['macd'] = macd
    data['signal_line'] = signal_line
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Kiểm tra nếu các cột cần thiết tồn tại
    required_columns = ['rsi', 'macd', 'signal_line', 'Longema', 'Redcross', 'Bluetriangle', 'parabolic_sar']
    for col in required_columns:
        if col not in data.columns:
            print(f"Lỗi: Cột '{col}' không tồn tại trong dữ liệu.")
            return None, None, None

    features = data[required_columns].dropna()
    target = data['target'].dropna()

    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2'],
        'max_iter': [1000]
    }
    
    grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=0, cv=5)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = grid.predict_proba(latest_features)[0]

    if 0.45 <= prediction_prob[0] <= 0.55 or 0.45 <= prediction_prob[1] <= 0.55:
        prediction_prob = None
        prediction = None
    else:
        prediction = grid.predict(latest_features)[0]

    return prediction, accuracy * 100, f1 * 100

# Hàm chính
def main():
    symbol = "XAUUSD"
    lookback = 1500

    # Phân tích xu hướng cho các khung thời gian
    trend_15m, accuracy_15m, f1_15m = analyze_trend(symbol, "15m", lookback)
    trend_h1, accuracy_h1, f1_h1 = analyze_trend(symbol, "1h", lookback)
    trend_h4, accuracy_h4, f1_h4 = analyze_trend(symbol, "4h", lookback)

    print("\n------------------- KẾT QUẢ DỰ BÁO -------------------")
    print(f"{'Khung thời gian':<15}{'Dự báo':<20}{'Độ chính xác':<20}{'Điểm F1':<20}")
    print("-----------------------------------------------------")
    
    if trend_15m is not None:
        print(f"{'15 phút':<15}{'Tăng' if trend_15m == 1 else 'Giảm' if trend_15m == 0 else 'Không rõ':<20}"
              f"{accuracy_15m:.2f}%{'':<15}{f1_15m:.2f}%")
    if trend_h1 is not None:
        print(f"{'1 giờ':<15}{'Tăng' if trend_h1 == 1 else 'Giảm' if trend_h1 == 0 else 'Không rõ':<20}"
              f"{accuracy_h1:.2f}%{'':<15}{f1_h1:.2f}%")
    if trend_h4 is not None:
        print(f"{'4 giờ':<15}{'Tăng' if trend_h4 == 1 else 'Giảm' if trend_h4 == 0 else 'Không rõ':<20}"
              f"{accuracy_h4:.2f}%{'':<15}{f1_h4:.2f}%")
    print("-----------------------------------------------------")

    if all(t is not None and t == 1 for t in [trend_15m, trend_h1, trend_h4]):
        print("\nKết luận: Xu hướng TĂNG trên cả 3 khung thời gian.")
    elif all(t is not None and t == 0 for t in [trend_15m, trend_h1, trend_h4]):
        print("\nKết luận: Xu hướng GIẢM trên cả 3 khung thời gian.")
    else:
        print("\nKết luận: Xu hướng KHÔNG RÕ RÀNG.")


if __name__ == "__main__":
    main()
    mt5.shutdown()
