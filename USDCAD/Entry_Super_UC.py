import MetaTrader5 as mt5
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from datetime import datetime
import pytz

# Hàm tính xác suất kết hợp của hai mô hình không hoàn toàn độc lập
def combined_probability(p1, p2):
    combined_prob = p1 + p2 - (p1 * p2)
    return combined_prob

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

# Hàm tính EMA Ribbon và các tín hiệu liên quan
def calculate_ema_ribbon(data):
    # Tính toán các đường EMA
    data['ema1'] = data['close'].ewm(span=5, adjust=False).mean()
    data['ema2'] = data['close'].ewm(span=11, adjust=False).mean()
    data['ema3'] = data['close'].ewm(span=15, adjust=False).mean()
    data['ema8'] = data['close'].ewm(span=34, adjust=False).mean()
    
    # Tạo tín hiệu từ EMA Ribbon
    data['Longema'] = (data['ema2'] > data['ema8']).astype(int)  # Longema: EMA2 > EMA8
    data['Redcross'] = (data['ema1'] < data['ema2']).astype(int)  # Redcross: EMA1 < EMA2
    data['Bluetriangle'] = (data['ema2'] > data['ema3']).astype(int)  # Bluetriangle: EMA2 > EMA3
    return data

# Hàm lấy dữ liệu từ MetaTrader 5 và tính toán Heikin-Ashi
def get_realtime_klines(symbol, timeframe, lookback):
    # Định nghĩa các khung thời gian
    timeframes = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    
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

    data.rename(columns={'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'tick_volume': 'volume'}, inplace=True)

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

    return data

# Phân tích xu hướng
def analyze_trend(symbol, interval, lookback):
    data = get_realtime_klines(symbol, interval, lookback)
    if data is None:
        print(f"Lỗi: Không thể phân tích xu hướng cho {symbol} với khung thời gian {interval}.")
        return None, None, None

    # Tính toán chỉ báo
    data['rsi'] = calculate_rsi(data, 14)
    data['macd'], data['signal_line'] = calculate_macd(data)
    data = calculate_parabolic_sar(data)
    data = calculate_ema_ribbon(data)

    # Tạo biến target cho học máy
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    features = data[['rsi', 'macd', 'signal_line', 'parabolic_sar', 'Longema', 'Redcross', 'Bluetriangle']].dropna()
    target = data['target'].dropna()

    # Đảm bảo số lượng hàng khớp nhau
    min_length = min(len(features), len(target))
    features = features.iloc[:min_length]
    target = target.iloc[:min_length]

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Chia dữ liệu huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Tuning mô hình
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, refit=True, verbose=0)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Dự đoán xu hướng thời gian thực
    latest_features = features_scaled[-1].reshape(1, -1)
    prediction_prob = grid.predict_proba(latest_features)[0]
    trend = 1 if prediction_prob[1] >= 0.55 else 0 if prediction_prob[1] <= 0.45 else -1

    return trend, accuracy * 100, f1 * 100


# Hàm trả về kết quả xu hướng cuối cùng
def get_final_trend_USDCAD():
    # Phân tích xu hướng cho hai khung thời gian
    trend_h1, accuracy_h1, f1_h1 = analyze_trend("USDCAD", "1h", 1500)
    trend_h4, accuracy_h4, f1_h4 = analyze_trend("USDCAD", "4h", 1500)

    # Tính xác suất kết hợp
    combined_acc = combined_probability(accuracy_h1 / 100, accuracy_h4 / 100)

    # Kiểm tra các điều kiện để quyết định kết quả
    if (trend_h1 == 1 and trend_h4 == 1 and combined_acc >= 0.88) or \
       (trend_h1 == 1 and accuracy_h1 > 71 and f1_h1 > 71) or \
       (trend_h4 == 1 and accuracy_h4 > 69 and f1_h4 > 70):
        return "Xu hướng tăng"
        
    elif (trend_h1 == 0 and trend_h4 == 0 and combined_acc >= 0.88) or \
         (trend_h1 == 0 and accuracy_h1 > 71 and f1_h1 > 71) or \
         (trend_h4 == 0 and accuracy_h4 > 69 and f1_h4 > 70):
        return "Xu hướng giảm"
        
    # Nếu một trong các khung thời gian có xu hướng không rõ ràng
    elif trend_h1 == -1 or trend_h4 == -1:
        return "Xu hướng không rõ ràng"
    
    else:
        return "Xu hướng không rõ ràng"
