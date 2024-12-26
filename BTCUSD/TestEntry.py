# Entry_Super.py
import random

def get_final_trend(client):
    # Danh sách các xu hướng có thể xảy ra
    trends = ["Xu hướng giảm","Xu hướng tăng"]
    
    # Lựa chọn ngẫu nhiên một xu hướng để trả về
    final_trend = random.choice(trends)
    
    print(f"Giả lập xu hướng: {final_trend}")
    return final_trend
