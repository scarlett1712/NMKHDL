"""
Module: fetch_flights.py
Task A1: Thu thập và quản lý luồng dữ liệu chuyến bay nội địa Việt Nam.
Các sân bay chính:
- HAN (Nội Bài - Hà Nội)
- SGN (Tân Sơn Nhất - TP.HCM)
- DAD (Đà Nẵng)
- CXR (Cam Ranh - Nha Trang)
- PQC (Phú Quốc)
Các hãng hàng không: Vietnam Airlines (VN), Vietjet Air (VJ), Bamboo Airways (QH), Vietravel Airlines (VU).
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

AIRPORTS = {
    'HAN': {'name': 'Nội Bài (Hà Nội)', 'lat': 21.2212, 'lon': 105.8072, 'capacity': 90},
    'SGN': {'name': 'Tân Sơn Nhất (TP.HCM)', 'lat': 10.8188, 'lon': 106.6518, 'capacity': 100},
    'DAD': {'name': 'Đà Nẵng', 'lat': 16.0439, 'lon': 108.1994, 'capacity': 60},
    'CXR': {'name': 'Cam Ranh (Nha Trang)', 'lat': 11.9981, 'lon': 109.2194, 'capacity': 45},
    'PQC': {'name': 'Phú Quốc', 'lat': 10.1698, 'lon': 103.9931, 'capacity': 40}
}

AIRLINES = {
    'Vietnam Airlines': {'code': 'VN', 'base_otp': 0.88, 'fleet': ['A321', 'A350', 'B787']},
    'Vietjet Air': {'code': 'VJ', 'base_otp': 0.74, 'fleet': ['A320', 'A321', 'A330']},
    'Bamboo Airways': {'code': 'QH', 'base_otp': 0.90, 'fleet': ['A320', 'A321', 'E190']},
    'Vietravel Airlines': {'code': 'VU', 'base_otp': 0.82, 'fleet': ['A321']}
}

ROUTES = [
    ('HAN', 'SGN', 115), # 115 phút bay
    ('SGN', 'HAN', 120),
    ('HAN', 'DAD', 80),
    ('DAD', 'HAN', 80),
    ('SGN', 'DAD', 85),
    ('DAD', 'SGN', 85),
    ('HAN', 'CXR', 110),
    ('CXR', 'HAN', 110),
    ('SGN', 'PQC', 55),
    ('PQC', 'SGN', 55)
]

def generate_vietnam_flights_stream(output_path: str = "data/raw/vietnam_flights_stream.csv",
                                    n_records: int = 42000,
                                    seed: int = 42) -> pd.DataFrame:
    """
    Tạo lập luồng dữ liệu chuyến bay nội địa Việt Nam chuẩn hóa, phản ánh đúng đặc thù thực tế:
    - Chuyến sáng sớm (6h - 8h) tỷ lệ đúng giờ cao (>92%).
    - Chuyến chiều tối (16h - 20h) bị trễ dây chuyền do nghẽn đường băng tại Tân Sơn Nhất.
    - Cuối tuần (Thứ 6, Chủ nhật) lượng khách đột biến làm tăng trễ.
    """
    np.random.seed(seed)
    print(f"[Task A1] Đang sinh luồng dữ liệu chuyến bay nội địa ({n_records} bản ghi)...")
    
    start_date = datetime(2025, 1, 1, 6, 0)
    records = []
    
    airline_names = list(AIRLINES.keys())
    airline_probs = [0.42, 0.40, 0.14, 0.04] # Thị phần chuyến bay
    
    route_indices = list(range(len(ROUTES)))
    raw_weights = [0.34, 0.34, 0.08, 0.08, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01]
    route_weights = np.array(raw_weights) / sum(raw_weights)
    
    current_time = start_date
    for i in range(n_records):
        # Tịnh tiến thời gian theo chuỗi liên tục
        step_minutes = np.random.choice([10, 15, 20, 25, 30])
        current_time += timedelta(minutes=int(step_minutes))
        
        # Nếu vào ban đêm (0h - 5h), nhảy sang 6h sáng hôm sau
        if current_time.hour < 6:
            current_time = current_time.replace(hour=6, minute=np.random.randint(0, 30))
            
        airline = np.random.choice(airline_names, p=airline_probs)
        airline_info = AIRLINES[airline]
        aircraft = np.random.choice(airline_info['fleet'])
        
        route_idx = np.random.choice(route_indices, p=route_weights)
        origin, dest, sched_duration = ROUTES[route_idx]
        
        flight_num = f"{airline_info['code']}{np.random.randint(100, 999)}"
        
        hour = current_time.hour
        dow = current_time.weekday()
        month = current_time.month
        
        # 1. Hiệu ứng khung giờ: Chiều tối trễ dây chuyền
        hour_delay_factor = 0.0
        if 6 <= hour <= 8:
            hour_delay_factor = -12.0 # Sáng sớm rất thông thoáng
        elif 9 <= hour <= 12:
            hour_delay_factor = 2.0
        elif 13 <= hour <= 16:
            hour_delay_factor = 10.0
        elif 17 <= hour <= 21:
            hour_delay_factor = 26.0 # Giờ cao điểm nghẽn Tân Sơn Nhất / Nội Bài
        else:
            hour_delay_factor = 15.0
            
        # 2. Hiệu ứng cuối tuần (Thứ 6 = 4, Chủ Nhật = 6)
        weekend_factor = 8.0 if dow in [4, 6] else 0.0
        
        # 3. Hiệu ứng hãng hàng không
        airline_delay_factor = 0.0
        if airline == 'Vietjet Air':
            airline_delay_factor = 14.0 # Do quay đầu máy bay dày đặc
        elif airline == 'Vietnam Airlines':
            airline_delay_factor = 2.0
        elif airline == 'Bamboo Airways':
            airline_delay_factor = 1.0
        else:
            airline_delay_factor = 8.0
            
        # 4. Hiệu ứng sân bay đích (Tân Sơn Nhất tắc nghẽn nhất)
        dest_factor = 12.0 if dest == 'SGN' else (8.0 if dest == 'HAN' else 0.0)
        
        # Phân phối trễ: Đa số đúng giờ/trễ nhẹ, một số ít trễ nặng (Lognormal/Exponential)
        base_noise = np.random.exponential(scale=14.0) - 8.0
        raw_delay = hour_delay_factor + weekend_factor + airline_delay_factor + dest_factor + base_noise
        
        # Làm tròn và chặn sàn (bay sớm tối đa 15 phút)
        delay_minutes = int(np.clip(round(raw_delay), -15, 240))
        
        # Trạng thái trễ
        if delay_minutes < 15:
            delay_status = 'Đúng giờ'
            is_delayed_30 = 0
        elif delay_minutes < 45:
            delay_status = 'Trễ vừa'
            is_delayed_30 = 1 if delay_minutes >= 30 else 0
        else:
            delay_status = 'Trễ nghiêm trọng'
            is_delayed_30 = 1
            
        passenger_load = int(np.random.normal(loc=175, scale=25))
        passenger_load = int(np.clip(passenger_load, 80, 220))
        
        records.append({
            'flight_id': flight_num,
            'airline': airline,
            'aircraft': aircraft,
            'origin': origin,
            'destination': dest,
            'scheduled_departure': current_time.strftime("%Y-%m-%d %H:%M:%S"),
            'scheduled_duration': sched_duration,
            'passenger_count': passenger_load,
            'delay_minutes': delay_minutes,
            'delay_status': delay_status,
            'is_delayed_30': is_delayed_30
        })
        
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Task A1 - Hoàn tất] Đã lưu {len(df)} chuyến bay vào: {output_path}")
    return df

if __name__ == "__main__":
    generate_vietnam_flights_stream()
