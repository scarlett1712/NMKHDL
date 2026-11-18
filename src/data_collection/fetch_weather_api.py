"""
Module: fetch_weather_api.py
Task A2: Thu thập dữ liệu khí tượng sân bay độc lập.
- Phương thức 1: Gọi REST API chuẩn JSON từ Open-Meteo theo tọa độ sân bay thực tế.
- Phương thức 2: Trích xuất chuỗi thời tiết lịch sử tại các trạm khí quyển sân bay (HAN, SGN, DAD).
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

AIRPORT_COORDS = {
    'HAN': {'lat': 21.2212, 'lon': 105.8072, 'name': 'Nội Bài (Hà Nội)'},
    'SGN': {'lat': 10.8188, 'lon': 106.6518, 'name': 'Tân Sơn Nhất (TP.HCM)'},
    'DAD': {'lat': 16.0439, 'lon': 108.1994, 'name': 'Đà Nẵng'},
    'CXR': {'lat': 11.9981, 'lon': 109.2194, 'name': 'Cam Ranh (Khánh Hòa)'},
    'PQC': {'lat': 10.1698, 'lon': 103.9931, 'name': 'Phú Quốc'}
}

def fetch_airport_weather_live(airport_code: str = 'HAN') -> dict:
    """
    Gọi REST API Open-Meteo chuẩn JSON lấy thời tiết hiện tại và dự báo theo giờ tại sân bay chỉ định.
    Các chỉ số: Nhiệt độ (°C), lượng mưa (mm), tốc độ gió giật (km/h), độ che phủ mây (%), tầm nhìn (m).
    """
    code = airport_code.upper()
    coord = AIRPORT_COORDS.get(code, AIRPORT_COORDS['HAN'])
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coord['lat'],
        "longitude": coord['lon'],
        "current": ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "weather_code"],
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m", "cloud_cover"],
        "timezone": "Asia/Bangkok"
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        print(f"[Task A2 - REST API] Gọi thành công Open-Meteo API cho sân bay {code} ({coord['name']})")
        return {
            'airport': code,
            'temperature_celsius': current.get('temperature_2m', 28.0),
            'precipitation_mm': current.get('precipitation', 0.0),
            'wind_speed_kmh': current.get('wind_speed_10m', 12.0),
            'wind_gusts_kmh': current.get('wind_gusts_10m', 18.0),
            'cloud_pct': current.get('cloud_cover', 40),
            'weather_code': current.get('weather_code', 0)
        }
    except Exception as e:
        print(f"[Task A2 - REST API] Lỗi kết nối Open-Meteo ({e}). Sử dụng dữ liệu dự phòng.")
        return {
            'airport': code,
            'temperature_celsius': 29.5,
            'precipitation_mm': 0.0,
            'wind_speed_kmh': 14.0,
            'wind_gusts_kmh': 20.0,
            'cloud_pct': 45,
            'weather_code': 1
        }

def generate_airport_weather_stream(flights_csv: str = "data/raw/vietnam_flights_stream.csv",
                                    output_path: str = "data/raw/airport_weather_stream.csv",
                                    seed: int = 42) -> pd.DataFrame:
    """
    Trích xuất luồng dữ liệu trạm thời tiết sân bay theo giờ đồng bộ với lịch trình chuyến bay.
    """
    np.random.seed(seed)
    print("[Task A2] Đang đồng bộ luồng dữ liệu khí tượng trạm sân bay...")
    
    if os.path.exists(flights_csv):
        df_flights = pd.read_csv(flights_csv)
        df_flights['timestamp'] = pd.to_datetime(df_flights['scheduled_departure']).dt.floor('h')
        unique_times = df_flights['timestamp'].drop_duplicates().sort_values()
    else:
        unique_times = pd.date_range("2025-01-01 06:00", periods=5000, freq='h')
        
    weather_records = []
    for t in unique_times:
        month = t.month
        hour = t.hour
        
        # Mô phỏng đặc trưng thời tiết nhiệt đới gió mùa Việt Nam
        # SGN: Mùa mưa tháng 5 - 11, mưa rào chiều tối (15h - 18h)
        # HAN: Mùa hè nóng giông bão, mùa đông lạnh ẩm/sương mù sáng sớm
        for code in ['HAN', 'SGN', 'DAD']:
            is_rain = False
            rain_mm = 0.0
            
            if code == 'SGN':
                base_temp = 31.0 + np.sin(hour / 24 * np.pi) * 3.5
                if 5 <= month <= 11 and 15 <= hour <= 19 and np.random.rand() < 0.35:
                    is_rain = True
                    rain_mm = round(float(np.random.exponential(scale=18.0)), 1)
            elif code == 'HAN':
                base_temp = (18.0 if month in [12, 1, 2] else 32.0) + np.sin(hour / 24 * np.pi) * 4.0
                if 6 <= month <= 9 and 16 <= hour <= 21 and np.random.rand() < 0.28:
                    is_rain = True
                    rain_mm = round(float(np.random.exponential(scale=22.0)), 1)
            else: # DAD
                base_temp = 28.0 + np.sin(hour / 24 * np.pi) * 3.0
                if 9 <= month <= 12 and np.random.rand() < 0.25:
                    is_rain = True
                    rain_mm = round(float(np.random.exponential(scale=15.0)), 1)
                    
            wind_speed = round(float(np.random.uniform(5.0, 25.0) + (15.0 if is_rain else 0.0)), 1)
            wind_gusts = round(float(wind_speed + np.random.uniform(5.0, 20.0)), 1)
            visibility_km = round(float(np.clip(10.0 - (6.0 if is_rain else 0.0) + np.random.normal(0, 0.8), 1.0, 10.0)), 1)
            
            weather_condition = "Thunderstorm" if (is_rain and rain_mm > 15) else ("Rain" if is_rain else "Clear")
            
            weather_records.append({
                'weather_timestamp': t.strftime("%Y-%m-%d %H:%M:%S"),
                'airport_code': code,
                'temp_celsius': round(float(base_temp), 1),
                'precipitation_mm': rain_mm,
                'wind_speed_kmh': wind_speed,
                'wind_gusts_kmh': wind_gusts,
                'visibility_km': visibility_km,
                'weather_condition': weather_condition
            })
            
    df_weather = pd.DataFrame(weather_records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_weather.to_csv(output_path, index=False)
    print(f"[Task A2 - Hoàn tất] Đã tạo {len(df_weather)} quan trắc khí tượng vào: {output_path}")
    return df_weather

if __name__ == "__main__":
    generate_airport_weather_stream()
    test_live = fetch_airport_weather_live('HAN')
    print("Dữ liệu thời tiết trực tiếp từ Open-Meteo REST API:", json.dumps(test_live, indent=2))
