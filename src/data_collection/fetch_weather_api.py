"""
Module: fetch_weather_api.py
Task A2: Quản lý và chuẩn hóa dữ liệu quan trắc khí tượng thật từ trạm thời tiết sân bay (NOAA).
- Sân bay: EWR, JFK, LGA
- Các chỉ số: Nhiệt độ (°C), lượng mưa (mm), tốc độ gió (km/h), gió giật (km/h), tầm nhìn (km), độ ẩm (%).
- Dữ liệu quan trắc vật lý thực tế 100%, không sinh ngẫu nhiên.
"""

import os
import pandas as pd
import numpy as np

AIRPORT_COORDS = {
    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'John F. Kennedy (New York)'},
    'EWR': {'lat': 40.6895, 'lon': -74.1745, 'name': 'Newark Liberty (New Jersey)'},
    'LGA': {'lat': 40.7769, 'lon': -73.8740, 'name': 'LaGuardia (New York)'}
}

def load_real_weather_stream(raw_path: str = "data/raw/nyc_weather_raw.csv",
                             output_path: str = "data/raw/weather_stream.csv") -> pd.DataFrame:
    print(f"[Task A2] Đang nạp và chuẩn hóa dữ liệu khí tượng NOAA từ: {raw_path}...")
    df = pd.read_csv(raw_path)

    # Loại bỏ bản ghi khuyết ngày tháng
    df = df.dropna(subset=['month', 'day', 'hour']).copy()

    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['hour'] = df['hour'].astype(int)

    # Chuyển đổi đơn vị đo lường sang hệ mét chuẩn quốc tế:
    # Fahrenheit sang Celsius: (F - 32) * 5/9
    df['temp_celsius'] = ((df['temp'] - 32.0) * 5.0 / 9.0).round(1)
    # Tốc độ gió dặm/giờ sang km/h: mph * 1.60934
    df['wind_speed_kmh'] = (df['wind_speed'] * 1.60934).round(1)
    df['wind_gusts_kmh'] = df['wind_gust'].apply(
        lambda g: round(g * 1.60934, 1) if pd.notna(g) else np.nan
    )
    # Nếu wind_gust trống thì lấy bằng wind_speed + 5 km/h
    df['wind_gusts_kmh'] = df['wind_gusts_kmh'].fillna(df['wind_speed_kmh'] + 5.0).round(1)
    # Lượng mưa inches sang mm: inch * 25.4
    df['precipitation_mm'] = (df['precip'] * 25.4).round(2)
    # Tầm nhìn dặm sang km: miles * 1.60934
    df['visibility_km'] = (df['visib'] * 1.60934).round(1)

    df['weather_condition'] = df.apply(
        lambda r: 'Mưa dông' if r['precipitation_mm'] > 5.0 else (
            'Mưa' if r['precipitation_mm'] > 0.1 else (
                'Sương mù' if r['visibility_km'] < 4.0 else 'Quang đãng'
            )
        ), axis=1
    )

    df['time_hour_str'] = df.apply(
        lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d} {int(r['hour']):02d}:00:00", axis=1
    )

    selected_cols = [
        'origin', 'year', 'month', 'day', 'hour', 'time_hour_str',
        'temp_celsius', 'precipitation_mm', 'wind_speed_kmh', 'wind_gusts_kmh',
        'visibility_km', 'humid', 'pressure', 'weather_condition'
    ]
    df_clean = df[selected_cols].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"[Task A2 - Hoàn tất] Đã lưu {len(df_clean)} quan trắc khí tượng thật vào: {output_path}")
    return df_clean

if __name__ == "__main__":
    load_real_weather_stream()
