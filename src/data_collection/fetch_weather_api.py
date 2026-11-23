"""
Module: fetch_weather_api.py
Task A2: Quản lý và chuẩn hóa dữ liệu quan trắc khí tượng thật từ trạm thời tiết sân bay (NOAA).
- Sân bay: EWR, JFK, LGA
- Các chỉ số: Nhiệt độ (°C), lượng mưa (mm), tốc độ gió (km/h), gió giật (km/h), tầm nhìn (km), độ ẩm (%).
- Dữ liệu quan trắc vật lý thực tế 100%, không sinh ngẫu nhiên.

Nguồn: bảng `weather` của gói nycflights13 (Wickham, CRAN) - quan trắc ASOS theo giờ của 3 trạm sân bay
năm 2013, tải bản CSV từ kho Rdatasets. Bản cũ trong repo chỉ có EWR (JFK 9 dòng, LGA 1 dòng) và là phiên bản
nycflights13 đời đầu (wind_gust luôn = wind_speed x 1.15), nên được thay bằng bản đầy đủ hiện hành.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests

WEATHER_SOURCE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/weather.csv"
HTTP_HEADERS = {'User-Agent': 'NMKHDL-coursework-pipeline/1.0 (educational project)'}

AIRPORT_COORDS = {
    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'John F. Kennedy (New York)'},
    'EWR': {'lat': 40.6895, 'lon': -74.1745, 'name': 'Newark Liberty (New Jersey)'},
    'LGA': {'lat': 40.7769, 'lon': -73.8740, 'name': 'LaGuardia (New York)'}
}

def download_weather_raw(output_path: str = "data/raw/nyc_weather_raw.csv",
                         url: str = WEATHER_SOURCE_URL) -> pd.DataFrame:
    """Tải bảng thời tiết nycflights13 đầy đủ 3 sân bay qua HTTP và lưu thành file thô."""
    print(f"[Task A2] Đang tải dữ liệu khí tượng từ: {url}")
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=60)
    resp.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    df = df.drop(columns=['rownames'], errors='ignore')

    counts = df['origin'].value_counts()
    missing = [a for a in AIRPORT_COORDS if counts.get(a, 0) < 8000]
    if missing:
        raise ValueError(f"Dữ liệu thời tiết tải về thiếu sân bay: {missing} (số dòng: {counts.to_dict()})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Task A2] Đã lưu {len(df)} quan trắc thô ({counts.to_dict()}) vào: {output_path}")
    return df

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
    # Kiểm tra tính hợp lệ vật lý: nycflights13 có lỗi cảm biến đã biết (wind_speed = 1048 mph ~ 1687 km/h).
    # Gió > 200 km/h (mạnh hơn bão cấp 4) không thể xảy ra ở NYC năm 2013 -> coi là khuyết, để cleaner điền.
    invalid_wind = df['wind_speed_kmh'] > 200
    if invalid_wind.any():
        print(f"[Task A2] Phát hiện {int(invalid_wind.sum())} giá trị gió phi vật lý (> 200 km/h) -> đặt NaN")
    df.loc[invalid_wind, ['wind_speed_kmh', 'wind_gusts_kmh']] = np.nan
    # Trạm ASOS chỉ ghi gió giật khi có giật; trống nghĩa là không có gió giật -> gió giật = tốc độ gió nền
    df['wind_gusts_kmh'] = df['wind_gusts_kmh'].fillna(df['wind_speed_kmh']).round(1)
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
    # Tải lại dữ liệu thô khi chưa có, hoặc khi chạy với cờ --refresh
    if "--refresh" in sys.argv or not os.path.exists("data/raw/nyc_weather_raw.csv"):
        download_weather_raw()
    load_real_weather_stream()
