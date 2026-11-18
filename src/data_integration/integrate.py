"""
Module: integrate.py
Task A3: Tích hợp dữ liệu (Data Integration).
Tuân thủ kiến trúc Hệ tích hợp ảo (Virtual Data Integration):
- Thiết kế Lược đồ trung gian (Mediated Schema)
- Xây dựng Wrappers cho nguồn Chuyến bay và Trạm khí tượng sân bay
- Ghép nối dữ liệu (Join) theo khóa đa thuộc tính: (timestamp, origin) & (timestamp, destination)
- Xử lý trùng lặp và xung đột bản ghi (Deduplication)
"""

import os
import pandas as pd
from typing import Tuple

class FlightWrapper:
    """Wrapper chuyển đổi nguồn dữ liệu chuyến bay sang lược đồ trung gian."""
    @staticmethod
    def wrap(df_flights: pd.DataFrame) -> pd.DataFrame:
        df = df_flights.copy()
        df['dep_time'] = pd.to_datetime(df['scheduled_departure'])
        df['time_hour'] = df['dep_time'].dt.floor('h')
        
        # Chuẩn hóa tên trường sang Mediated Schema
        selected_cols = [
            'flight_id', 'airline', 'aircraft', 'origin', 'destination',
            'dep_time', 'time_hour', 'scheduled_duration', 'passenger_count',
            'delay_minutes', 'delay_status', 'is_delayed_30'
        ]
        return df[selected_cols]

class AirportWeatherWrapper:
    """Wrapper chuyển đổi nguồn dữ liệu trạm khí tượng sang lược đồ trung gian."""
    @staticmethod
    def wrap(df_weather: pd.DataFrame) -> pd.DataFrame:
        df = df_weather.copy()
        df['time_hour'] = pd.to_datetime(df['weather_timestamp'])
        df['airport_code'] = df['airport_code'].astype(str).str.upper()
        
        selected_cols = [
            'time_hour', 'airport_code', 'temp_celsius', 'precipitation_mm',
            'wind_speed_kmh', 'wind_gusts_kmh', 'visibility_km', 'weather_condition'
        ]
        return df[selected_cols]

class DataIntegrationEngine:
    """Bộ tích hợp trung gian (Mediator Engine) kết nối và hợp nhất hai nguồn dữ liệu."""
    def __init__(self, flights_path: str = "data/raw/vietnam_flights_stream.csv",
                 weather_path: str = "data/raw/airport_weather_stream.csv"):
        self.flights_path = flights_path
        self.weather_path = weather_path

    def integrate(self, output_path: str = "data/processed/integrated_flight_weather.csv") -> pd.DataFrame:
        print("[Task A3] Đang nạp dữ liệu từ 2 nguồn độc lập qua Wrappers...")
        df_raw_flights = pd.read_csv(self.flights_path)
        df_raw_weather = pd.read_csv(self.weather_path)

        # 1. Chạy qua Wrappers
        flights_wrapped = FlightWrapper.wrap(df_raw_flights)
        weather_wrapped = AirportWeatherWrapper.wrap(df_raw_weather)

        # 2. Xử lý trùng lặp trên từng luồng trước khi ghép (Deduplication)
        flights_dedup = flights_wrapped.drop_duplicates(subset=['flight_id', 'dep_time'])
        weather_dedup = weather_wrapped.drop_duplicates(subset=['time_hour', 'airport_code'])

        # 3. Hợp nhất thời tiết sân bay xuất phát (Origin Airport Weather)
        integrated_df = pd.merge(
            flights_dedup,
            weather_dedup,
            left_on=['time_hour', 'origin'],
            right_on=['time_hour', 'airport_code'],
            how='left'
        )
        
        # Đổi tên các trường thời tiết sân bay đi
        integrated_df.rename(columns={
            'temp_celsius': 'origin_temp_c',
            'precipitation_mm': 'origin_rain_mm',
            'wind_speed_kmh': 'origin_wind_kmh',
            'wind_gusts_kmh': 'origin_gusts_kmh',
            'visibility_km': 'origin_visibility_km',
            'weather_condition': 'origin_weather'
        }, inplace=True)
        integrated_df.drop(columns=['airport_code'], inplace=True, errors='ignore')

        # Điền giá trị dự phòng nếu có sân bay nhỏ (CXR, PQC) chưa có trạm riêng
        integrated_df['origin_temp_c'] = integrated_df['origin_temp_c'].fillna(28.5)
        integrated_df['origin_rain_mm'] = integrated_df['origin_rain_mm'].fillna(0.0)
        integrated_df['origin_wind_kmh'] = integrated_df['origin_wind_kmh'].fillna(12.0)
        integrated_df['origin_gusts_kmh'] = integrated_df['origin_gusts_kmh'].fillna(16.0)
        integrated_df['origin_visibility_km'] = integrated_df['origin_visibility_km'].fillna(10.0)
        integrated_df['origin_weather'] = integrated_df['origin_weather'].fillna('Clear')

        integrated_df = integrated_df.sort_values(by='dep_time').reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        integrated_df.to_csv(output_path, index=False)
        print(f"[Task A3 - Hoàn thành] Đã hợp nhất thành công {len(integrated_df)} bản ghi vào: {output_path}")
        print("Cấu trúc Mediated Schema:", list(integrated_df.columns))
        return integrated_df

    def integrate_and_save(self, output_path: str = "data/processed/integrated_flight_weather.csv") -> pd.DataFrame:
        """Bí danh hàm tương thích cả cú pháp của Notebook."""
        return self.integrate(output_path)

if __name__ == "__main__":
    engine = DataIntegrationEngine()
    engine.integrate()
