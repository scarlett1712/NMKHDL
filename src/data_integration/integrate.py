"""
Module: integrate.py
Task A3: Tích hợp dữ liệu đa nguồn (Data Integration).
Tuân thủ kiến trúc Hệ tích hợp ảo (Virtual Data Integration):
- Thiết kế Lược đồ trung gian (Mediated Schema)
- Xây dựng Wrappers cho nguồn Chuyến bay thật và nguồn Trạm khí tượng NOAA thật
- Ghép nối dữ liệu (Join) theo khóa đa thuộc tính: (origin, year, month, day, hour)
- Xử lý trùng lặp và xung đột bản ghi (Deduplication)
"""

import os
import pandas as pd

class FlightWrapper:
    """Wrapper chuyển đổi nguồn dữ liệu chuyến bay thực tế sang Lược đồ trung gian."""
    @staticmethod
    def wrap(df_flights: pd.DataFrame) -> pd.DataFrame:
        df = df_flights.copy()
        df['dep_timestamp'] = pd.to_datetime(df['dep_timestamp'])
        return df

class AirportWeatherWrapper:
    """Wrapper chuyển đổi nguồn dữ liệu trạm khí tượng NOAA sang Lược đồ trung gian."""
    @staticmethod
    def wrap(df_weather: pd.DataFrame) -> pd.DataFrame:
        df = df_weather.copy()
        selected_cols = [
            'origin', 'year', 'month', 'day', 'hour',
            'temp_celsius', 'precipitation_mm', 'wind_speed_kmh', 'wind_gusts_kmh',
            'visibility_km', 'humid', 'pressure', 'weather_condition'
        ]
        return df[selected_cols]

class DataIntegrationEngine:
    def __init__(self, flights_path: str = "data/raw/flights_stream.csv",
                 weather_path: str = "data/raw/weather_stream.csv"):
        self.flights_path = flights_path
        self.weather_path = weather_path

    def integrate(self, output_path: str = "data/processed/integrated_flight_weather.csv") -> pd.DataFrame:
        print("[Task A3] Đang nạp và hợp nhất 2 nguồn dữ liệu THẬT qua Wrappers...")
        df_raw_flights = pd.read_csv(self.flights_path)
        df_raw_weather = pd.read_csv(self.weather_path)

        # 1. Chuyển đổi qua Wrappers
        flights_wrapped = FlightWrapper.wrap(df_raw_flights)
        weather_wrapped = AirportWeatherWrapper.wrap(df_raw_weather)

        # 2. Xử lý trùng lặp (Deduplication)
        flights_dedup = flights_wrapped.drop_duplicates(subset=['flight_id'])
        weather_dedup = weather_wrapped.drop_duplicates(subset=['origin', 'year', 'month', 'day', 'hour'])

        # 3. Hợp nhất theo Mediated Schema: (origin, year, month, day, hour)
        integrated_df = pd.merge(
            flights_dedup,
            weather_dedup,
            on=['origin', 'year', 'month', 'day', 'hour'],
            how='left'
        )

        # Đổi tên các trường thời tiết sang tiền tố origin_
        integrated_df.rename(columns={
            'temp_celsius': 'origin_temp_c',
            'precipitation_mm': 'origin_rain_mm',
            'wind_speed_kmh': 'origin_wind_kmh',
            'wind_gusts_kmh': 'origin_gusts_kmh',
            'visibility_km': 'origin_visibility_km',
            'weather_condition': 'origin_weather'
        }, inplace=True)

        # Sắp xếp theo dòng thời gian cất cánh
        integrated_df = integrated_df.sort_values(by='dep_timestamp').reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        integrated_df.to_csv(output_path, index=False)
        print(f"[Task A3 - Hoàn thành] Đã hợp nhất thành công {len(integrated_df)} chuyến bay thật vào: {output_path}")
        print("Cấu trúc Mediated Schema:", list(integrated_df.columns))
        return integrated_df

if __name__ == "__main__":
    engine = DataIntegrationEngine()
    engine.integrate()
