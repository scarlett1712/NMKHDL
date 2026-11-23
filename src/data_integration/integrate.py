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
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.splits import sort_chronologically

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

class AirportTrafficWrapper:
    """Wrapper cho nguồn Web Crawler: quy mô sân bay đến (hub tier, lượt khách lên máy bay)."""
    REFERENCE_YEAR = 2016  # Năm cũ nhất có trên Wikipedia, gần 2013 nhất

    @classmethod
    def attach(cls, df: pd.DataFrame, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            print(f"[Task A3] Cảnh báo: chưa có {path} (chạy web_crawler.py trước) - bỏ qua nguồn này.")
            return df
        traffic = pd.read_csv(path)
        traffic = traffic[['iata', 'hub_tier', f'enplanements_{cls.REFERENCE_YEAR}']].rename(columns={
            'iata': 'dest',
            'hub_tier': 'dest_hub_tier',
            f'enplanements_{cls.REFERENCE_YEAR}': 'dest_enplanements_m'
        })
        traffic['dest_enplanements_m'] = (traffic['dest_enplanements_m'] / 1e6).round(3)
        out = df.merge(traffic, on='dest', how='left')
        # Sân bay không có trong danh sách large/medium hub: hub_tier = 0, lượt khách = 0 (dưới ngưỡng medium hub)
        out['dest_hub_tier'] = out['dest_hub_tier'].fillna(0).astype(int)
        out['dest_enplanements_m'] = out['dest_enplanements_m'].fillna(0.0)
        print(f"[Task A3] Đã ghép dữ liệu Web Crawler: {(out['dest_hub_tier'] > 0).mean() * 100:.1f}% chuyến bay có sân bay đến là hub")
        return out

class DataIntegrationEngine:
    def __init__(self, flights_path: str = "data/raw/flights_stream.csv",
                 weather_path: str = "data/raw/weather_stream.csv",
                 airport_traffic_path: str = "data/raw/crawled_us_airport_traffic.csv"):
        self.flights_path = flights_path
        self.weather_path = weather_path
        self.airport_traffic_path = airport_traffic_path

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
        integrated_df = integrated_df.rename(columns={
            'temp_celsius': 'origin_temp_c',
            'precipitation_mm': 'origin_rain_mm',
            'wind_speed_kmh': 'origin_wind_kmh',
            'wind_gusts_kmh': 'origin_gusts_kmh',
            'visibility_km': 'origin_visibility_km',
            'weather_condition': 'origin_weather'
        })

        # 4. Hợp nhất nguồn thứ 3 - dữ liệu cào từ Web (Wikipedia/FAA) theo khóa sân bay đến (dest = IATA)
        integrated_df = AirportTrafficWrapper.attach(integrated_df, self.airport_traffic_path)

        # Sắp xếp theo dòng thời gian cất cánh (ổn định, dùng chung quy ước với các bước sau)
        integrated_df = sort_chronologically(integrated_df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        integrated_df.to_csv(output_path, index=False)
        print(f"[Task A3 - Hoàn thành] Đã hợp nhất thành công {len(integrated_df)} chuyến bay thật vào: {output_path}")
        print("Cấu trúc Mediated Schema:", list(integrated_df.columns))
        return integrated_df

if __name__ == "__main__":
    engine = DataIntegrationEngine()
    engine.integrate()
