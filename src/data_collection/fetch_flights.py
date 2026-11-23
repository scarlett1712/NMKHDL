"""
Module: fetch_flights.py
Task A1: Thu thập và quản lý luồng dữ liệu chuyến bay thực tế (NYC Flights Benchmark).
- Nguồn dữ liệu: 100% chuyến bay thương mại thật từ các siêu sân bay:
  + JFK (John F. Kennedy International Airport)
  + EWR (Newark Liberty International Airport)
  + LGA (LaGuardia Airport)
- Hãng hàng không: United Airlines (UA), American Airlines (AA), Delta (DL), JetBlue (B6), Southwest (WN)...
- Dữ liệu nguyên bản, không chứa bất kỳ luật sinh hay độ trễ hardcode nào.
"""

import os
import pandas as pd
import numpy as np

def load_real_flights_stream(raw_path: str = "data/raw/nyc_flights_raw.csv",
                             output_path: str = "data/raw/flights_stream.csv",
                             n_samples: int = 50000,
                             seed: int = 42) -> pd.DataFrame:
    print(f"[Task A1] Đang nạp và chuẩn hóa dữ liệu chuyến bay thực tế từ: {raw_path}...")
    df = pd.read_csv(raw_path)

    # Loại bỏ các chuyến bay bị hủy (không có thông tin giờ cất cánh / trễ)
    df = df.dropna(subset=['dep_delay', 'hour', 'month', 'day']).copy()

    # Tạo timestamp chuẩn hóa datetime
    df['hour'] = df['hour'].astype(int)
    df['minute'] = df['minute'].fillna(0).astype(int)
    df['dep_time_str'] = df.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d} {int(r['hour']):02d}:{int(r['minute']):02d}:00", axis=1)
    df['dep_timestamp'] = pd.to_datetime(df['dep_time_str'], errors='coerce')
    df = df.dropna(subset=['dep_timestamp']).sort_values(by='dep_timestamp').reset_index(drop=True)

    # Lấy mẫu đại diện 50,000 chuyến bay phân bố đều theo dòng thời gian
    if len(df) > n_samples:
        indices = np.linspace(0, len(df) - 1, n_samples, dtype=int)
        df_sample = df.iloc[indices].copy().reset_index(drop=True)
    else:
        df_sample = df.copy()

    # Chuẩn hóa tên trường
    df_sample['flight_id'] = df_sample.apply(lambda r: f"{r['carrier']}{int(r['flight'])}_{pd.to_datetime(r['dep_timestamp']).strftime('%Y%m%d')}_{r.name+1:05d}", axis=1)
    df_sample['airline'] = df_sample['carrier']
    df_sample['delay_minutes'] = df_sample['dep_delay'].astype(float)
    df_sample['is_delayed_30'] = (df_sample['delay_minutes'] >= 30).astype(int)
    df_sample['delay_status'] = df_sample['delay_minutes'].apply(
        lambda m: 'Đúng giờ' if m < 15 else ('Trễ vừa' if m < 45 else 'Trễ nghiêm trọng')
    )

    selected_cols = [
        'flight_id', 'airline', 'tailnum', 'origin', 'dest',
        'dep_timestamp', 'year', 'month', 'day', 'hour', 'minute',
        'air_time', 'distance', 'delay_minutes', 'delay_status', 'is_delayed_30'
    ]
    df_clean = df_sample[selected_cols].copy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"[Task A1 - Hoàn tất] Đã lưu {len(df_clean)} chuyến bay THẬT vào: {output_path}")
    return df_clean

if __name__ == "__main__":
    load_real_flights_stream()
