"""
Module: feature_engineering.py
Task C6: Kỹ nghệ đặc trưng (Feature Engineering, 5 & 6).
- Trích xuất đặc trưng thời gian (Giờ bay, Ngày trong tuần, Tháng, Cuối tuần).
- Mã hóa chu kỳ lượng giác (Sine/Cosine của Hour và DayOfWeek).
- Tạo biến độ trễ chuỗi thời gian & mật độ sân bay:
  + airport_delay_lag_1h: Mức trễ bình quân của sân bay ở giờ liền trước.
  + airport_delay_rolling_3h: Trung bình trượt mức trễ sân bay 3 giờ trước đó.
- Mã hóa One-Hot cho Hãng hàng không, Loại máy bay, Sân bay xuất phát/đích.
- Phân cụm K-Means cho điều kiện vận hành sân bay (Khung giờ & Lưu lượng bay - không dùng nhãn mục tiêu).
- Mô hình hóa đường cong giá vé máy bay theo ngày đặt trước (Dynamic Pricing Curve).
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def build_features(cleaned_path: str = "data/processed/cleaned_flight_data.csv",
                   output_path: str = "data/processed/featured_flight_dataset.csv") -> pd.DataFrame:
    print("[Task C6] Bắt đầu quá trình kỹ nghệ đặc trưng...")
    df = pd.read_csv(cleaned_path)
    df['dep_time'] = pd.to_datetime(df['dep_time'])
    df = df.sort_values(by='dep_time').reset_index(drop=True)

    # 1. Trích xuất thuộc tính lịch biểu
    df['hour'] = df['dep_time'].dt.hour
    df['dayofweek'] = df['dep_time'].dt.dayofweek # 0: Thứ 2, 6: Chủ Nhật
    df['month'] = df['dep_time'].dt.month
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in [4, 6] else 0) # Thứ 6 & CN là cao điểm bay
    df['is_peak_hours'] = df['hour'].apply(lambda x: 1 if 16 <= x <= 20 else (1 if 7 <= x <= 9 else 0))

    # 2. Mã hóa chu kỳ (Cyclical Encoding) - Giúp mô hình bảo toàn tính khép kín của vòng quay 24h
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    # 3. Kỹ nghệ đặc trưng chuỗi thời gian (Time-Series Congestion Lags)
    # Tính mức trễ trung bình của toàn mạng lưới bay tại giờ t-1 và trung bình trượt 3 giờ (tránh lookahead)
    hourly_avg_delay = df.groupby(df['dep_time'].dt.floor('h'))['delay_minutes'].mean().reset_index()
    hourly_avg_delay.columns = ['time_hour', 'network_hourly_delay']
    hourly_avg_delay['network_delay_lag_1h'] = hourly_avg_delay['network_hourly_delay'].shift(1)
    hourly_avg_delay['network_delay_rolling_3h'] = hourly_avg_delay['network_hourly_delay'].shift(1).rolling(3).mean()

    df['time_hour'] = df['dep_time'].dt.floor('h')
    df = pd.merge(df, hourly_avg_delay[['time_hour', 'network_delay_lag_1h', 'network_delay_rolling_3h']], on='time_hour', how='left')
    df.drop(columns=['time_hour'], inplace=True, errors='ignore')
    df['network_delay_lag_1h'] = df['network_delay_lag_1h'].fillna(df['delay_minutes'].median())
    df['network_delay_rolling_3h'] = df['network_delay_rolling_3h'].fillna(df['delay_minutes'].median())

    # 4. Đặc trưng thời tiết nhị phân & rủi ro cao
    df['is_rain'] = (df['origin_rain_mm'] > 0.1).astype(int)
    df['is_high_wind'] = (df['origin_gusts_kmh'] > 25.0).astype(int)
    df['is_low_visibility'] = (df['origin_visibility_km'] < 5.0).astype(int)

    # 5. Phân cụm K-Means cho điều kiện khung giờ & tải khách
    # KHÔNG dùng nhãn mục tiêu delay_minutes để tránh rò rỉ dữ liệu (Target Leakage)
    cluster_features = df[['hour', 'passenger_count']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['flight_load_cluster'] = kmeans.fit_predict(cluster_features)

    # 6. Mã hóa One-Hot cho Hãng hàng không, Sân bay và Loại tàu bay
    airline_dummies = pd.get_dummies(df['airline'], prefix='airline', drop_first=True, dtype=int)
    origin_dummies = pd.get_dummies(df['origin'], prefix='origin', drop_first=True, dtype=int)
    dest_dummies = pd.get_dummies(df['destination'], prefix='dest', drop_first=True, dtype=int)
    aircraft_dummies = pd.get_dummies(df['aircraft'], prefix='aircraft', drop_first=True, dtype=int)

    df = pd.concat([df, airline_dummies, origin_dummies, dest_dummies, aircraft_dummies], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Task C6 - Hoàn thành] Đã tạo thành công {df.shape[1]} đặc trưng cho {len(df)} chuyến bay vào: {output_path}")
    return df

def generate_airfare_pricing_curve_dataset(output_path: str = "data/processed/airfare_pricing_dataset.csv",
                                           seed: int = 42) -> pd.DataFrame:
    """
    Sinh dữ liệu mô hình hóa đường cong giá vé máy bay theo chu kỳ đặt trước (Days Before Departure).
    Quy luật Dynamic Pricing:
    - 60 - 35 ngày trước bay: Giá ổn định cơ sở (~1,200,000 - 1,400,000 VNĐ).
    - 15 - 28 ngày trước bay: VÙNG ĐÁY GIÁ VÉ (Vé khuyến mãi hãng kích cầu: ~950,000 - 1,150,000 VNĐ).
    - 7 - 14 ngày trước bay: Bắt đầu tăng dần (~1,450,000 - 1,800,000 VNĐ).
    - 0 - 6 ngày trước bay: GIÁ ĐỈNH (Vé giờ chót cho khách công vụ: ~2,200,000 - 3,500,000 VNĐ).
    """
    np.random.seed(seed)
    records = []
    routes = [('HAN', 'SGN', 1450000), ('HAN', 'DAD', 980000), ('SGN', 'DAD', 950000), ('SGN', 'PQC', 780000)]
    airlines = ['Vietnam Airlines', 'Vietjet Air', 'Bamboo Airways']
    
    for origin, dest, base_fare in routes:
        for airline in airlines:
            airline_mult = 1.35 if airline == 'Vietnam Airlines' else (1.15 if airline == 'Bamboo Airways' else 0.90)
            for days_ahead in range(60, -1, -1):
                # Đường cong phi tuyến U-shape lệch
                if days_ahead >= 30:
                    time_factor = 1.0 + (days_ahead - 30) * 0.003
                elif 14 <= days_ahead < 30:
                    time_factor = 0.82 + (days_ahead - 14) * 0.008 # Đáy giá vé
                elif 7 <= days_ahead < 14:
                    time_factor = 1.05 + (14 - days_ahead) * 0.04
                else: # 0 - 6 ngày
                    time_factor = 1.45 + (7 - days_ahead) * 0.14
                    
                noise = np.random.normal(0, 0.04)
                final_fare = round(base_fare * airline_mult * (time_factor + noise) / 1000) * 1000
                
                records.append({
                    'origin': origin,
                    'destination': dest,
                    'airline': airline,
                    'days_before_departure': days_ahead,
                    'fare_vnd': int(final_fare)
                })
                
    df_pricing = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pricing.to_csv(output_path, index=False)
    print(f"[Airfare Pricing] Đã tạo {len(df_pricing)} điểm dữ liệu đường cong giá vé vào: {output_path}")
    return df_pricing

if __name__ == "__main__":
    build_features()
    generate_airfare_pricing_curve_dataset()
