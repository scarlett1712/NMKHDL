"""
Module: feature_engineering.py
Giai đoạn C: Kỹ nghệ đặc trưng (Feature Engineering) trên dữ liệu thực tế.
- Trích xuất thuộc tính lịch biểu: Khung giờ, Ngày trong tuần, Tháng, Ngày cao điểm bay (is_peak_day), Giờ cao điểm (is_peak_hours).
- Mã hóa chu kỳ lượng giác (Cyclical Sine/Cosine).
- Tính toán độ trễ chuỗi thời gian mạng bay theo từng sân bay:
  + airport_delay_lag_1h: Mức trễ bình quân của sân bay ở giờ liền trước (t-1).
  + airport_delay_rolling_3h: Trung bình trượt mức trễ sân bay 3 giờ trước đó.
- Phân cụm K-Means cho điều kiện vận hành sân bay (chuẩn hóa StandardScaler, k=3).
- Mã hóa One-Hot cho Hãng hàng không, Sân bay xuất phát và Top sân bay đến.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "models_saved"

def build_features(cleaned_path: str = "data/processed/cleaned_flight_data.csv",
                   output_path: str = "data/processed/featured_flight_dataset.csv") -> pd.DataFrame:
    print("[Task C] Bắt đầu quá trình kỹ nghệ đặc trưng trên dữ liệu THẬT...")
    df = pd.read_csv(cleaned_path)
    df['dep_timestamp'] = pd.to_datetime(df['dep_timestamp'])
    df = df.sort_values(by='dep_timestamp').reset_index(drop=True)

    # 1. Trích xuất thuộc tính lịch biểu
    df['hour'] = df['dep_timestamp'].dt.hour
    df['dayofweek'] = df['dep_timestamp'].dt.dayofweek # 0: Thứ 2, ..., 6: Chủ Nhật
    df['month'] = df['dep_timestamp'].dt.month

    # Ngày cao điểm trong tuần (Thứ 5, Thứ 6, Chủ Nhật là cao điểm bay thương mại)
    df['is_peak_day'] = df['dayofweek'].apply(lambda x: 1 if x in [3, 4, 6] else 0)
    # Khung giờ cao điểm (Chiều tối 16h-20h và sáng sớm 7h-9h)
    df['is_peak_hours'] = df['hour'].apply(lambda x: 1 if 16 <= x <= 20 else (1 if 7 <= x <= 9 else 0))

    # 2. Mã hóa chu kỳ lượng giác (Cyclical Encoding)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

    # 3. Kỹ nghệ đặc trưng trễ chuỗi thời gian cục bộ theo sân bay xuất phát (Airport Congestion Lags)
    # Tính mức trễ trung bình của sân bay xuất phát ở giờ t-1 và trung bình trượt 3 giờ
    df['time_hour'] = df['dep_timestamp'].dt.floor('h')
    airport_hourly = df.groupby(['origin', 'time_hour'])['delay_minutes'].mean().reset_index()
    airport_hourly.columns = ['origin', 'time_hour', 'airport_hourly_delay']
    airport_hourly = airport_hourly.sort_values(by=['origin', 'time_hour']).reset_index(drop=True)

    airport_hourly['network_delay_lag_1h'] = airport_hourly.groupby('origin')['airport_hourly_delay'].shift(1)
    airport_hourly['network_delay_rolling_3h'] = airport_hourly.groupby('origin')['airport_hourly_delay'].shift(1).rolling(3).mean()

    df = pd.merge(df, airport_hourly[['origin', 'time_hour', 'network_delay_lag_1h', 'network_delay_rolling_3h']], on=['origin', 'time_hour'], how='left')
    df.drop(columns=['time_hour'], inplace=True, errors='ignore')

    median_delay = float(df['delay_minutes'].median())
    df['network_delay_lag_1h'] = df['network_delay_lag_1h'].fillna(median_delay)
    df['network_delay_rolling_3h'] = df['network_delay_rolling_3h'].fillna(median_delay)

    # 4. Đặc trưng rủi ro thời tiết thực tế
    df['is_rain'] = (df['origin_rain_mm'] > 0.1).astype(int)
    df['is_high_wind'] = (df['origin_gusts_kmh'] > 25.0).astype(int)
    df['is_low_visibility'] = (df['origin_visibility_km'] < 8.0).astype(int)

    # 5. Phân cụm K-Means cho điều kiện vận hành (K-Means Clustering)
    # Áp dụng StandardScaler trước khi phân cụm trên 2 chiều: Khung giờ & Cự ly bay
    os.makedirs(MODELS_DIR, exist_ok=True)
    cluster_cols = ['hour', 'distance']
    kmeans_scaler = StandardScaler()
    cluster_features_scaled = kmeans_scaler.fit_transform(df[cluster_cols].fillna(df[cluster_cols].median()))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['flight_load_cluster'] = kmeans.fit_predict(cluster_features_scaled)

    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(kmeans_scaler, os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    print(f"[Task C] Đã phân cụm K-Means chuẩn hóa và lưu tại: {MODELS_DIR}")

    # 6. Mã hóa One-Hot cho Hãng bay và Sân bay
    # Giữ lại các hãng bay chính (chiếm > 95% thị phần)
    top_carriers = df['airline'].value_counts().nlargest(10).index
    df['carrier_clean'] = df['airline'].apply(lambda c: c if c in top_carriers else 'Other')
    carrier_dummies = pd.get_dummies(df['carrier_clean'], prefix='carrier', drop_first=True, dtype=int)

    origin_dummies = pd.get_dummies(df['origin'], prefix='origin', drop_first=True, dtype=int)

    top_dests = df['dest'].value_counts().nlargest(15).index
    df['dest_clean'] = df['dest'].apply(lambda d: d if d in top_dests else 'Other')
    dest_dummies = pd.get_dummies(df['dest_clean'], prefix='dest', drop_first=True, dtype=int)

    df = pd.concat([df, carrier_dummies, origin_dummies, dest_dummies], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Task C - Hoàn thành] Đã tạo thành công {df.shape[1]} đặc trưng cho {len(df)} chuyến bay THẬT vào: {output_path}")
    return df

if __name__ == "__main__":
    build_features()
