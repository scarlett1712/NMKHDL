"""
Module: feature_engineering.py
Giai đoạn C: Kỹ nghệ đặc trưng (Feature Engineering) trên dữ liệu thực tế.
- Trích xuất thuộc tính lịch biểu: Khung giờ, Ngày trong tuần, Tháng, Ngày cao điểm bay (is_peak_day), Giờ cao điểm (is_peak_hours).
- Mã hóa chu kỳ lượng giác (Cyclical Sine/Cosine).
- Tính toán độ trễ chuỗi thời gian theo từng sân bay xuất phát (theo đúng lịch giờ, không theo số dòng):
  + network_delay_lag_1h: Mức trễ bình quân của sân bay ở giờ liền trước (t-1).
  + network_delay_rolling_3h: Trung bình mức trễ sân bay trong 3 giờ trước đó (t-1, t-2, t-3).
- Đặc trưng từ nguồn Web Crawler: dest_hub_tier, dest_enplanements_m (đã ghép ở bước tích hợp).
- Phân cụm K-Means cho điều kiện vận hành (chuẩn hóa StandardScaler, k=3).
- Mã hóa One-Hot cho Hãng hàng không, Sân bay xuất phát và Top sân bay đến.

CHỐNG RÒ RỈ: K-Means, StandardScaler của K-Means, median điền lag và danh sách top-k hãng/sân bay
đều chỉ được fit trên tập Train (quy ước chung tại src/preprocessing/splits.py).
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.preprocessing.splits import sort_chronologically, train_mask

MODELS_DIR = "models_saved"
KMEANS_COLS = ['hour', 'distance']


def _airport_congestion_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trễ trung bình theo giờ của từng sân bay xuất phát, dịch theo thời gian thực:
    lag_1h = giờ t-1, rolling_3h = trung bình các giờ t-1..t-3 (giờ không có chuyến nào bị bỏ qua).
    Tính riêng cho từng sân bay nên cửa sổ trượt không tràn sang sân bay khác.
    """
    hourly = df.groupby(['origin', 'time_hour'])['delay_minutes'].mean()
    parts = []
    for origin, s in hourly.groupby(level='origin'):
        s = s.droplevel('origin').asfreq('h')  # điền đủ mọi giờ (NaN nếu giờ đó không có chuyến)
        prev = s.shift(1)
        parts.append(pd.DataFrame({
            'origin': origin,
            'time_hour': s.index,
            'network_delay_lag_1h': prev.values,
            'network_delay_rolling_3h': prev.rolling(3, min_periods=1).mean().values,
        }))
    return pd.concat(parts, ignore_index=True)


def build_features(cleaned_path: str = "data/processed/cleaned_flight_data.csv",
                   output_path: str = "data/processed/featured_flight_dataset.csv") -> pd.DataFrame:
    print("[Task C] Bắt đầu quá trình kỹ nghệ đặc trưng trên dữ liệu THẬT...")
    df = pd.read_csv(cleaned_path)
    df['dep_timestamp'] = pd.to_datetime(df['dep_timestamp'])
    df = sort_chronologically(df)
    is_train = train_mask(df)

    # 1. Trích xuất thuộc tính lịch biểu
    df['hour'] = df['dep_timestamp'].dt.hour
    df['dayofweek'] = df['dep_timestamp'].dt.day_of_week  # 0: Thứ 2, ..., 6: Chủ Nhật
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

    # 3. Trễ chuỗi thời gian theo sân bay xuất phát (Airport Congestion Lags)
    df['time_hour'] = df['dep_timestamp'].dt.floor('h')
    lags = _airport_congestion_lags(df)
    df = df.merge(lags, on=['origin', 'time_hour'], how='left')
    df = df.drop(columns=['time_hour'], errors='ignore')

    # Giờ đầu ngày / sau khoảng trống không có giờ trước -> điền median của Train
    lag_fill = float(df.loc[is_train, 'network_delay_lag_1h'].median())
    roll_fill = float(df.loc[is_train, 'network_delay_rolling_3h'].median())
    df['network_delay_lag_1h'] = df['network_delay_lag_1h'].fillna(lag_fill)
    df['network_delay_rolling_3h'] = df['network_delay_rolling_3h'].fillna(roll_fill)

    # 4. Đặc trưng rủi ro thời tiết thực tế
    df['is_rain'] = (df['origin_rain_mm'] > 0.1).astype(int)
    df['is_high_wind'] = (df['origin_gusts_kmh'] > 25.0).astype(int)
    df['is_low_visibility'] = (df['origin_visibility_km'] < 8.0).astype(int)

    # 5. Phân cụm K-Means (Khung giờ & Cự ly bay) - scaler và K-Means CHỈ fit trên Train
    os.makedirs(MODELS_DIR, exist_ok=True)
    kmeans_scaler = StandardScaler().fit(df.loc[is_train, KMEANS_COLS])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(kmeans_scaler.transform(df.loc[is_train, KMEANS_COLS]))
    df['flight_load_cluster'] = kmeans.predict(kmeans_scaler.transform(df[KMEANS_COLS]))

    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(kmeans_scaler, os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    print(f"[Task C] Đã fit K-Means (k=3) trên tập Train và lưu tại: {MODELS_DIR}")

    # 6. Mã hóa One-Hot - danh sách top hãng / top điểm đến lấy theo tần suất trên Train
    top_carriers = df.loc[is_train, 'airline'].value_counts().nlargest(10).index
    df['carrier_clean'] = df['airline'].where(df['airline'].isin(top_carriers), 'Other')
    carrier_dummies = pd.get_dummies(df['carrier_clean'], prefix='carrier', drop_first=True, dtype=int)

    origin_dummies = pd.get_dummies(df['origin'], prefix='origin', drop_first=True, dtype=int)

    top_dests = df.loc[is_train, 'dest'].value_counts().nlargest(15).index
    df['dest_clean'] = df['dest'].where(df['dest'].isin(top_dests), 'Other')
    dest_dummies = pd.get_dummies(df['dest_clean'], prefix='dest', drop_first=True, dtype=int)

    df = pd.concat([df, carrier_dummies, origin_dummies, dest_dummies], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Task C - Hoàn thành] Đã tạo thành công {df.shape[1]} cột cho {len(df)} chuyến bay THẬT vào: {output_path}")
    return df

if __name__ == "__main__":
    build_features()
